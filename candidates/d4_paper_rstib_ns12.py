import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VARIANT = 'paper_rstib_ns12'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RSTIBNet(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
        self.z_dim = z_dim

    def forward(self, x, n_sample=1):
        stats = self.enc(x)
        mu, logvar = stats[..., :self.z_dim], torch.clamp(stats[..., self.z_dim:], -8.0, 4.0)
        std = torch.exp(0.5 * logvar)
        kl = -0.5 * torch.mean(1.0 + logvar - mu ** 2 - std ** 2)
        if n_sample <= 1 and not self.training:
            return self.dec(mu), kl
        preds = []
        for _ in range(n_sample):
            eps = torch.randn_like(std)
            preds.append(self.dec(mu + std * eps))
        return torch.mean(torch.stack(preds, dim=0), dim=0), kl

def run_d4(train_npz, eval_npz, items):
    tr = np.load(train_npz)
    ev = np.load(eval_npz)
    X_tr = tr["X_train"].astype(np.float32)  # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float32)  # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float32)  # [16, B, 12, 96, 3]
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])
    N_tr, L, S, C = X_tr.shape

    if VARIANT == "base_huber_ridge":
        # Per-sensor clipped Ridge forecaster
        X_flat = np.clip(X_tr, -3.5, 3.5).transpose(0, 2, 1, 3).reshape(-1, L * C)  # [N_tr*S, 36]
        Y_flat = Y_tr.transpose(0, 2, 1).reshape(-1, 12)  # [N_tr*S, 12]
        Xb = np.hstack([X_flat, np.ones((len(X_flat), 1), dtype=np.float32)])
        W = np.linalg.solve(Xb.T @ Xb + 50.0 * np.eye(Xb.shape[1], dtype=np.float32), Xb.T @ Y_flat)
        outs = []
        for it in items:
            bi = int(it["block_idx"])
            Xe = np.clip(X_blocks[bi], -3.5, 3.5).transpose(0, 2, 1, 3)  # [B, S, 12, 3]
            B_sz = Xe.shape[0]
            Xef = Xe.reshape(-1, L * C)
            Xefb = np.hstack([Xef, np.ones((len(Xef), 1), dtype=np.float32)])
            pred_n = (Xefb @ W).reshape(B_sz, S, 12).transpose(0, 2, 1)
            outs.append({"id": it["id"], "prediction": (pred_n * flow_std + flow_mean).tolist()})
        return outs

    if VARIANT in ("paper_rstib_ns12", "autosota_rstib_ns50", "comp_spatial_mlp"):
        torch.manual_seed(90)
        if VARIANT == "comp_spatial_mlp":
            # Joint cross-sensor bottleneck MLP
            in_d = L * S * C
            out_d = 12 * S
            net = nn.Sequential(nn.Linear(in_d, 384), nn.SiLU(), nn.Dropout(0.15), nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, out_d)).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
            xt = torch.tensor(X_tr.reshape(N_tr, -1), device=DEVICE)
            yt = torch.tensor(Y_tr.reshape(N_tr, -1), device=DEVICE)
            for ep in range(75):
                net.train()
                perm = torch.randperm(N_tr, device=DEVICE)
                for b0 in range(0, N_tr, 128):
                    sl = perm[b0:b0 + 128]
                    # Mild input noise during training
                    xn = xt[sl]
                    loss = F.l1_loss(net(xn), yt[sl])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            outs = []
            with torch.no_grad():
                for it in items:
                    bi = int(it["block_idx"])
                    xb = torch.tensor(X_blocks[bi].reshape(len(X_blocks[bi]), -1), device=DEVICE)
                    pn = net(xb).view(-1, 12, S).cpu().numpy()
                    outs.append({"id": it["id"], "prediction": (pn * flow_std + flow_mean).tolist()})
            return outs

        # RSTIB per-sensor + global context IB MLP
        # Note: AutoSOTA sets info_beta=0.0 AND n_sample=50 (+ CPU/repeated forward passes!)
        info_beta = 0.0 if VARIANT == "autosota_rstib_ns50" else 1e-3
        n_samp_eval = 50 if VARIANT == "autosota_rstib_ns50" else 12
        n_restarts = 4 if VARIANT == "autosota_rstib_ns50" else 1

        models = []
        for r_idx in range(n_restarts):
            torch.manual_seed(90 + r_idx * 17)
            net = RSTIBNet(L * C + L, 12, z_dim=48).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            # Per-sensor history [N_tr, S, L*C] + global highway mean flow [N_tr, S, L]
            glob_tr = np.mean(X_tr[:, :, :, 0], axis=2, keepdims=True).transpose(0, 2, 1)  # [N_tr, 1, L]
            glob_tr = np.repeat(glob_tr, S, axis=1)
            loc_tr = X_tr.transpose(0, 2, 1, 3).reshape(N_tr, S, L * C)
            xt = torch.tensor(np.concatenate([loc_tr, glob_tr], axis=2), device=DEVICE)  # [N_tr, S, 48]
            yt = torch.tensor(Y_tr.transpose(0, 2, 1), device=DEVICE)  # [N_tr, S, 12]
            for ep in range(55):
                net.train()
                perm = torch.randperm(N_tr, device=DEVICE)
                for b0 in range(0, N_tr, 128):
                    sl = perm[b0:b0 + 128]
                    pred, kl = net(xt[sl], n_sample=2)
                    loss = F.l1_loss(pred, yt[sl]) + info_beta * kl
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            models.append(net)

        outs = []
        with torch.no_grad():
            for it in items:
                bi = int(it["block_idx"])
                Xb = X_blocks[bi]
                B_sz = len(Xb)
                glob_e = np.repeat(np.mean(Xb[:, :, :, 0], axis=2, keepdims=True).transpose(0, 2, 1), S, axis=1)
                loc_e = Xb.transpose(0, 2, 1, 3).reshape(B_sz, S, L * C)
                xe = torch.tensor(np.concatenate([loc_e, glob_e], axis=2), device=DEVICE)
                ens_p = []
                for m in models:
                    p_m, _ = m(xe, n_sample=n_samp_eval)
                    ens_p.append(p_m.cpu().numpy())
                pn = np.mean(ens_p, axis=0).transpose(0, 2, 1)
                outs.append({"id": it["id"], "prediction": (pn * flow_std + flow_mean).tolist()})
        return outs

    if VARIANT == "decision_mad_top8_affine":
        # Proposer D: Median/MAD clipping + top-8 correlated neighbor imputed mean + sensor-bias MAE forecaster
        med = np.median(X_tr, axis=0, keepdims=True)  # [1, 12, 96, 3]
        mad = np.maximum(1e-3, np.median(np.abs(X_tr - med), axis=0, keepdims=True))
        # Sensor flow correlation matrix [96, 96]
        F_mat = X_tr[:, -1, :, 0]  # [N_tr, 96]
        corr = np.corrcoef(F_mat.T)
        np.fill_diagonal(corr, -2.0)
        top8 = np.argsort(-corr, axis=1)[:, :8]  # [96, 8]

        def clean_and_nb(X_in):
            # Detect exact zero dropout or >4.5 MAD spikes
            bad = (X_in == 0.0) | (np.abs(X_in - med) > 4.5 * mad)
            Xc = np.where(bad, med, X_in)
            Xc = np.clip(Xc, med - 6.0 * mad, med + 6.0 * mad)
            # Neighbor mean [B, 12, 96, 3]
            X_nb = np.mean(Xc[:, :, top8, :], axis=3)
            return Xc, X_nb

        torch.manual_seed(91)
        W_shared = nn.Linear(72, 12).to(DEVICE)
        b_sens = nn.Parameter(torch.zeros(1, S, 12, device=DEVICE))
        opt = torch.optim.Adam([*W_shared.parameters(), b_sens], lr=3e-3)

        rng = np.random.RandomState(91)
        dm0 = rng.rand(N_tr, 1, S, 1) < 0.15
        Xc_tr, Xnb_tr = clean_and_nb(np.where(dm0, 0.0, X_tr))
        feat_tr = np.concatenate([Xc_tr.transpose(0, 2, 1, 3).reshape(N_tr, S, 36), Xnb_tr.transpose(0, 2, 1, 3).reshape(N_tr, S, 36)], axis=2)
        ft_all = torch.tensor(feat_tr, device=DEVICE)
        yt_all = torch.tensor(Y_tr.transpose(0, 2, 1), device=DEVICE)

        for ep in range(75):
            perm = torch.randperm(N_tr, device=DEVICE)
            for b0 in range(0, N_tr, 128):
                sl = perm[b0:b0 + 128]
                pred = W_shared(ft_all[sl]) + b_sens
                loss = F.l1_loss(pred, yt_all[sl])
                opt.zero_grad(); loss.backward(); opt.step()

        outs = []
        with torch.no_grad():
            for it in items:
                bi = int(it["block_idx"])
                Xc, Xnb = clean_and_nb(X_blocks[bi])
                B_sz = len(Xc)
                feat = np.concatenate([Xc.transpose(0, 2, 1, 3).reshape(B_sz, S, 36), Xnb.transpose(0, 2, 1, 3).reshape(B_sz, S, 36)], axis=2)
                pn = (W_shared(torch.tensor(feat, device=DEVICE)) + b_sens).cpu().numpy().transpose(0, 2, 1)
                outs.append({"id": it["id"], "prediction": (pn * flow_std + flow_mean).tolist()})
        return outs

    if VARIANT == "coord_kriging_kalman_residual":
        # Proposer A: Exact Spatial Ridge-Kriging Reconstruction of Corrupted/Dropped Sensors + Temporal-Median Spike Filter + Gated Spatial-Temporal Residual Net!
        med_s = np.median(X_tr, axis=(0, 1), keepdims=True)  # [1, 1, 96, 3]
        mad_s = np.maximum(1e-3, np.median(np.abs(X_tr - med_s), axis=(0, 1), keepdims=True))
        # Spatial covariance across 96 sensors per channel c in 0..2
        Sigma_c = []
        for c_i in range(C):
            Fl = X_tr[:, :, :, c_i].reshape(-1, S)
            Sigma_c.append((Fl.T @ Fl) / len(Fl) + 0.08 * np.eye(S, dtype=np.float32))

        def kriging_clean(X_arr):
            # X_arr: [B, 12, 96, 3]
            X_out = X_arr.copy()
            # 1) Detect sensor-wide dropout (all 12 steps == 0) AND point impulse spikes (|x - temporal_median| > 3.2)
            is_zero_drop = np.all(np.abs(X_out[:, :, :, 0]) < 1e-6, axis=1)  # [B, 96]
            t_med = np.median(X_out, axis=1, keepdims=True)
            is_spike = (np.abs(X_out - t_med) > 3.0) | (np.abs(X_out - med_s) > 4.2 * mad_s)
            X_out = np.where(is_spike, t_med, X_out)

            # 2) Reconstruct dropped sensors via closed-form spatial Kriging from observed highway sensors!
            for b_i in range(len(X_out)):
                miss = np.where(is_zero_drop[b_i])[0]
                if len(miss) > 0 and len(miss) < S:
                    obs = np.where(~is_zero_drop[b_i])[0]
                    for c_i in range(C):
                        Sg = Sigma_c[c_i]
                        W_krig = Sg[np.ix_(miss, obs)] @ np.linalg.inv(Sg[np.ix_(obs, obs)])  # [|M|, |O|]
                        X_out[b_i, :, miss, c_i] = W_krig @ X_out[b_i, :, obs, c_i]
            return X_out

        # Precompute spatial graph propagation matrix (row-normalized positive flow correlation)
        C_flow = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T).astype(np.float32)
        np.fill_diagonal(C_flow, 0.0)
        W_graph = np.maximum(0.0, C_flow) ** 2
        W_graph /= (np.sum(W_graph, axis=1, keepdims=True) + 1e-6)
        W_g_t = torch.tensor(W_graph, device=DEVICE)

        class SpatialTemporalResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_skip = nn.Linear(L * C, 12)
                self.fc1 = nn.Linear(L * C * 2 + S, 192)
                self.fc2 = nn.Linear(192, 12)
                self.sens_emb = nn.Embedding(S, S)
                nn.init.eye_(self.sens_emb.weight)

            def forward(self, x_clean):
                # x_clean: [B, 12, S, 3]
                B_sz = x_clean.shape[0]
                loc = x_clean.permute(0, 2, 1, 3).reshape(B_sz, S, L * C)
                nb = torch.einsum("ij,btjc->btic", W_g_t, x_clean).permute(0, 2, 1, 3).reshape(B_sz, S, L * C)
                se = self.sens_emb.weight.unsqueeze(0).expand(B_sz, -1, -1)
                h = F.silu(self.fc1(torch.cat([loc, nb, se], dim=-1)))
                out = self.lin_skip(loc) + self.fc2(h)
                return out.permute(0, 2, 1)  # [B, 12, S]

        torch.manual_seed(92)
        net = SpatialTemporalResNet().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=65)
        rng = np.random.RandomState(92)
        dm_tr = rng.rand(N_tr, 1, S, 1) < 0.12
        X_tr_imp = kriging_clean(np.where(dm_tr, 0.0, X_tr))
        xt_all = torch.tensor(X_tr_imp, device=DEVICE)
        yt_all = torch.tensor(Y_tr, device=DEVICE)

        for ep in range(65):
            net.train()
            perm = torch.randperm(N_tr, device=DEVICE)
            for b0 in range(0, N_tr, 128):
                sl = perm[b0:b0 + 128]
                loss = F.l1_loss(net(xt_all[sl]), yt_all[sl])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        net.eval()
        outs = []
        with torch.no_grad():
            for it in items:
                bi = int(it["block_idx"])
                X_clean = kriging_clean(X_blocks[bi])
                pn = net(torch.tensor(X_clean, device=DEVICE)).cpu().numpy()
                outs.append({"id": it["id"], "prediction": (pn * flow_std + flow_mean).tolist()})
        return outs

    raise ValueError(VARIANT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_d4(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
