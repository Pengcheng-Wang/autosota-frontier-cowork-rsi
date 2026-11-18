import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_spike_repair_l1(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)
    Y_tr = tr["Y_train"].astype(np.float64)
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape
    D = L * S * C
    r_rank = 32

    # GPU SVD warm-start for Rank-32 Khatri-Rao PPCA
    Flat_tr = X_tr.reshape(N_tr, D)
    mu_d = np.mean(Flat_tr, axis=0, keepdims=True)
    sd_d = np.maximum(1e-4, np.std(Flat_tr, axis=0, keepdims=True))
    Z_tr = (Flat_tr - mu_d) / sd_d

    _, s_vals, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    U = Vt[:r_rank].T
    lam = np.maximum(1e-3, (s_vals[:r_rank] ** 2) / N_tr)
    nu = np.maximum(0.02, np.mean((Z_tr - (Z_tr @ U) @ U.T) ** 2, axis=0))
    Uw = U / np.sqrt(nu[:, None])
    KR = (Uw[:, :, None] * Uw[:, None, :]).reshape(D, r_rank * r_rank)
    prior_diag = np.diag(1.0 / lam)
    U_div_nu = U / nu[:, None]

    # Sensorwise MAD scale for Laplace/Gaussian mixture spike posterior
    padded_tr = np.pad(X_tr, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    med3_tr = np.median(np.stack([padded_tr[:, k:k + L] for k in range(3)], axis=0), axis=0)
    sig_s = np.maximum(0.08, 1.4826 * np.median(np.abs(X_tr - med3_tr), axis=(0, 1), keepdims=True))

    def spike_repair(X_in):
        B_sz = len(X_in)
        drop_s = (np.sum(np.abs(X_in) < 1e-7, axis=(1, 3)) >= 30)
        pad = np.pad(X_in, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
        m3 = np.median(np.stack([pad[:, k:k + L] for k in range(3)], axis=0), axis=0)

        # Pass 1: mask obvious dropouts + >3*sig_s temporal jumps -> Rank-32 PPCA L_hat
        m0 = ((~drop_s[:, None, :, None]) & (np.abs(X_in - m3) < 3.0 * sig_s)).reshape(B_sz, D).astype(np.float64)
        Z_in = (X_in.reshape(B_sz, D) - mu_d) / sd_d
        A0 = (m0 @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c0 = np.linalg.solve(A0, ((m0 * Z_in) @ U_div_nu)[:, :, None]).squeeze(-1)
        L_hat = ((c0 @ U.T) * sd_d + mu_d).reshape(B_sz, L, S, C)

        # Laplace(b=3.2) vs Gaussian(sig_s) spike posterior pi on e = X_in - L_hat!
        e = X_in - L_hat
        b_lap = 3.2
        rho_sp = 0.035
        log_lap = np.log(rho_sp / (2.0 * b_lap)) - np.abs(e) / b_lap
        log_gau = np.log((1.0 - rho_sp) / (np.sqrt(2.0 * np.pi) * sig_s)) - 0.5 * ((e / sig_s) ** 2)
        pi_sp = 1.0 / (1.0 + np.exp(np.clip(log_gau - log_lap, -30.0, 30.0)))
        pi_sp = np.where(drop_s[:, None, :, None], 1.0, pi_sp)

        # Soft precision weights (1 - pi_sp) in Pass-2 Khatri-Rao PPCA solve!
        w_obs = (1.0 - pi_sp).reshape(B_sz, D)
        A1 = (w_obs @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c1 = np.linalg.solve(A1, ((w_obs * Z_in) @ U_div_nu)[:, :, None]).squeeze(-1)
        L_hat2 = ((c1 @ U.T) * sd_d + mu_d).reshape(B_sz, L, S, C)
        X_tilde = (1.0 - pi_sp) * X_in + pi_sp * L_hat2
        return X_tilde, L_hat2, c1, pi_sp, drop_s.astype(np.float64)

    corr = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T)
    np.fill_diagonal(corr, -2.0)
    top8 = np.argsort(-corr, axis=1)[:, :8]

    rng = np.random.RandomState(5040)
    X_tr_c = np.where(rng.rand(N_tr, 1, S, 1) < 0.12, 0.0, X_tr)
    X_tr_c = X_tr_c + (rng.rand(*X_tr.shape) < 0.03) * rng.normal(0.0, 4.5, size=X_tr.shape)
    Xt_tr, Lh_tr, Cl_tr, Pi_tr, Df_tr = spike_repair(X_tr_c)

    # Single-pass weighted ridge per sensor (replaces slow IRLS loop!)
    def s_feats(Xt, Cl, Df, s_i):
        B_sz = len(Xt)
        return np.hstack([
            Xt[:, :, s_i, :].reshape(B_sz, 36),
            Xt[:, -1, s_i, :],
            Xt[:, -4:, top8[s_i], 0].reshape(B_sz, 32),
            Cl,
            Df[:, s_i:s_i + 1],
            np.ones((B_sz, 1)),
        ])

    dp = 36 + 3 + 32 + 32 + 2
    W_r = np.zeros((S, dp, 12), dtype=np.float64)
    reg = 4.5 * np.eye(dp); reg[-1, -1] = 1e-6
    Yr_tr = np.empty((N_tr, L, S), dtype=np.float32)
    for s_i in range(S):
        Phi = s_feats(Xt_tr, Cl_tr, Df_tr, s_i)
        # Weight clean low-distortion training windows slightly higher
        w_row = 1.0 - 0.5 * np.mean(Pi_tr[:, :, s_i, 0], axis=1)
        Pw = Phi.T * w_row[None, :]
        W_r[s_i] = np.linalg.solve(Pw @ Phi + reg, Pw @ Y_tr[:, :, s_i])
        Yr_tr[:, :, s_i] = (Phi @ W_r[s_i]).astype(np.float32)

    # GPU 3x192 GELU L1 Residual-Net with 16-d learned sensor embeddings!
    def build_net_in(Xt, Lh, Yr, Df):
        B_sz = len(Xt)
        loc = Xt.transpose(0, 2, 1, 3).reshape(B_sz, S, 36)
        lh_f = Lh[:, :, :, 0].transpose(0, 2, 1)  # [B, 96, 12]
        yr_s = Yr.transpose(0, 2, 1)              # [B, 96, 12]
        nb8 = Xt[:, -4:, top8, 0].transpose(0, 2, 1, 3).reshape(B_sz, S, 32)  # [B, 96, 32]
        return np.concatenate([yr_s, lh_f, loc, nb8, Df[:, :, None]], axis=2).astype(np.float32)  # 93 dims

    class SpikeResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Parameter(torch.randn(1, S, 16) * 0.02)
            self.net = nn.Sequential(
                nn.Linear(93 + 16, 192),
                nn.GELU(),
                nn.Linear(192, 192),
                nn.GELU(),
                nn.Linear(192, 12),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x):
            e = self.emb.expand(x.shape[0], -1, -1)
            return self.net(torch.cat([x, e], dim=-1))

    torch.manual_seed(404)
    model = SpikeResNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-2)
    Fin_tr = torch.tensor(build_net_in(Xt_tr, Lh_tr, Yr_tr, Df_tr), device=DEVICE)
    Ybase_t = torch.tensor(Yr_tr.transpose(0, 2, 1), device=DEVICE)
    Ytrue_t = torch.tensor(Y_tr.transpose(0, 2, 1).astype(np.float32), device=DEVICE)

    for _ in range(45):
        perm = torch.randperm(N_tr, device=DEVICE)
        for b0 in range(0, N_tr, 256):
            sl = perm[b0:b0 + 256]
            delta = torch.clamp(model(Fin_tr[sl]), -3.0, 3.0)
            loss = F.l1_loss(Ybase_t[sl] + delta, Ytrue_t[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    outs = []
    with torch.no_grad():
        for it in items:
            bi = int(it["block_idx"])
            Xt_e, Lh_e, Cl_e, _, Df_e = spike_repair(X_blocks[bi])
            B_sz = len(Xt_e)
            Yr_e = np.empty((B_sz, L, S), dtype=np.float32)
            for s_i in range(S):
                Yr_e[:, :, s_i] = (s_feats(Xt_e, Cl_e, Df_e, s_i) @ W_r[s_i]).astype(np.float32)
            Fin_e = torch.tensor(build_net_in(Xt_e, Lh_e, Yr_e, Df_e), device=DEVICE)
            Yb_et = torch.tensor(Yr_e.transpose(0, 2, 1), device=DEVICE)
            Yn = (Yb_et + torch.clamp(model(Fin_e), -3.0, 3.0)).cpu().numpy().transpose(0, 2, 1)
            outs.append({"id": it["id"], "prediction": (Yn * flow_std + flow_mean).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_spike_repair_l1(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
