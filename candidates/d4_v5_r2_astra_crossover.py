import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_d4_r2_astra(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)   # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float64)   # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape
    D = L * S * C
    r_rank = 24

    # 1) Astra local 3-point median spike-odds shrinkage + Rank-24 1-Pass Khatri-Rao PPCA RePAIR!
    mu_d = np.mean(X_tr.reshape(N_tr, D), axis=0, keepdims=True)
    sd_d = np.maximum(1e-4, np.std(X_tr.reshape(N_tr, D), axis=0, keepdims=True))
    Z_tr = (X_tr.reshape(N_tr, D) - mu_d) / sd_d

    _, s_vals, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    U = Vt[:r_rank].T  # [3456, 24]
    lam = np.maximum(1e-3, (s_vals[:r_rank] ** 2) / N_tr)
    nu = np.maximum(0.02, np.mean((Z_tr - (Z_tr @ U) @ U.T) ** 2, axis=0))
    Uw = U / np.sqrt(nu[:, None])
    KR = (Uw[:, :, None] * Uw[:, None, :]).reshape(D, r_rank * r_rank)  # [3456, 576] -> ultra-fast!
    prior_diag = np.diag(1.0 / lam)
    U_div_nu = U / nu[:, None]

    def local_3med(X_in):
        out = np.empty_like(X_in)
        out[:, 0] = np.median(X_in[:, [0, 1, 2]], axis=1)
        out[:, -1] = np.median(X_in[:, [-3, -2, -1]], axis=1)
        for t in range(1, L - 1):
            out[:, t] = np.median(X_in[:, [t - 1, t, t + 1]], axis=1)
        return out

    res_tr = X_tr - local_3med(X_tr)
    var_clean = np.maximum(0.008, np.mean(res_tr ** 2, axis=(0, 1), keepdims=True))
    var_tot = var_clean + 4.5 ** 2

    def fast_hybrid_repair(X_in):
        B_sz = len(X_in)
        drop_s = (np.sum(np.abs(X_in) < 1e-7, axis=(1, 3)) >= 30)  # [B, 96]
        m_loc = local_3med(X_in)
        e = X_in - m_loc
        # Posterior spike probability
        lp1 = np.log(0.03) - 0.5 * np.log(var_tot) - 0.5 * (e ** 2) / var_tot
        lp0 = np.log(0.97) - 0.5 * np.log(var_clean) - 0.5 * (e ** 2) / var_clean
        q_sp = 1.0 / (1.0 + np.exp(np.clip(lp0 - lp1, -30.0, 30.0)))
        X_soft = X_in - q_sp * e * ((4.5 ** 2) / var_tot)

        # Hard-mask high-probability spikes (q > 0.35) AND dropped sensors for 1-GEMM Rank-24 PPCA Kriging!
        obs_mask = (~drop_s[:, None, :, None]) & (q_sp < 0.35)
        M_flat = obs_mask.reshape(B_sz, D).astype(np.float64)
        Z_soft = (X_soft.reshape(B_sz, D) - mu_d) / sd_d

        A_b = (M_flat @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c_lat = np.linalg.solve(A_b, ((M_flat * Z_soft) @ U_div_nu)[:, :, None]).squeeze(-1)
        Z_ppca = c_lat @ U.T
        # Blend soft-shrunk observation with PPCA posterior mean weighted by (1 - q_sp)!
        w_keep = M_flat * (1.0 - q_sp.reshape(B_sz, D))
        Z_fin = w_keep * Z_soft + (1.0 - w_keep) * Z_ppca
        X_rep = (Z_fin * sd_d + mu_d).reshape(B_sz, L, S, C)
        return X_rep, c_lat, drop_s.astype(np.float64)

    # Top-6 correlated sensors
    corr = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T)
    np.fill_diagonal(corr, -2.0)
    top6 = np.argsort(-corr, axis=1)[:, :6]

    # Corrupt training windows + fast 1-pass hybrid repair (~1.5s!)
    rng = np.random.RandomState(54)
    X_tr_c = np.where(rng.rand(N_tr, 1, S, 1) < 0.12, 0.0, X_tr)
    X_tr_c = X_tr_c + (rng.rand(*X_tr.shape) < 0.03) * rng.normal(0.0, 4.5, size=X_tr.shape)
    Xr_tr, Cl_tr, Df_tr = fast_hybrid_repair(X_tr_c)

    # 1-Shot per-sensor Ridge Base Forecast (no slow IRLS loops -> ~0.3s!)
    def make_s_phi(Xr, Cl, Df, s_i):
        B_sz = len(Xr)
        loc = Xr[:, :, s_i, :].reshape(B_sz, 36)
        peers = Xr[:, :, top6[s_i], 0].reshape(B_sz, 72)
        return np.hstack([loc, Xr[:, -1, s_i, :], peers, Cl, Df[:, s_i:s_i + 1], np.ones((B_sz, 1))])

    d_p = 36 + 3 + 72 + r_rank + 2
    W_base = np.zeros((S, d_p, 12), dtype=np.float64)
    reg = 5.0 * np.eye(d_p); reg[-1, -1] = 1e-6
    Y_base_tr = np.empty((N_tr, L, S), dtype=np.float32)
    for s_i in range(S):
        Phi_s = make_s_phi(Xr_tr, Cl_tr, Df_tr, s_i)
        W_base[s_i] = np.linalg.solve(Phi_s.T @ Phi_s + reg, Phi_s.T @ Y_tr[:, :, s_i])
        Y_base_tr[:, :, s_i] = (Phi_s @ W_base[s_i]).astype(np.float32)

    # Shared Zero-Initialized GPU Residual-MLP trained with exact L1 (MAE) loss (~0.8s on A100 GPU!)
    def build_mlp_tensor(Xr, Cl, Df, Y_b):
        # Per (window, sensor) features: local [36] + peer mean [12] + latent [24] + base_pred [12] + drop_flag [1] = 85 dims!
        B_sz = len(Xr)
        loc = Xr.transpose(0, 2, 1, 3).reshape(B_sz, S, 36)
        peer_m = np.mean(Xr[:, :, top6, 0], axis=3).transpose(0, 2, 1)  # [B, 96, 12]
        lat_rep = np.repeat(Cl[:, None, :], S, axis=1)                   # [B, 96, 24]
        yb_s = Y_b.transpose(0, 2, 1)                                    # [B, 96, 12]
        df_s = Df[:, :, None]                                            # [B, 96, 1]
        return np.concatenate([loc, peer_m, lat_rep, yb_s, df_s], axis=2).astype(np.float32)

    torch.manual_seed(54)
    res_mlp = nn.Sequential(
        nn.Linear(85, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 12),
    ).to(DEVICE)
    nn.init.zeros_(res_mlp[-1].weight); nn.init.zeros_(res_mlp[-1].bias)
    sens_emb = nn.Parameter(torch.zeros(1, S, 12, device=DEVICE))
    opt = torch.optim.AdamW([*res_mlp.parameters(), sens_emb], lr=2.5e-3, weight_decay=1e-2)

    Ft_tr = torch.tensor(build_mlp_tensor(Xr_tr, Cl_tr, Df_tr, Y_base_tr), device=DEVICE)
    Yt_tr = torch.tensor(Y_tr.transpose(0, 2, 1).astype(np.float32), device=DEVICE)
    Yb_tr_t = torch.tensor(Y_base_tr.transpose(0, 2, 1), device=DEVICE)

    for _ in range(45):
        perm = torch.randperm(N_tr, device=DEVICE)
        for b0 in range(0, N_tr, 256):
            sl = perm[b0:b0 + 256]
            pred = Yb_tr_t[sl] + res_mlp(Ft_tr[sl]) + sens_emb
            loss = F.l1_loss(pred, Yt_tr[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    res_mlp.eval()

    outs = []
    with torch.no_grad():
        for it in items:
            bi = int(it["block_idx"])
            Xr_e, Cl_e, Df_e = fast_hybrid_repair(X_blocks[bi])
            B_sz = len(Xr_e)
            Yb_e = np.empty((B_sz, L, S), dtype=np.float32)
            for s_i in range(S):
                Yb_e[:, :, s_i] = (make_s_phi(Xr_e, Cl_e, Df_e, s_i) @ W_base[s_i]).astype(np.float32)
            Ft_e = torch.tensor(build_mlp_tensor(Xr_e, Cl_e, Df_e, Yb_e), device=DEVICE)
            Yb_et = torch.tensor(Yb_e.transpose(0, 2, 1), device=DEVICE)
            Yn_fin = (Yb_et + res_mlp(Ft_e) + sens_emb).cpu().numpy().transpose(0, 2, 1)
            outs.append({"id": it["id"], "prediction": (Yn_fin * flow_std + flow_mean).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_d4_r2_astra(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
