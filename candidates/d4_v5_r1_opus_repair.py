import argparse, json
import numpy as np

def run_repair(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)   # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float64)   # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape
    D = L * S * C  # 3456
    r_rank = 32

    # 1) Per-(time, sensor, channel) z-score on clean train + Rank-32 SVD PPCA
    Flat_tr = X_tr.reshape(N_tr, D)
    mu_d = np.mean(Flat_tr, axis=0, keepdims=True)
    sd_d = np.maximum(1e-4, np.std(Flat_tr, axis=0, keepdims=True))
    Z_tr = (Flat_tr - mu_d) / sd_d  # [N_tr, 3456]

    # Fast economy SVD on Z_tr
    _, s_vals, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    U = Vt[:r_rank].T  # [3456, 32]
    lam = np.maximum(1e-3, (s_vals[:r_rank] ** 2) / N_tr)  # [32]
    Z_rec = (Z_tr @ U) @ U.T
    nu = np.maximum(0.02, np.mean((Z_tr - Z_rec) ** 2, axis=0))  # [3456]

    # Precompute Khatri-Rao K = [vec(u_i u_i^T) / nu_i] in R^{3456 x 1024} -> 1 GEMM gives all B 32x32 precision matrices!
    U_weighted = U / np.sqrt(nu[:, None])
    KR = (U_weighted[:, :, None] * U_weighted[:, None, :]).reshape(D, r_rank * r_rank)  # [3456, 1024]
    prior_diag = np.diag(1.0 / lam)  # [32, 32]
    U_div_nu = U / nu[:, None]  # [3456, 32]

    def repair_batch(X_in):
        # X_in: [B, 12, 96, 3]
        B_sz = len(X_in)
        # Detect sensor-block dropout (>= 30/36 entries exactly 0 in X_in!)
        zero_cnt = np.sum(np.abs(X_in) < 1e-7, axis=(1, 3))  # [B, 96]
        drop_sens = zero_cnt >= 30  # [B, 96]

        # 5-point running temporal median along L=12
        padded = np.pad(X_in, ((0, 0), (2, 2), (0, 0), (0, 0)), mode="edge")
        win5 = np.stack([padded[:, k:k + L] for k in range(5)], axis=0)
        t_med = np.median(win5, axis=0)
        spike1 = np.abs(X_in - t_med) > 0.85

        M_mask = (~drop_sens[:, None, :, None]) & (~spike1)
        M_flat = M_mask.reshape(B_sz, D).astype(np.float64)
        Z_in = (X_in.reshape(B_sz, D) - mu_d) / sd_d

        # Pass 1: 1-GEMM batched 32x32 posterior solve!
        A_batch = (M_flat @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        rhs = (M_flat * Z_in) @ U_div_nu  # [B, 32]
        c_lat = np.linalg.solve(A_batch, rhs[:, :, None]).squeeze(-1)  # [B, 32]
        Z_p1 = c_lat @ U.T

        # Pass 2: re-flag > 3.8 * sqrt(nu) outliers against low-rank posterior & re-solve!
        spike2 = np.abs(Z_in - Z_p1) > (3.8 * np.sqrt(nu[None, :]))
        M_flat2 = M_flat * (~spike2).astype(np.float64)
        A_batch2 = (M_flat2 @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        rhs2 = (M_flat2 * Z_in) @ U_div_nu
        c_lat2 = np.linalg.solve(A_batch2, rhs2[:, :, None]).squeeze(-1)
        Z_p2 = c_lat2 @ U.T

        Z_hat = M_flat2 * Z_in + (1.0 - M_flat2) * Z_p2
        X_rep = (Z_hat * sd_d + mu_d).reshape(B_sz, L, S, C)
        return X_rep, c_lat2, drop_sens.astype(np.float64)

    # 2) Top-5 correlated peer sensors on clean flow
    F_clean = X_tr[:, :, :, 0].reshape(-1, S)
    corr = np.corrcoef(F_clean.T)
    np.fill_diagonal(corr, -2.0)
    top5 = np.argsort(-corr, axis=1)[:, :5]  # [96, 5]

    # 3) Channel-matched training corruption (12% block dropout + 3% N(0, 4.5^2) spikes) + Repair!
    rng = np.random.RandomState(504)
    X_tr_corr = X_tr.copy()
    dm = rng.rand(N_tr, 1, S, 1) < 0.12
    X_tr_corr = np.where(dm, 0.0, X_tr_corr)
    sm = rng.rand(*X_tr_corr.shape) < 0.03
    X_tr_corr = X_tr_corr + sm * rng.normal(0.0, 4.5, size=X_tr_corr.shape)

    X_tr_rep, C_tr_lat, D_tr_flag = repair_batch(X_tr_corr)

    def make_sensor_features(X_r, C_lat, D_fl, s_idx):
        # Local repaired [12*3=36] + last-step anchor [3] + 5 peer flows [60] + latent c [32] + drop_flag [1] + 1 = 133 dims!
        B_sz = len(X_r)
        loc = X_r[:, :, s_idx, :].reshape(B_sz, 36)
        last3 = X_r[:, -1, s_idx, :]
        peers = X_r[:, :, top5[s_idx], 0].reshape(B_sz, 60)
        df_s = D_fl[:, s_idx:s_idx + 1]
        return np.hstack([loc, last3, peers, C_lat, df_s, np.ones((B_sz, 1))])

    # Fit 96 per-sensor ridges with 2 IRLS-LAD (MAE-matched!) reweighting passes
    W_sens = np.zeros((S, 133, 12), dtype=np.float64)
    reg_eye = 4.0 * np.eye(133, dtype=np.float64)
    reg_eye[-1, -1] = 1e-6
    for s_i in range(S):
        Phi_s = make_sensor_features(X_tr_rep, C_tr_lat, D_tr_flag, s_i)  # [N_tr, 133]
        Y_s = Y_tr[:, :, s_i]  # [N_tr, 12]
        W_cur = np.linalg.solve(Phi_s.T @ Phi_s + reg_eye, Phi_s.T @ Y_s)
        for _ in range(2):
            res_mae = np.mean(np.abs(Y_s - Phi_s @ W_cur), axis=1)
            w_irls = 1.0 / np.maximum(0.05, res_mae)
            w_irls /= np.mean(w_irls)
            Phi_w = Phi_s.T * w_irls[None, :]
            W_cur = np.linalg.solve(Phi_w @ Phi_s + reg_eye, Phi_w @ Y_s)
        W_sens[s_i] = W_cur

    outs = []
    for it in items:
        bi = int(it["block_idx"])
        X_r, C_lat, D_fl = repair_batch(X_blocks[bi])
        B_sz = len(X_r)
        Y_pred_norm = np.empty((B_sz, L, S), dtype=np.float64)
        for s_i in range(S):
            Phi_e = make_sensor_features(X_r, C_lat, D_fl, s_i)
            Y_pred_norm[:, :, s_i] = Phi_e @ W_sens[s_i]
        Y_raw = (Y_pred_norm * flow_std + flow_mean).astype(np.float32)
        outs.append({"id": it["id"], "prediction": Y_raw.tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_repair(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
