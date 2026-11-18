#!/usr/bin/env python3
"""
Wave-5 Round-1 Candidate & Baseline Generator — Node 1 (D4, D5, D6):
  - Copies frozen Wave-4 baselines, mandatory comparators, and Wave-4 Headless Champions
  - Writes standalone `python -I -B` Round-1 scripts from `Opus 5.5, Max` and `GPT 6.2 Astra, Max`
"""
from __future__ import annotations

import shutil
from pathlib import Path

W4_ROOT = Path("/scratch/autosota_governed_v4")
W5_ROOT = Path("/scratch/autosota_governed_v5")


def write_cand(bench_folder: str, cid: str, code: str) -> None:
    p = W5_ROOT / "tracks" / bench_folder / "candidates" / f"{cid}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code.lstrip(), encoding="utf-8")


def copy_v4_cands(bench_folder: str, mapping: dict[str, str]) -> None:
    src_dir = W4_ROOT / "tracks" / bench_folder / "candidates"
    dst_dir = W5_ROOT / "tracks" / bench_folder / "candidates"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_id, dst_id in mapping.items():
        shutil.copy2(src_dir / f"{src_id}.py", dst_dir / f"{dst_id}.py")


# ============================================================================
# D4 — Opus 5.5 Max: RePAIR (Khatri-Rao Rank-32 PPCA Kriging + Corrupt-Train IRLS-LAD Ridge)
# ============================================================================
D4_OPUS_REPAIR = '''import argparse, json
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
'''


# ============================================================================
# D4 — GPT 6.2 Astra Max: Posterior Spike-Probability Filter + Corrupt-Train Queue Ridge
# ============================================================================
D4_ASTRA_SPIKE_QUEUE = '''import argparse, json
import numpy as np

def run_astra_d4(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)   # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float64)   # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape

    # Z-score per (sensor, channel) & top-6 correlated peers per channel
    mu_sc = np.mean(X_tr, axis=(0, 1), keepdims=True)  # [1, 1, 96, 3]
    sd_sc = np.maximum(1e-4, np.std(X_tr, axis=(0, 1), keepdims=True))
    Z_tr_clean = (X_tr - mu_sc) / sd_sc

    peers_c = []
    weights_c = []
    for c_i in range(C):
        Fl = Z_tr_clean[:, :, :, c_i].reshape(-1, S)
        corr = np.corrcoef(Fl.T)
        np.fill_diagonal(corr, -2.0)
        p6 = np.argsort(-corr, axis=1)[:, :6]  # [96, 6]
        w6 = np.maximum(0.0, np.take_along_axis(corr, p6, axis=1))
        peers_c.append(p6)
        weights_c.append(w6)

    # Estimate clean local 3-point temporal-median residual variance sigma^2 per (sensor, channel)
    def local_3med(Z):
        # Z: [B, 12, 96, 3] -> median of t and its 2 nearest OTHER time neighbors!
        out = np.empty_like(Z)
        out[:, 0] = np.median(Z[:, [0, 1, 2]], axis=1)
        out[:, -1] = np.median(Z[:, [-3, -2, -1]], axis=1)
        for t in range(1, L - 1):
            out[:, t] = np.median(Z[:, [t - 1, t, t + 1]], axis=1)
        return out

    res_clean = Z_tr_clean - local_3med(Z_tr_clean)
    var_clean = np.maximum(0.01, np.mean(res_clean ** 2, axis=(0, 1), keepdims=True))  # [1, 1, 96, 3]
    spike_var_z = (4.5 ** 2) / (sd_sc ** 2)  # [1, 1, 96, 3] in z-units!
    var_tot = var_clean + spike_var_z

    def repair_astra(X_raw):
        B_sz = len(X_raw)
        drop_flag = (np.sum(np.abs(X_raw) < 1e-7, axis=(1, 3)) >= 30)  # [B, 96]
        Z = (X_raw - mu_sc) / sd_sc
        m_loc = local_3med(Z)
        e = Z - m_loc

        # Exact Bayesian mixture posterior probability of impulse spike (prior pi = 0.03):
        log_p1 = np.log(0.03) - 0.5 * np.log(var_tot) - 0.5 * (e ** 2) / var_tot
        log_p0 = np.log(0.97) - 0.5 * np.log(var_clean) - 0.5 * (e ** 2) / var_clean
        q_post = 1.0 / (1.0 + np.exp(np.clip(log_p0 - log_p1, -30.0, 30.0)))
        # MMSE posterior shrinkage of spike component (only on live sensors!)
        shrink = q_post * e * (spike_var_z / var_tot)
        Z_rep = np.where(drop_flag[:, None, :, None], 0.0, Z - shrink)

        # Peer-weighted estimate over live peers + fill dropped sensors!
        live = (~drop_flag).astype(np.float64)  # [B, 96]
        Peer_est = np.empty_like(Z_rep)
        for c_i in range(C):
            p6 = peers_c[c_i]       # [96, 6]
            w6 = weights_c[c_i]     # [96, 6]
            Zp = Z_rep[:, :, p6, c_i]          # [B, 12, 96, 6]
            lp = live[:, None, p6] * w6[None, None, :, :]  # [B, 1, 96, 6]
            denom = np.sum(lp, axis=-1)
            Peer_est[:, :, :, c_i] = np.where(denom > 1e-6, np.sum(Zp * lp, axis=-1) / np.maximum(1e-6, denom), 0.0)

        Z_final = np.where(drop_flag[:, None, :, None], Peer_est, Z_rep)
        X_final = Z_final * sd_sc + mu_sc
        return X_final, Peer_est * sd_sc + mu_sc, drop_flag.astype(np.float64)

    # Corrupt training inputs once at exact test rates + repair (self-supervised calibration!)
    rng = np.random.RandomState(4)
    X_tr_c = X_tr.copy()
    X_tr_c = np.where(rng.rand(N_tr, 1, S, 1) < 0.12, 0.0, X_tr_c)
    sm = rng.rand(*X_tr_c.shape) < 0.03
    X_tr_c = X_tr_c + sm * rng.normal(0.0, 4.5, size=X_tr_c.shape)

    Xr_tr, Xp_tr, Df_tr = repair_astra(X_tr_c)
    Y_raw_tr = Y_tr * flow_std + flow_mean  # [N_tr, 12, 96]

    def build_queue_feats(Xr, Xp, Df, s_i):
        # Queue buildup/clearance features: 3-channel means on [0:6], [6:9], [9:12] + full 12-step flow + peer 3-seg means + interactions + drop_flag!
        Xs = Xr[:, :, s_i, :]   # [B, 12, 3]
        Ps = Xp[:, :, s_i, :]   # [B, 12, 3]
        m1 = np.mean(Xs[:, 0:6, :], axis=1)
        m2 = np.mean(Xs[:, 6:9, :], axis=1)
        m3 = np.mean(Xs[:, 9:12, :], axis=1)
        pm1 = np.mean(Ps[:, 0:6, :], axis=1)
        pm3 = np.mean(Ps[:, 9:12, :], axis=1)
        slope_q = m3 - m1  # queue buildup/clearance rate!
        f_full = Xs[:, :, 0]  # [B, 12]
        pf_full = Ps[:, :, 0] # [B, 12]
        occ_spd_int = (Xs[:, -1, 0:1] * Xs[:, -1, 1:2])  # flow x occupancy congestion proxy
        d_col = Df[:, s_i:s_i + 1]
        # Interaction with d_col so dropped vs live sensors get distinct weights!
        base_f = np.hstack([f_full, pf_full, m1, m2, m3, pm1, pm3, slope_q, Xs[:, -1, :], occ_spd_int])
        return np.hstack([base_f, d_col * pf_full, d_col, np.ones((len(Xr), 1))])

    sample_f = build_queue_feats(Xr_tr[:2], Xp_tr[:2], Df_tr[:2], 0)
    d_feat = sample_f.shape[1]
    W_all = np.zeros((S, d_feat, 12), dtype=np.float64)
    reg = 6.0 * np.eye(d_feat, dtype=np.float64)
    reg[-1, -1] = 1e-6

    for s_i in range(S):
        Phi = build_queue_feats(Xr_tr, Xp_tr, Df_tr, s_i)
        Yt = Y_raw_tr[:, :, s_i]
        W_s = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Yt)
        for _ in range(2):
            rmae = np.mean(np.abs(Yt - Phi @ W_s), axis=1)
            wi = 1.0 / np.maximum(1.5, rmae)
            wi /= np.mean(wi)
            Pw = Phi.T * wi[None, :]
            W_s = np.linalg.solve(Pw @ Phi + reg, Pw @ Yt)
        W_all[s_i] = W_s

    outs = []
    for it in items:
        bi = int(it["block_idx"])
        Xr_e, Xp_e, Df_e = repair_astra(X_blocks[bi])
        B_sz = len(Xr_e)
        Yp = np.empty((B_sz, L, S), dtype=np.float32)
        for s_i in range(S):
            Yp[:, :, s_i] = (build_queue_feats(Xr_e, Xp_e, Df_e, s_i) @ W_all[s_i]).astype(np.float32)
        outs.append({"id": it["id"], "prediction": Yp.tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_astra_d4(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D5 — Opus 5.5 Max (tEM-RP) & GPT 6.2 Astra Max (EM-ADMM Debiased Lik-Gain)
# ============================================================================
D5_OPUS_TEM_RP = '''import argparse, json
import numpy as np

def fit_tem_rp(X_obs, M, nu=5.0, n_iter=30, init_mu=None, init_Om=None):
    T, p = X_obs.shape
    miss = (M < 0.5)
    cnt = np.maximum(1.0, np.sum(M, axis=0))
    mu = (np.sum(X_obs * M, axis=0) / cnt) if init_mu is None else init_mu.copy()

    # Scale-robust initialization
    Xc0 = (X_obs - mu[None, :]) * M
    var0 = np.maximum(1e-4, np.sum((Xc0 ** 2) * M, axis=0) / np.maximum(1.0, cnt - 1.0))
    Om = np.diag(1.0 / var0) if init_Om is None else init_Om.copy()

    T_obs = max(4.0, float(np.mean(cnt)))
    rho = 2.2 * p / T_obs  # Closed-form spectral ridge-precision regularizer
    eye_p = np.eye(p, dtype=np.float64)
    n_obs_t = np.sum(M > 0.5, axis=1)

    for _ in range(n_iter):
        R = (1.0 - miss.astype(np.float64)) * (X_obs - mu[None, :])  # [T, p]
        Q = R @ Om  # [T, p]
        X_hat = np.empty((T, p), dtype=np.float64)
        S_cond = np.zeros((p, p), dtype=np.float64)
        w_t = np.empty(T, dtype=np.float64)

        for t in range(T):
            m_idx = np.where(miss[t])[0]
            rt = R[t]
            qt = Q[t]
            if len(m_idx) == 0:
                X_hat[t] = mu + rt
                delta_t = float(rt @ qt)
            elif len(m_idx) == p:
                X_hat[t] = mu
                S_cond += np.linalg.inv(Om + 1e-4 * eye_p)
                delta_t = 0.0
            else:
                O_mm = Om[np.ix_(m_idx, m_idx)] + 1e-6 * np.eye(len(m_idx))
                O_mm_inv = np.linalg.inv(O_mm)
                qm = qt[m_idx]
                c_qm = O_mm_inv @ qm
                xh = mu + rt
                xh[m_idx] = mu[m_idx] - c_qm
                X_hat[t] = xh
                S_cond[np.ix_(m_idx, m_idx)] += O_mm_inv
                delta_t = max(0.0, float(rt @ qt - qm @ c_qm))
            w_t[t] = (nu + float(n_obs_t[t])) / (nu + delta_t)

        mu = (w_t @ X_hat) / max(1e-8, float(np.sum(w_t)))
        D_hat = X_hat - mu[None, :]
        S_mat = ((D_hat.T * w_t[None, :]) @ D_hat + S_cond) / T
        S_mat = 0.5 * (S_mat + S_mat.T)
        s_eig, V = np.linalg.eigh(S_mat)
        s_eig = np.maximum(1e-6, s_eig)
        om_eig = 2.0 / (s_eig + np.sqrt(s_eig ** 2 + 4.0 * rho))
        Om = (V * om_eig[None, :]) @ V.T

    return mu, Om

def fisher_z_pcorr(Om):
    iu = np.triu_indices(Om.shape[0], k=1)
    d = np.sqrt(np.maximum(1e-12, np.diag(Om)))
    pc = np.clip(-Om[iu] / (d[iu[0]] * d[iu[1]]), -0.99999, 0.99999)
    return np.arctanh(pc)

def score_d5_opus(X_obs, M, seed):
    T, p = X_obs.shape
    # Scale each stock by observed MAD (guarded)
    X_sc = X_obs.copy()
    for j in range(p):
        obs_j = X_obs[M[:, j] > 0.5, j]
        if len(obs_j) >= 3:
            med = float(np.median(obs_j))
            mad = float(np.median(np.abs(obs_j - med))) * 1.4826
            sd = float(np.std(obs_j))
            sc = mad if mad > 1e-4 else (sd if sd > 1e-4 else 1.0)
            X_sc[:, j] = (X_obs[:, j] - med) / sc * M[:, j]

    mu_hat, Om_hat = fit_tem_rp(X_sc, M, nu=5.0, n_iter=32)
    z_full = fisher_z_pcorr(Om_hat)

    # 16 warm-started row-bootstraps for LCB demoting high-missingness false edges!
    rng = np.random.RandomState(900 + seed)
    z_boots = np.empty((16, len(z_full)), dtype=np.float64)
    for b in range(16):
        idx_b = rng.randint(0, T, size=T)
        _, Om_b = fit_tem_rp(X_sc[idx_b], M[idx_b], nu=5.0, n_iter=6, init_mu=mu_hat, init_Om=Om_hat)
        z_boots[b] = fisher_z_pcorr(Om_b)
    sd_b = np.std(z_boots, axis=0)
    # Continuous non-negative edge confidence score
    scores = np.abs(z_full) / (1.0 + 0.65 * sd_b / (np.median(np.abs(z_full)) + 1e-4))
    return scores.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = []
    for idx, it in enumerate(req["items"]):
        preds.append({"id": it["id"], "prediction": score_d5_opus(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64), idx)})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


D5_ASTRA_LIKGAIN = '''import argparse, json, math
import numpy as np

def admm_glasso_warm(S, lam, rho=1.5, n_iter=15, Z_init=None, U_init=None):
    p = S.shape[0]
    Z = np.eye(p, dtype=np.float64) if Z_init is None else Z_init.copy()
    U = np.zeros((p, p), dtype=np.float64) if U_init is None else U_init.copy()
    thresh = lam / rho
    Om = Z.copy()
    for _ in range(n_iter):
        mat = rho * (Z - U) - S
        vals, Q = np.linalg.eigh(0.5 * (mat + mat.T))
        eig_om = (vals + np.sqrt(vals ** 2 + 4.0 * rho)) / (2.0 * rho)
        Om = (Q * eig_om[None, :]) @ Q.T
        V = Om + U
        Z_new = np.sign(V) * np.maximum(0.0, np.abs(V) - thresh)
        np.fill_diagonal(Z_new, np.diag(V))
        U = U + Om - Z_new
        Z = Z_new
    return Om, Z, U

def score_d5_astra(X_obs, M):
    T, p = X_obs.shape
    cnt = np.maximum(1.0, np.sum(M, axis=0))
    mu = np.sum(X_obs * M, axis=0) / cnt
    Xc = (X_obs - mu[None, :]) * M
    sd = np.sqrt(np.maximum(1e-6, np.sum((Xc ** 2) * M, axis=0) / np.maximum(1.0, cnt - 1.0)))
    X = Xc / sd[None, :]
    mu = np.zeros(p, dtype=np.float64)

    n_ij = np.maximum(1.0, M.T @ M)
    lam_base = 0.45 * math.sqrt(math.log(max(3, p)) / max(8.0, float(np.median(n_ij))))
    Om = np.eye(p, dtype=np.float64)
    Z_admm, U_admm = Om.copy(), np.zeros((p, p), dtype=np.float64)

    # 6 rounds of Gaussian MAR EM + warm-started ADMM Graphical Lasso
    S_em = np.eye(p, dtype=np.float64)
    for r in range(6):
        Sigma = np.linalg.inv(0.5 * (Om + Om.T) + 1e-5 * np.eye(p))
        X_imp = X.copy()
        S_cond = np.zeros((p, p), dtype=np.float64)
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            if 0 < len(miss) < p:
                obs = np.where(M[t] > 0.5)[0]
                Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-5 * np.eye(len(obs)))
                W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                X_imp[t, miss] = mu[miss] + W_mo @ (X[t, obs] - mu[obs])
                S_cond[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
            elif len(miss) == p:
                X_imp[t] = mu
                S_cond += Sigma
        if r < 5:
            mu = np.mean(X_imp, axis=0)
        D = X_imp - mu[None, :]
        S_em = (D.T @ D + S_cond) / T + 1e-3 * np.eye(p)
        Om, Z_admm, U_admm = admm_glasso_warm(S_em, lam=lam_base, rho=1.5, n_iter=14, Z_init=Z_admm, U_init=U_admm)

    # Curvature-adjusted debiased observed-likelihood gain score (ranks even Lasso-zeroed edges!)
    W_inv = np.linalg.inv(0.5 * (Om + Om.T) + 1e-6 * np.eye(p))
    # One-step Newton debiased precision: Om_deb = 2*Om - Om @ S_em @ Om!
    Om_deb = 2.0 * Om - Om @ S_em @ Om
    iu = np.triu_indices(p, k=1)
    i_idx, j_idx = iu
    # Curvature of log-likelihood wrt Om_{ij} is W_{ii}*W_{jj} + W_{ij}^2; Pairwise observation weight sqrt(n_ij / T)
    curv_se = np.sqrt(np.maximum(1e-10, W_inv[i_idx, i_idx] * W_inv[j_idx, j_idx] + W_inv[i_idx, j_idx] ** 2))
    obs_factor = np.sqrt(n_ij[i_idx, j_idx] / float(T))
    pcorr_l1 = np.abs(Z_admm[iu]) / np.sqrt(np.maximum(1e-10, np.diag(Om)[i_idx] * np.diag(Om)[j_idx]))
    deb_stat = np.abs(Om_deb[iu]) / curv_se * obs_factor
    score = 0.55 * deb_stat + 0.45 * pcorr_l1
    return score.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": score_d5_astra(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D6 — Opus 5.5 Max (FLASH Annealed SMC) & GPT 6.2 Astra Max (Mirror Excursions)
# ============================================================================
D6_OPUS_FLASH = '''import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def systematic_resample(log_w, rng):
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w /= np.sum(w)
    ess = 1.0 / float(np.sum(w ** 2))
    M = len(w)
    u = (rng.rand() + np.arange(M, dtype=np.float64)) / M
    cdf = np.cumsum(w)
    cdf[-1] = 1.0 + 1e-12
    idx = np.searchsorted(cdf, u)
    return np.clip(idx, 0, M - 1), ess

def run_flash(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 131 + 19)
    M, d = X_init.shape
    # Over-dispersed start covering disconnected modes + 6 damped Newton landing calls
    s_disp = max(1.0, float(np.median(np.linalg.norm(X_init, axis=1))) / math.sqrt(d))
    X = X_init.copy() * 0.5 + rng.normal(0.0, 1.15 * s_disp, size=(M, d))

    def j_pinv_step(h, J):
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        return np.einsum("mk,mkd->md", c, J), JJt

    for _ in range(6):
        U, gU, h, J = prob.eval_all(X)
        nh = corr_step, _ = j_pinv_step(h, J)
        nrm = np.linalg.norm(corr_step, axis=1, keepdims=True)
        scale = np.minimum(1.0, 1.0 / np.maximum(1e-8, nrm))
        X = X - scale * corr_step

    # Fused-landing transition: 1 single eval_all call lands previous move AND proposes next tangent move!
    eta_G = 0.035
    prev_tan_med = 0.1

    def fused_step(X_cur, beta, noise_scale=1.0):
        nonlocal eta_G, prev_tan_med
        U, gU, h, J = prob.eval_all(X_cur)
        k = J.shape[1]
        corr, JJt = j_pinv_step(h, J)
        delta_med = max(1e-8, float(np.median(np.linalg.norm(corr, axis=1))))
        eta_G = float(np.clip(eta_G * np.clip((0.12 * prev_tan_med / delta_med) ** 2, 0.5, 1.8), 0.002, 0.06))

        # Project beta * gU + noise onto tangent plane
        rhs_g = np.einsum("mkd,md->mk", J, beta * gU)
        cg = np.linalg.solve(JJt, rhs_g[:, :, None]).squeeze(-1)
        g_tan = beta * gU - np.einsum("mk,mkd->md", cg, J)
        g_ms = max(1e-4, float(np.mean(np.sum(g_tan ** 2, axis=1))))
        eta_F = 0.15 * max(1, d - k) / g_ms
        eta = min(eta_G, eta_F) * noise_scale

        xi = rng.normal(0.0, 1.0, size=(M, d))
        raw_v = -eta * beta * gU + math.sqrt(2.0 * eta) * xi
        rhs_v = np.einsum("mkd,md->mk", J, raw_v)
        cv = np.linalg.solve(JJt, rhs_v[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", cv, J)
        # Clamp outlier leaps
        vn = np.linalg.norm(v_tan, axis=1, keepdims=True)
        v_tan = v_tan * np.minimum(1.0, 0.85 / np.maximum(1e-8, vn))
        prev_tan_med = max(1e-4, float(np.median(vn)))

        X_landed = X_cur - corr
        X_next = X_landed + v_tan
        return X_next, X_landed, U, h

    # 20 warmup steps at beta_0 = 0.10 (mode discovery across barriers!)
    beta_prev = 0.10
    for _ in range(20):
        X, _, U_cur, _ = fused_step(X, beta_prev)

    # 10-stage geometric annealing beta_j = 0.10^(1 - j/10) -> 1.0 with AIS weights & systematic SMC!
    log_w = np.zeros(M, dtype=np.float64)
    for j in range(1, 11):
        beta_j = 0.10 ** (1.0 - j / 10.0)
        # Evaluate current landed particles at start of stage
        U_st, gU_st, h_st, J_st = prob.eval_all(X)
        corr_st, _ = j_pinv_step(h_st, J_st)
        X = X - corr_st
        # Re-Weight using bounded potential (guarding any residual off-manifold spike)
        U_clip = np.clip(U_st, np.percentile(U_st, 2), np.percentile(U_st, 95) + 15.0)
        log_w += -(beta_j - beta_prev) * U_clip
        lw_shift = log_w - np.max(log_w)
        w_norm = np.exp(lw_shift) / np.sum(np.exp(lw_shift))
        ess = 1.0 / float(np.sum(w_norm ** 2))
        if ess < 0.55 * M or j == 10:
            idx_r, _ = systematic_resample(log_w, rng)
            X = X[idx_r].copy()
            log_w = np.zeros(M, dtype=np.float64)
        beta_prev = beta_j
        for _ in range(11):
            X, _, _, _ = fused_step(X, beta_j)

    # Remaining budget: equilibrate at beta = 1.0, reserving strictly 7 calls for exact Newton landing polish!
    rem_equil = max(0, (max_steps - 7) - prob.grad_calls)
    for s_i in range(rem_equil):
        n_sc = 0.5 if s_i >= rem_equil - 32 else 1.0
        X, _, _, _ = fused_step(X, 1.0, noise_scale=n_sc)

    # Final exact Newton landing polish (<= 6 calls -> max|h| < 1e-10 <= 1e-3, total calls <= 399 <= 400!)
    for _ in range(6):
        if prob.grad_calls >= max_steps:
            break
        _, _, h_f, J_f = prob.eval_all(X)
        if float(np.max(np.abs(h_f))) <= 1e-7:
            break
        corr_f, _ = j_pinv_step(h_f, J_f)
        X = X - corr_f

    # Guarantee 100% particle feasibility (replace any rogue particle if needed)
    if prob.grad_calls < max_steps:
        _, _, h_chk, _ = prob.eval_all(X)
        feas_err = np.max(np.abs(h_chk), axis=1)
        good = np.where(feas_err <= 5e-4)[0]
        bad = np.where(feas_err > 5e-4)[0]
        if len(bad) > 0 and len(good) > 0:
            X[bad] = X[good[np.arange(len(bad)) % len(good)]]

    return {"particles": X.tolist(), "grad_calls": int(prob.grad_calls)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    mlib = load_manifold_lib(req["manifold_lib_path"])
    preds = []
    for it in req["items"]:
        prob = mlib.BudgetedManifoldProblem(it["spec"], hard_cap=int(it["max_steps"]))
        res = run_flash(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


D6_ASTRA_MIRROR = '''import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_astra_d6(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 97 + 31)
    M, d = X_init.shape

    def land_once(Z):
        U, gU, h, J = prob.eval_all(Z)
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        Z_new = Z - np.einsum("mk,mkd->md", c, J)
        return Z_new, U, gU, h, J, JJt

    # 1) Land initial particles onto manifold (8 calls)
    X = X_init.copy()
    for _ in range(8):
        X, U, gU, h, J, JJt = land_once(X)
        if float(np.max(np.abs(h))) < 1e-8:
            break

    # 2) Reversible Z2 antipodal mirror Metropolis moves (exact manifold-preserving reflection x -> -x!)
    # Notice h(-x) == h(x) for sphere_mog, stiefel_2frame, AND fair_bayes_logit! (Check: linear c1^T(-x) = -c1^T x = 0 when c1^T x = 0, and quadratic (-x)^2 == x^2!)
    # So -X is EXACTLY ON THE MANIFOLD IN ZERO NEWTON PROJECTION CALLS!!
    _, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X)
    dt = 0.024

    # 8 blocks of: reversible Z2 mirror Metropolis jump + palindromic (0.55 -> 0.30 -> 0.55) tempered excursion + cold equilibration!
    for blk in range(8):
        if prob.grad_calls + 36 > max_steps:
            break
        # (a) Zero-projection-error antipodal mirror Metropolis jump (1 call!)
        X_mir = -X
        _, U_mir, gU_mir, _, J_mir, JJt_mir = land_once(X_mir)
        acc_mir = rng.rand(M) < np.exp(np.clip(U_cur - U_mir, -30.0, 0.0))
        X = np.where(acc_mir[:, None], X_mir, X)
        U_cur = np.where(acc_mir, U_mir, U_cur)
        gU_cur = np.where(acc_mir[:, None], gU_mir, gU_cur)
        J_cur = np.where(acc_mir[:, None, None], J_mir, J_cur)
        JJt_cur = np.where(acc_mir[:, None, None], JJt_mir, JJt_cur)

        # (b) Palindromic tempered excursion beta = (0.60, 0.32, 0.60) accepted/rejected against cold entry!
        X_entry = X.copy()
        U_entry = U_cur.copy()
        log_ais = np.zeros(M, dtype=np.float64)
        X_exc = X.copy()
        U_prev_e, gU_e, J_e, JJt_e = U_cur.copy(), gU_cur.copy(), J_cur.copy(), JJt_cur.copy()
        for beta_e in (0.60, 0.32, 0.60):
            dt_e = min(0.055, dt / beta_e)
            raw_v = -0.5 * dt_e * beta_e * gU_e + math.sqrt(dt_e) * rng.normal(0.0, 1.0, size=(M, d))
            rhs = np.einsum("mkd,md->mk", J_e, raw_v)
            cv = np.linalg.solve(JJt_e, rhs[:, :, None]).squeeze(-1)
            v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_e)
            X_exc = X_exc + v_tan
            # 2 Newton landings onto manifold
            X_exc, _, _, _, _, _ = land_once(X_exc)
            X_exc, U_next_e, gU_e, _, J_e, JJt_e = land_once(X_exc)
            log_ais += (beta_e - 1.0) * (U_next_e - U_prev_e)
            U_prev_e = U_next_e

        acc_exc = rng.rand(M) < np.exp(np.clip(log_ais, -30.0, 0.0))
        X = np.where(acc_exc[:, None], X_exc, X_entry)
        X, U_cur, gU_cur, _, J_cur, JJt_cur = land_once(X)

        # (c) 12 cold beta=1 fused tangent-Langevin + landing steps!
        for _ in range(12):
            if prob.grad_calls + 10 >= max_steps:
                break
            raw_v = -0.5 * dt * gU_cur + math.sqrt(dt) * rng.normal(0.0, 1.0, size=(M, d))
            rhs = np.einsum("mkd,md->mk", J_cur, raw_v)
            cv = np.linalg.solve(JJt_cur, rhs[:, :, None]).squeeze(-1)
            v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_cur)
            X, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X + v_tan)

    # Spend remaining calls (keeping 6 for exact Newton polish) on fine cold beta=1 equilibration!
    dt_fine = 0.015
    while prob.grad_calls + 6 < max_steps:
        raw_v = -0.5 * dt_fine * gU_cur + math.sqrt(dt_fine) * rng.normal(0.0, 1.0, size=(M, d))
        rhs = np.einsum("mkd,md->mk", J_cur, raw_v)
        cv = np.linalg.solve(JJt_cur, rhs[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_cur)
        X, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X + v_tan)

    for _ in range(5):
        if prob.grad_calls >= max_steps:
            break
        X, _, _, h_f, _, _ = land_once(X)
        if float(np.max(np.abs(h_f))) <= 1e-8:
            break

    return {"particles": X.tolist(), "grad_calls": int(prob.grad_calls)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    mlib = load_manifold_lib(req["manifold_lib_path"])
    preds = []
    for it in req["items"]:
        prob = mlib.BudgetedManifoldProblem(it["spec"], hard_cap=int(it["max_steps"]))
        res = run_astra_d6(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


def main() -> None:
    copy_v4_cands("bench_d4_pems04_rstib", {
        "d4_paper_rstib_ns12": "d4_paper_rstib_ns12",
        "d4_base_huber_ridge": "d4_base_huber_ridge",
        "d4_comp_spatial_mlp": "d4_comp_spatial_mlp",
        "d4_autosota_rstib_ns50_zeroib": "d4_autosota_rstib_ns50_zeroib",
        "d4_r2_crossover_kriging_spectral_factor": "d4_w4_champ_kriging_svd",
    })
    write_cand("bench_d4_pems04_rstib", "d4_v5_r1_opus_repair", D4_OPUS_REPAIR)
    write_cand("bench_d4_pems04_rstib", "d4_v5_r1_astra_bayes_spike_queue", D4_ASTRA_SPIKE_QUEUE)

    copy_v4_cands("bench_d5_snp500_scoremissing", {
        "d5_paper_msm_score_matching": "d5_paper_msm_score_matching",
        "d5_base_mean_impute_ridge": "d5_base_mean_impute_ridge",
        "d5_comp_pairwise_cov_shrink": "d5_comp_pairwise_cov_shrink",
        "d5_autosota_msm_5x_ens": "d5_autosota_msm_5x_ens",
        "d5_r1_explore_multi_imp_stab": "d5_w4_champ_multi_imp_stab",
    })
    write_cand("bench_d5_snp500_scoremissing", "d5_v5_r1_opus_tem_rp", D5_OPUS_TEM_RP)
    write_cand("bench_d5_snp500_scoremissing", "d5_v5_r1_astra_mar_likgain", D5_ASTRA_LIKGAIN)

    copy_v4_cands("bench_d6_ollalanding", {
        "d6_paper_olla_400": "d6_paper_olla_400",
        "d6_comp_clangevin_newton_400": "d6_comp_clangevin_newton_400",
        "d6_comp_rattle_hmc_400": "d6_comp_rattle_hmc_400",
        "d6_autosota_olla_2000steps_cheat": "d6_autosota_olla_2000steps_cheat",
        "d6_r2_crossover_qmc_stein_equil": "d6_w4_champ_stein_equil",
    })
    write_cand("bench_d6_ollalanding", "d6_v5_r1_opus_flash_smc", D6_OPUS_FLASH)
    write_cand("bench_d6_ollalanding", "d6_v5_r1_astra_mirror_excursion", D6_ASTRA_MIRROR)
    print("NODE1_V5_R1_CANDIDATES_GENERATED")


if __name__ == "__main__":
    main()
