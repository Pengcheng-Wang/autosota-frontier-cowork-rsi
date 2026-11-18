import argparse, json
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
