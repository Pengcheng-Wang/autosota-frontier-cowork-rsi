import argparse, json
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
