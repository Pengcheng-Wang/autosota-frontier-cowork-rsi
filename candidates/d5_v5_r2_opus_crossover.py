import argparse, json, math
import numpy as np

def admm_glasso_warm(R, lam, rho=1.5, n_iter=20, Z_init=None, U_init=None):
    p = R.shape[0]
    Z = np.eye(p, dtype=np.float64) if Z_init is None else Z_init.copy()
    U = np.zeros((p, p), dtype=np.float64) if U_init is None else U_init.copy()
    thresh = lam / rho
    Om = Z.copy()
    for _ in range(n_iter):
        mat = rho * (Z - U) - R
        vals, Q = np.linalg.eigh(0.5 * (mat + mat.T))
        eig_om = (vals + np.sqrt(vals ** 2 + 4.0 * rho)) / (2.0 * rho)
        Om = (Q * eig_om[None, :]) @ Q.T
        V = Om + U
        Zn = np.sign(V) * np.maximum(0.0, np.abs(V) - thresh)
        np.fill_diagonal(Zn, np.diag(V))
        U = U + Om - Zn
        Z = Zn
    return Om, Z, U

def mar_em_corr(X, M, nu=None, n_iter=6):
    T, p = X.shape
    mu = np.zeros(p, dtype=np.float64)
    Sigma = np.eye(p, dtype=np.float64)
    for r in range(n_iter):
        X_imp = X.copy()
        S_cond = np.zeros((p, p), dtype=np.float64)
        w_t = np.ones(T, dtype=np.float64)
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            obs = np.where(M[t] > 0.5)[0]
            if 0 < len(miss) < p:
                Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                do = X[t, obs] - mu[obs]
                X_imp[t, miss] = mu[miss] + W_mo @ do
                S_cond[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
                if nu is not None:
                    w_t[t] = (nu + len(obs)) / (nu + max(0.0, float(do @ (Soo_inv @ do))))
            elif len(miss) == p:
                X_imp[t] = mu
                S_cond += Sigma
            elif nu is not None and len(obs) == p:
                Soo_inv = np.linalg.inv(Sigma + 1e-4 * np.eye(p))
                do = X[t] - mu
                w_t[t] = (nu + p) / (nu + max(0.0, float(do @ (Soo_inv @ do))))
        mu = (w_t @ X_imp) / max(1e-8, float(np.sum(w_t)))
        D = X_imp - mu[None, :]
        Sigma = 0.92 * (((D.T * w_t[None, :]) @ D + S_cond) / T) + 0.08 * np.eye(p)

    vals, vecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    S_psd = (vecs * np.maximum(0.03, vals)[None, :]) @ vecs.T
    d = np.sqrt(np.maximum(1e-10, np.diag(S_psd)))
    return S_psd / np.outer(d, d)

def rank_norm(v):
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(v))
    return ranks

def score_d5_r2_opus(X_obs, M, seed):
    T, p = X_obs.shape
    cnt = np.maximum(1.0, np.sum(M, axis=0))
    mu0 = np.sum(X_obs * M, axis=0) / cnt
    Xc = (X_obs - mu0[None, :]) * M
    sd0 = np.sqrt(np.maximum(1e-6, np.sum((Xc ** 2) * M, axis=0) / np.maximum(1.0, cnt - 1.0)))
    X = Xc / sd0[None, :]

    n_ij = np.maximum(1.0, M.T @ M)
    iu = np.triu_indices(p, k=1)
    i_idx, j_idx = iu
    sqrt_nij = np.sqrt(n_ij[i_idx, j_idx])
    lam0 = math.sqrt(math.log(max(3, p)) / max(8.0, float(np.median(n_ij))))

    R_gauss = mar_em_corr(X, M, nu=None, n_iter=6)
    R_t5 = mar_em_corr(X, M, nu=5.0, n_iter=6)

    rank_sum = np.zeros(len(i_idx), dtype=np.float64)
    n_views = 0

    for R_mat in (R_gauss, R_t5):
        Z_w, U_w = None, None
        for mult in (0.35, 0.55, 0.85):
            Om, Z_w, U_w = admm_glasso_warm(R_mat, lam=lam0 * mult, rho=1.5, n_iter=22, Z_init=Z_w, U_init=U_w)
            Om_deb = 2.0 * Om - Om @ R_mat @ Om
            denom = np.sqrt(np.maximum(1e-10, Om[i_idx, i_idx] * Om[j_idx, j_idx] + Om[i_idx, j_idx] ** 2))
            z_stat = np.abs(Om_deb[iu]) * sqrt_nij / denom
            pcorr_sp = np.abs(Z_w[iu]) / np.sqrt(np.maximum(1e-10, Om[i_idx, i_idx] * Om[j_idx, j_idx]))
            rank_sum += rank_norm(0.65 * z_stat + 0.35 * pcorr_sp * sqrt_nij)
            n_views += 1

    # 4 fast 75%-subsample stability refits
    rng = np.random.RandomState(550 + seed)
    sub_n = max(12, int(0.75 * T))
    for _ in range(4):
        idx = rng.choice(T, size=sub_n, replace=False)
        Rs = mar_em_corr(X[idx], M[idx], nu=5.0, n_iter=4)
        nij_s = np.maximum(1.0, M[idx].T @ M[idx])
        Om_s, Z_s, _ = admm_glasso_warm(Rs, lam=lam0 * 0.50, rho=1.5, n_iter=18)
        Om_deb_s = 2.0 * Om_s - Om_s @ Rs @ Om_s
        den_s = np.sqrt(np.maximum(1e-10, Om_s[i_idx, i_idx] * Om_s[j_idx, j_idx] + Om_s[i_idx, j_idx] ** 2))
        zs = np.abs(Om_deb_s[iu]) * np.sqrt(nij_s[i_idx, j_idx]) / den_s
        rank_sum += 0.5 * rank_norm(zs)
        n_views += 0.5

    return (rank_sum / n_views).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": score_d5_r2_opus(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64), i)} for i, it in enumerate(req["items"])]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
