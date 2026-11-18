import argparse, json, math
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
