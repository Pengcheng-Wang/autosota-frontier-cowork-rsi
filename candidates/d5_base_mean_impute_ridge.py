import argparse, json
import numpy as np
import torch

torch.set_num_threads(1)
VARIANT = 'base_mean_impute_ridge'

def standardize_obs(X_obs, M):
    cnt = np.maximum(1.0, np.sum(M, axis=0, keepdims=True))
    mean = np.sum(X_obs * M, axis=0, keepdims=True) / cnt
    Xc = (X_obs - mean) * M
    var = np.sum((Xc ** 2) * M, axis=0, keepdims=True) / np.maximum(1.0, cnt - 1.0)
    std = np.where(var > 1e-8, np.sqrt(var), 1.0)
    return Xc / std

def pcorr_scores_from_prec(Theta):
    iu = np.triu_indices(Theta.shape[0], k=1)
    d = np.sqrt(np.maximum(1e-12, np.diag(Theta)))
    pcorr = np.abs(Theta[iu]) / (d[iu[0]] * d[iu[1]])
    return pcorr

def glasso_ista_np(S_cov, lam=0.12, n_iter=80):
    """Fast proximal-gradient / dual block or penalized precision estimator in pure NumPy."""
    p = S_cov.shape[0]
    # Nodewise L1-penalized partial correlation (Meinshausen-Buhlmann neighborhood selection!)
    # Exact coordinate descent for each variable j against X_{-j} via covariance gram S_cov!
    B = np.zeros((p, p), dtype=np.float64)
    S_reg = S_cov + 0.04 * np.eye(p)
    for j in range(p):
        idx = [k for k in range(p) if k != j]
        G = S_reg[np.ix_(idx, idx)]
        c = S_reg[idx, j]
        beta = np.zeros(p - 1, dtype=np.float64)
        diag_G = np.maximum(1e-8, np.diag(G))
        # 35 fast coordinate descent sweeps
        r = c.copy()  # c - G @ beta
        for _ in range(35):
            for a in range(p - 1):
                old = beta[a]
                rho = r[a] + diag_G[a] * old
                new_v = np.sign(rho) * max(0.0, abs(rho) - lam) / diag_G[a]
                if new_v != old:
                    r -= G[:, a] * (new_v - old)
                    beta[a] = new_v
        res_var = max(0.05, S_reg[j, j] - float(c @ beta))
        B[idx, j] = -beta / res_var
        B[j, j] = 1.0 / res_var
    # Symmetrize Meinshausen-Buhlmann precision
    Theta = 0.5 * (np.abs(B) + np.abs(B.T))
    np.fill_diagonal(Theta, np.diag(B))
    return Theta

def estimate_edges(X_obs, M):
    X = standardize_obs(X_obs, M)
    T, p = X.shape
    iu = np.triu_indices(p, k=1)

    if VARIANT == "base_mean_impute_ridge":
        S = (X.T @ X) / T
        Theta = np.linalg.inv(S + 0.25 * np.eye(p))
        return pcorr_scores_from_prec(Theta).tolist()

    if VARIANT == "comp_pairwise_cov_shrink":
        n_ij = np.maximum(1.0, M.T @ M)
        S_pair = (X.T @ X) / n_ij
        # Project to PSD via eigenvalue clipping + shrinkage
        vals, vecs = np.linalg.eigh(0.5 * (S_pair + S_pair.T))
        vals_psd = np.maximum(0.08, vals)
        S_psd = (vecs * vals_psd[None, :]) @ vecs.T
        S_shrunk = 0.80 * S_psd + 0.20 * np.diag(np.diag(S_psd))
        Theta = np.linalg.inv(S_shrunk)
        return pcorr_scores_from_prec(Theta).tolist()

    if VARIANT in ("paper_msm", "autosota_msm_5x"):
        # Marginal Score Matching (variational imputation + Hyvarinen score matching objective on precision)
        n_runs = 5 if VARIANT == "autosota_msm_5x" else 1
        scores_acc = np.zeros(len(iu[0]), dtype=np.float64)
        for r in range(n_runs):
            torch.manual_seed(10 + r * 37)
            # Initial pairwise-imputed Gaussian sampler + score matching optimization of Theta
            n_ij = np.maximum(1.0, M.T @ M)
            S_init = (X.T @ X) / n_ij
            vals, vecs = np.linalg.eigh(0.5 * (S_init + S_init.T))
            S_psd = (vecs * np.maximum(0.10, vals)[None, :]) @ vecs.T
            X_imp = X.copy()
            rng = np.random.RandomState(10 + r)
            # Impute missing entries conditionally under S_psd (1 MC draw as in MSM variational_fisher)
            for t in range(T):
                miss = np.where(M[t] < 0.5)[0]
                if 0 < len(miss) < p:
                    obs = np.where(M[t] > 0.5)[0]
                    Soo_inv = np.linalg.inv(S_psd[np.ix_(obs, obs)] + 0.05 * np.eye(len(obs)))
                    mu_c = S_psd[np.ix_(miss, obs)] @ (Soo_inv @ X[t, obs])
                    cov_c = S_psd[np.ix_(miss, miss)] - S_psd[np.ix_(miss, obs)] @ Soo_inv @ S_psd[np.ix_(obs, miss)]
                    std_d = np.sqrt(np.maximum(1e-4, np.diag(cov_c)))
                    X_imp[t, miss] = mu_c + rng.normal(0, 1, size=len(miss)) * std_d

            Xt = torch.tensor(X_imp, dtype=torch.float64)
            Theta_t = torch.eye(p, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([Theta_t], lr=0.03)
            S_emp = (Xt.T @ Xt) / T
            n_steps = 220 if VARIANT == "autosota_msm_5x" else 120
            for _ in range(n_steps):
                Sym = 0.5 * (Theta_t + Theta_t.T)
                # Hyvarinen Gaussian Score Matching loss: 0.5 * Tr(Theta @ S_emp @ Theta) - Tr(Theta) + L1 off-diag
                off = Sym - torch.diag(torch.diag(Sym))
                loss = 0.5 * torch.trace(Sym @ S_emp @ Sym) - torch.trace(Sym) + 0.08 * torch.sum(torch.abs(off))
                opt.zero_grad(); loss.backward(); opt.step()
            Th_np = 0.5 * (Theta_t.detach().numpy() + Theta_t.detach().numpy().T)
            scores_acc += pcorr_scores_from_prec(Th_np)
        return (scores_acc / n_runs).tolist()

    if VARIANT == "decision_em_gauss":
        # Proposer D: 6-iteration exact Gaussian EM accumulating conditional first & second moments + reliability factor!
        mu = np.zeros(p, dtype=np.float64)
        Sigma = np.eye(p, dtype=np.float64)
        for _ in range(6):
            M1 = np.zeros((T, p), dtype=np.float64)
            S_acc = np.zeros((p, p), dtype=np.float64)
            for t in range(T):
                obs = np.where(M[t] > 0.5)[0]
                miss = np.where(M[t] < 0.5)[0]
                xt_full = mu.copy()
                xt_full[obs] = X[t, obs]
                if len(miss) > 0 and len(obs) > 0:
                    Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                    W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                    xt_full[miss] = mu[miss] + W_mo @ (X[t, obs] - mu[obs])
                    cov_mm = Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
                    S_acc[np.ix_(miss, miss)] += cov_mm
                M1[t] = xt_full
            mu = np.mean(M1, axis=0)
            Xc_em = M1 - mu[None, :]
            Sigma_em = (Xc_em.T @ Xc_em + S_acc) / T
            Sigma = 0.90 * Sigma_em + 0.10 * np.diag(np.diag(Sigma_em)) + 1e-4 * np.eye(p)
        Omega = np.linalg.inv(Sigma)
        n_ij = M.T @ M
        pcorr = pcorr_scores_from_prec(Omega)
        rel = np.sqrt(n_ij[iu] / (n_ij[iu] + 20.0))
        return (pcorr * rel).tolist()

    if VARIANT == "coord_em_mb_glasso":
        # Proposer A: Exact Gaussian EM Sufficient-Statistic Covariance + Multi-Lambda Meinshausen-Buhlmann Stability Selection!
        # Combines exact conditional second-moment EM completion (eliminating single-imputation noise of MSM)
        # with L1 neighborhood selection across 3 sparsity levels to match sparse GraphicalLasso precision graphs!
        Sigma = np.eye(p, dtype=np.float64)
        n_ij = np.maximum(1.0, M.T @ M)
        S0 = (X.T @ X) / n_ij
        vals, vecs = np.linalg.eigh(0.5 * (S0 + S0.T))
        Sigma = 0.85 * ((vecs * np.maximum(0.08, vals)[None, :]) @ vecs.T) + 0.15 * np.eye(p)

        for _ in range(5):
            M1 = X.copy()
            S_cond = np.zeros((p, p), dtype=np.float64)
            for t in range(T):
                miss = np.where(M[t] < 0.5)[0]
                if len(miss) > 0:
                    obs = np.where(M[t] > 0.5)[0]
                    if len(obs) > 0:
                        Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                        W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                        M1[t, miss] = W_mo @ X[t, obs]
                        S_cond[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
                    else:
                        S_cond[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)]
            M1_c = M1 - np.mean(M1, axis=0, keepdims=True)
            Sigma_em = (M1_c.T @ M1_c + S_cond) / T
            Sigma = 0.92 * Sigma_em + 0.08 * np.diag(np.diag(Sigma_em)) + 1e-4 * np.eye(p)

        # Combine multi-scale Meinshausen-Buhlmann sparse precision + shrunk inverse covariance
        s1 = pcorr_scores_from_prec(glasso_ista_np(Sigma, lam=0.06))
        s2 = pcorr_scores_from_prec(glasso_ista_np(Sigma, lam=0.12))
        s3 = pcorr_scores_from_prec(np.linalg.inv(0.88 * Sigma + 0.12 * np.diag(np.diag(Sigma))))
        rel = np.sqrt(n_ij[iu] / (n_ij[iu] + 15.0))
        return ((0.40 * s1 + 0.35 * s2 + 0.25 * s3) * rel).tolist()

    raise ValueError(VARIANT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = []
    for it in req["items"]:
        X_obs = np.asarray(it["X_obs"], dtype=np.float64)
        M = np.asarray(it["mask"], dtype=np.float64)
        preds.append({"id": it["id"], "prediction": estimate_edges(X_obs, M)})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
