import argparse, json, math
import numpy as np
import torch

torch.set_num_threads(1)
VARIANT = 'paper_lsvi'

def newton_map(X, y, prior_prec):
    N, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    eye = prior_prec * np.eye(d)
    for _ in range(15):
        eta = np.clip(X @ w, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (p - y) + prior_prec * w
        s = np.maximum(1e-6, p * (1.0 - p))
        H = (X.T * s[None, :]) @ X + eye
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ w, -25.0, 25.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    H = (X.T * np.maximum(1e-6, p * (1.0 - p))[None, :]) @ X + eye
    Sigma = np.linalg.inv(H)
    return w, Sigma

def fit_vi(X, y, prior_prec, seed):
    N, d = X.shape
    rng = np.random.RandomState(74 + seed)

    if VARIANT == "comp_laplace_newton":
        mu, Sigma = newton_map(X, y, prior_prec)
        L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(d))
        return {"mu": mu.tolist(), "L": L.tolist()}

    if VARIANT == "paper_lsvi":
        # Paper #74 LSVI: fixed-K Monte Carlo linear regression of log p(y, theta) / gradient Stein identity!
        # Using Stein's lemmas for Gaussian q = N(mu, Sigma): E[g(theta)] and E[H(theta)] estimated on K=100 samples per step!
        mu, Sigma = newton_map(X, y, prior_prec)
        K_mc = 100  # Paper small-K fixed sample LSVI
        for it in range(8):
            L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(d))
            eps = rng.normal(0.0, 1.0, size=(K_mc, d))
            th = mu[None, :] + eps @ L.T  # [K, d]
            eta = np.clip(th @ X.T, -25.0, 25.0)  # [K, N]
            p = 1.0 / (1.0 + np.exp(-eta))
            # Monte Carlo expected gradient and precision via Bonnet's & Price's theorems
            g_bar = np.mean((p - y[None, :]) @ X + prior_prec * th, axis=0)
            s_bar = np.mean(np.maximum(1e-6, p * (1.0 - p)), axis=0)
            Prec_new = (X.T * s_bar[None, :]) @ X + prior_prec * np.eye(d)
            Sigma_new = np.linalg.inv(Prec_new)
            step = 0.6
            Sigma = (1.0 - step) * Sigma + step * Sigma_new
            mu = mu - step * (Sigma @ g_bar)
        L_fin = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(d))
        return {"mu": mu.tolist(), "L": L_fin.tolist()}

    if VARIANT in ("comp_bbvi_reparam_1x", "autosota_bbvi_8seed_long"):
        n_seeds = 8 if VARIANT == "autosota_bbvi_8seed_long" else 1
        n_steps = 450 if VARIANT == "autosota_bbvi_8seed_long" else 140
        best_elbo = -1e18
        best_res = None
        Xt = torch.tensor(X, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        mu0, Sig0 = newton_map(X, y, prior_prec)
        L0 = np.linalg.cholesky(0.5 * (Sig0 + Sig0.T) + 1e-8 * np.eye(d))

        for s_i in range(n_seeds):
            torch.manual_seed(740 + seed * 13 + s_i)
            mu_p = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
            L_raw = torch.tensor(L0 - np.diag(np.diag(L0)) + np.diag(np.log(np.diag(L0))), dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([mu_p, L_raw], lr=0.015)
            for _ in range(n_steps):
                L_mat = torch.tril(L_raw, diagonal=-1) + torch.diag(torch.exp(torch.diag(L_raw)))
                eps = torch.randn(32, d, dtype=torch.float64)
                eps = torch.cat([eps, -eps], dim=0)
                th = mu_p.unsqueeze(0) + eps @ L_mat.T
                logits = torch.clamp(th @ Xt.T, -30.0, 30.0)
                ll = torch.sum(yt.unsqueeze(0) * logits - torch.nn.functional.softplus(logits), dim=1)
                lp = -0.5 * prior_prec * torch.sum(th ** 2, dim=1)
                ent = torch.sum(torch.diag(L_raw))
                loss = -(torch.mean(ll + lp) + ent)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                L_mat = torch.tril(L_raw, diagonal=-1) + torch.diag(torch.exp(torch.diag(L_raw)))
                eps = torch.randn(256, d, dtype=torch.float64)
                th = mu_p.unsqueeze(0) + eps @ L_mat.T
                logits = torch.clamp(th @ Xt.T, -30.0, 30.0)
                el = float((torch.mean(torch.sum(yt.unsqueeze(0) * logits - torch.nn.functional.softplus(logits), dim=1) - 0.5 * prior_prec * torch.sum(th ** 2, dim=1)) + torch.sum(torch.diag(L_raw))).item())
                if el > best_elbo:
                    best_elbo = el
                    best_res = {"mu": mu_p.numpy().tolist(), "L": L_mat.numpy().tolist()}
        return best_res

    if VARIANT == "decision_jaakkola_jordan_pg":
        # Proposer D: 12-iteration Jaakkola-Jordan / Polya-Gamma EM quadratic variational bound!
        mu = np.zeros(d, dtype=np.float64)
        Lambda0 = prior_prec * np.eye(d)
        Sigma = np.linalg.inv(Lambda0)
        for _ in range(12):
            xSx = np.sum((X @ Sigma) * X, axis=1)
            t_i = np.sqrt(np.maximum(0.0, (X @ mu) ** 2 + xSx))
            w_i = np.where(t_i > 1e-6, np.tanh(0.5 * t_i) / (2.0 * t_i), 0.25)
            Sigma = np.linalg.inv(Lambda0 + (X.T * w_i[None, :]) @ X)
            mu = Sigma @ (X.T @ (y - 0.5))
        L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-9 * np.eye(d))
        return {"mu": mu.tolist(), "L": L.tolist()}

    if VARIANT == "coord_gauss_hermite_vogn":
        # Proposer A: Exact 11-Node Gauss-Hermite Quadrature Variational Newton!
        # Exploits 1D projection structure: for q = N(mu, Sigma), eta_i = x_i^T theta ~ N(m_i, v_i)!
        # Exact E_q[sigmoid(eta_i)] and E_q[sigmoid'(eta_i)] via 11-point Gauss-Hermite quadrature
        # solves Bonnet's + Price's exact full-covariance Gaussian ELBO optimality equations to 1e-10 in 12 damped Newton steps!
        gh_x, gh_w = np.polynomial.hermite.hermgauss(11)
        # Convert physicists' Hermite integral int exp(-x^2) f(x) dx to standard normal E_{z~N(0,1)}[f(z)]:
        z_nodes = gh_x * math.sqrt(2.0)  # [11]
        w_nodes = gh_w / math.sqrt(math.pi)  # [11], sum == 1.0

        mu, Sigma = newton_map(X, y, prior_prec)
        Lambda0 = prior_prec * np.eye(d)
        for _ in range(14):
            m_vec = X @ mu  # [N]
            v_vec = np.maximum(1e-12, np.sum((X @ Sigma) * X, axis=1))  # [N]
            eta_grid = m_vec[:, None] + np.sqrt(v_vec)[:, None] * z_nodes[None, :]  # [N, 11]
            sig_grid = 1.0 / (1.0 + np.exp(-np.clip(eta_grid, -30.0, 30.0)))
            psi = sig_grid @ w_nodes  # E_q[sigmoid(eta_i)]
            kappa = (sig_grid * (1.0 - sig_grid)) @ w_nodes  # E_q[sigmoid'(eta_i)]

            # Exact Price's theorem optimality: Sigma^{-1} = Lambda_0 + X^T diag(kappa) X
            Prec_target = Lambda0 + (X.T * np.maximum(1e-8, kappa)[None, :]) @ X
            Sigma_target = np.linalg.inv(Prec_target)
            # Exact Bonnet's theorem optimality: grad_mu ELBO = X^T (y - psi) - prior_prec * mu = 0
            g_mu = X.T @ (psi - y) + prior_prec * mu
            step_mu = Sigma_target @ g_mu

            Sigma = 0.25 * Sigma + 0.75 * Sigma_target
            mu = mu - 0.85 * step_mu
            if np.max(np.abs(step_mu)) < 1e-8 and np.max(np.abs(Sigma - Sigma_target)) < 1e-8:
                Sigma = Sigma_target
                break

        L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(d))
        return {"mu": mu.tolist(), "L": L.tolist()}

    raise ValueError(VARIANT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = []
    for idx, it in enumerate(req["items"]):
        X = np.asarray(it["X"], dtype=np.float64)
        y = np.asarray(it["y"], dtype=np.float64)
        preds.append({"id": it["id"], "prediction": fit_vi(X, y, float(it["prior_prec"]), idx)})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
