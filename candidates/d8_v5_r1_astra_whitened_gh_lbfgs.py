import argparse, json, math
import numpy as np
import torch

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def vi_astra(X, y, prior_prec):
    N, d = X.shape
    p = prior_prec
    I = np.eye(d, dtype=np.float64)
    W = np.linalg.cholesky(np.linalg.solve(p * I + 0.25 * (X.T @ X), I))
    XW = X @ W
    s_sign = 2.0 * y - 1.0
    D = -s_sign[:, None] * XW
    K_prior = p * (W.T @ W)

    # 25-node Gauss-Hermite quadrature built into numpy.polynomial.hermite!
    gh_x, gh_w = np.polynomial.hermite.hermgauss(25)
    z_gh = gh_x * math.sqrt(2.0)
    w_gh = gh_w / math.sqrt(math.pi)

    tril_i, tril_j = np.tril_indices(d, k=-1)

    def unpack(v):
        m = v[:d]
        log_d = v[d:2 * d]
        T = np.zeros((d, d), dtype=np.float64)
        np.fill_diagonal(T, np.exp(log_d))
        T[tril_i, tril_j] = v[2 * d:]
        return m, log_d, T

    def nll_and_grad(v):
        m, log_d, T = unpack(v)
        mean_u = D @ m
        DT = D @ T
        var_u = np.maximum(1e-14, np.sum(DT ** 2, axis=1))
        sd_u = np.sqrt(var_u)

        U_grid = mean_u[:, None] + sd_u[:, None] * z_gh[None, :]
        soft_grid = np.logaddexp(0.0, U_grid)
        sig_grid = expit(U_grid)

        nll_data = float(np.sum(soft_grid @ w_gh))
        g_mean = sig_grid @ w_gh
        g_sd = (sig_grid * z_gh[None, :]) @ w_gh

        grad_m = D.T @ g_mean + K_prior @ m
        scale_n = g_sd / sd_u
        grad_T = (D.T * scale_n[None, :]) @ DT + K_prior @ T

        prior_quad = 0.5 * float(m @ (K_prior @ m) + np.sum(T * (K_prior @ T)))
        loss = nll_data + prior_quad - float(np.sum(log_d))

        grad_log_d = np.diag(grad_T) * np.exp(log_d) - 1.0
        grad_off = grad_T[tril_i, tril_j]
        return loss, np.concatenate([grad_m, grad_log_d, grad_off])

    # Pure-NumPy BFGS with Armijo backtracking (35 quadratic-convergent steps, 0 scipy dependency!)
    x = np.zeros(2 * d + len(tril_i), dtype=np.float64)
    n_p = len(x)
    H_inv = np.eye(n_p, dtype=np.float64)
    f_val, g_val = nll_and_grad(x)
    for _ in range(40):
        if np.max(np.abs(g_val)) < 1e-9:
            break
        p_dir = -(H_inv @ g_val)
        if float(p_dir @ g_val) >= 0.0:
            p_dir = -g_val
            H_inv = np.eye(n_p)
        step = 1.0
        c1_sl = 1e-4 * float(p_dir @ g_val)
        for _ in range(10):
            x_try = x + step * p_dir
            f_try, g_try = nll_and_grad(x_try)
            if f_try <= f_val + step * c1_sl:
                s_v = x_try - x
                y_v = g_try - g_val
                sy = float(s_v @ y_v)
                if sy > 1e-12:
                    rho = 1.0 / sy
                    Hy = H_inv @ y_v
                    H_inv = H_inv - rho * (np.outer(s_v, Hy) + np.outer(Hy, s_v)) + (rho * (1.0 + rho * float(y_v @ Hy))) * np.outer(s_v, s_v)
                x, f_val, g_val = x_try, f_try, g_try
                break
            step *= 0.5
        else:
            break

    m_hat, _, T_hat = unpack(x)
    mu = W @ m_hat
    Sigma = (W @ T_hat) @ (W @ T_hat).T
    L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-12 * I)
    return {"mu": mu.tolist(), "L": L.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = []
    for it in req["items"]:
        res = vi_astra(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
