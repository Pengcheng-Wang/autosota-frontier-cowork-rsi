import argparse, json, math
import numpy as np
import torch

torch.set_num_threads(1)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
_CACHE = {}

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def qmc_eps(n_half, d):
    if (n_half, d) in _CACHE:
        return _CACHE[(n_half, d)]
    U = np.empty((n_half, d), dtype=np.float64)
    for j in range(d):
        b = PRIMES[j % len(PRIMES)]
        k = np.arange(1, n_half + 1, dtype=np.int64)
        f = 1.0; r = np.zeros(n_half, dtype=np.float64)
        while np.any(k > 0):
            f /= b; r += f * (k % b); k //= b
        U[:, j] = np.clip(r, 1e-6, 1.0 - 1e-6)
    eh = (math.sqrt(2.0) * torch.erfinv(2.0 * torch.tensor(U, dtype=torch.float64) - 1.0)).numpy()
    out = np.vstack([eh, -eh])
    _CACHE[(n_half, d)] = out
    return out

def fit_d8_r2_astra(X, y, alpha):
    N, d = X.shape
    t = np.where(y > 0.5, 1.0, -1.0)
    I = np.eye(d, dtype=np.float64)
    e = np.linspace(-8.5, 8.5, 161)
    w = np.exp(-0.5 * e ** 2); w /= np.sum(w)

    def stats(mu, S):
        s = np.sqrt(np.maximum(1e-14, np.einsum("nd,de,ne->n", X, S, X)))
        Z = t[:, None] * ((X @ mu)[:, None] + s[:, None] * e[None, :])
        q = expit(-Z)
        return float(-np.sum(np.logaddexp(0.0, -Z) @ w)), t * (q @ w), (q * (1.0 - q)) @ w

    mu = np.zeros(d, dtype=np.float64)
    P = alpha * I + 0.25 * (X.T @ X)
    S = np.linalg.inv(P)
    ell, g, lam = stats(mu, S)
    f = ell - 0.5 * alpha * (float(mu @ mu) + float(np.trace(S))) + 0.5 * float(np.linalg.slogdet(S)[1])

    for _ in range(45):
        Pt = alpha * I + (X * np.maximum(1e-10, lam)[:, None]).T @ X
        gr = X.T @ g - alpha * mu
        rho = 1.0
        while True:
            Pn = (1.0 - rho) * P + rho * Pt
            Sn = np.linalg.inv(0.5 * (Pn + Pn.T))
            mn = mu + rho * (Sn @ gr)
            rst = stats(mn, Sn)
            fn = rst[0] - 0.5 * alpha * (float(mn @ mn) + float(np.trace(Sn))) + 0.5 * float(np.linalg.slogdet(Sn)[1])
            if fn >= f - 1e-12 or rho < 1e-4:
                break
            rho *= 0.5
        done = abs(fn - f) < 1e-12
        mu, P, S, (ell, g, lam), f = mn, Pn, Sn, rst, fn
        if done:
            break

    L = np.linalg.cholesky(0.5 * (S + S.T) + 1e-12 * I)
    return {"mu": mu.tolist(), "L": L.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": fit_d8_r2_astra(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
