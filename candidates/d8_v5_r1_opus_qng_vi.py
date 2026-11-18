import argparse, json, math
import numpy as np

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def qng_vi(X, y, alpha, iters=65, tol=1e-12):
    N, d = X.shape
    t = np.where(y > 0.5, 1.0, -1.0)
    I = np.eye(d, dtype=np.float64)
    # 161-point [-8.5, 8.5] trapezoid rule on standard normal -> <1e-10 quadrature error!
    e = np.linspace(-8.5, 8.5, 161)
    w = np.exp(-0.5 * e ** 2)
    w /= np.sum(w)

    def stats(mu, S):
        s = np.sqrt(np.maximum(1e-14, np.einsum("nd,de,ne->n", X, S, X)))
        Z = t[:, None] * ((X @ mu)[:, None] + s[:, None] * e[None, :])
        q = expit(-Z)
        ell = float(-np.sum(np.logaddexp(0.0, -Z) @ w))
        g = t * (q @ w)
        lam = (q * (1.0 - q)) @ w
        return ell, g, lam

    def elbo_val(mu, S, ell):
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return -1e18
        return ell - 0.5 * alpha * (float(mu @ mu) + float(np.trace(S))) + 0.5 * float(logdet)

    mu = np.zeros(d, dtype=np.float64)
    P = alpha * I + 0.25 * (X.T @ X)
    S = np.linalg.inv(P)
    ell, g, lam = stats(mu, S)
    f = elbo_val(mu, S, ell)

    for _ in range(iters):
        Pt = alpha * I + (X * np.maximum(1e-10, lam)[:, None]).T @ X
        gr = X.T @ g - alpha * mu
        rho = 1.0
        while True:
            Pn = (1.0 - rho) * P + rho * Pt
            Sn = np.linalg.inv(0.5 * (Pn + Pn.T))
            mn = mu + rho * (Sn @ gr)
            r_st = stats(mn, Sn)
            fn = elbo_val(mn, Sn, r_st[0])
            if fn >= f - 1e-12 or rho < 1e-4:
                break
            rho *= 0.5
        done = abs(fn - f) < tol
        mu, P, S, (ell, g, lam), f = mn, Pn, Sn, r_st, fn
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
    preds = []
    for it in req["items"]:
        res = qng_vi(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
