import argparse, json
import numpy as np

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def qng_vi_241(X, y, alpha):
    N, d = X.shape
    t = np.where(y > 0.5, 1.0, -1.0)
    I = np.eye(d, dtype=np.float64)
    e = np.linspace(-9.0, 9.0, 241)
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

    for _ in range(70):
        Pt = alpha * I + (X * np.maximum(1e-10, lam)[:, None]).T @ X
        gr = X.T @ g - alpha * mu
        rho = 1.0
        while True:
            Pn = (1.0 - rho) * P + rho * Pt
            Sn = np.linalg.inv(0.5 * (Pn + Pn.T))
            mn = mu + rho * (Sn @ gr)
            rst = stats(mn, Sn)
            fn = rst[0] - 0.5 * alpha * (float(mn @ mn) + float(np.trace(Sn))) + 0.5 * float(np.linalg.slogdet(Sn)[1])
            if fn >= f - 1e-13 or rho < 1e-4:
                break
            rho *= 0.5
        done = abs(fn - f) < 1e-13
        mu, P, S, (ell, g, lam), f = mn, Pn, Sn, rst, fn
        if done:
            break

    L = np.linalg.cholesky(0.5 * (S + S.T) + 1e-13 * I)
    return {"mu": mu.tolist(), "L": L.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": qng_vi_241(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
