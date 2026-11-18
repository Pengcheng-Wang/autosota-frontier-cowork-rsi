import argparse, json
import numpy as np

def bit_maxcut(A, seed, K=64, T=150, top=16, tabuT=42):
    rng = np.random.default_rng(seed)
    A = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    tot = float(A.sum()) * 0.25
    c0 = 0.5 * np.sqrt(max(1.0, n - 1.0) / max(1e-12, float((A ** 2).sum())))

    X = 0.02 * (rng.random((K, n)) - 0.5)
    P = 0.02 * (rng.random((K, n)) - 0.5)
    best = np.full(K, -np.inf, dtype=np.float64)
    bS = np.ones((K, n), dtype=np.float64)

    for t in range(T):
        S = np.where(X >= 0.0, 1.0, -1.0)
        AS = S @ A
        if 2 * t >= T:
            c = tot - 0.25 * np.sum(S * AS, axis=1)
            imp = c > best
            best[imp] = c[imp]
            bS[imp] = S[imp]
        P += -(1.0 - t / T) * X - c0 * AS
        X += P
        np.clip(X, -1.0, 1.0, out=X)
        P *= (np.abs(X) < 1.0)

    S = bS[np.argsort(-best)[:top]].copy()
    H = S @ A
    k = len(S)
    ar = np.arange(k)

    # Exact 1-flip steepest ascent
    for _ in range(4 * n):
        G = S * H
        i = np.argmax(G, axis=1)
        m = G[ar, i] > 1e-9
        if not np.any(m):
            break
        si = S[ar, i] * m
        S[ar, i] -= 2.0 * si
        H -= 2.0 * si[:, None] * A[i]

    cut = tot - 0.25 * np.sum(S * H, axis=1)
    best_t = cut.copy()
    bS_t = S.copy()
    tb = np.zeros((k, n), dtype=np.int64)
    ten = rng.integers(max(2, n // 20 + 1), max(4, n // 7 + 2), size=k)

    for t in range(1, tabuT + 1):
        G = S * H
        allowed = (tb <= t) | (G > (best_t - cut)[:, None] + 1e-9)
        i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
        si = S[ar, i].copy()
        cut += si * H[ar, i]
        S[ar, i] = -si
        H -= 2.0 * si[:, None] * A[i]
        tb[ar, i] = t + ten
        j = cut > best_t + 1e-9
        best_t[j] = cut[j]
        bS_t[j] = S[j]

    return bS_t[int(np.argmax(best_t))].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": bit_maxcut(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
