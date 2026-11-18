import argparse, itertools, json
import numpy as np

def popcount_arr(masks, n):
    return np.sum(((masks[:, None] >> np.arange(n)[None, :]) & 1), axis=1)

def zeta_sub(arr, n):
    out = arr.astype(np.float64, copy=True)
    for b in range(n):
        bit = 1 << b
        for base in range(0, 1 << n, bit << 1):
            out[base + bit: base + (bit << 1)] += out[base: base + bit]
    return out

def zeta_sup(arr, n):
    out = arr.astype(np.float64, copy=True)
    for b in range(n):
        bit = 1 << b
        for base in range(0, 1 << n, bit << 1):
            out[base: base + bit] += out[base + bit: base + (bit << 1)]
    return out

def build_W_ksii(n):
    N_m = 1 << n
    masks = np.arange(N_m, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1)
    sizes = np.sum(bits, axis=1)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    W = np.zeros((n + len(pairs), N_m), dtype=np.float64)
    for a in range(1, N_m):
        k = int(sizes[a])
        wm = (1.0 / k) - (0.5 if k >= 2 else 0.0)
        for i in range(n):
            if (a >> i) & 1:
                W[i, a] = wm
        if k >= 2:
            wp = 1.0 / (k - 1.0)
            for p_idx, (i, j) in enumerate(pairs):
                if ((a >> i) & 1) and ((a >> j) & 1):
                    W[n + p_idx, a] = wp
    return W, sizes

def propose_b1(gid, n):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(gid)) % 100000 + 31
    rng = np.random.RandomState(seed)
    full = (1 << n) - 1
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = popcount_arr(masks, n)
    qs = [int(m) for m in masks[(sizes <= 2) | (sizes >= n - 2)]]
    seen = set(qs)
    # Stratified middle complement pairs across sizes 3, 4, 5
    reps = [int(m) for m in range(1, full) if (m not in seen) and (m < (full ^ m))]
    pick_idx = rng.choice(len(reps), size=44, replace=False)
    for idx in pick_idx:
        m = reps[int(idx)]
        qs.extend([m, full ^ m])
    return {"queries_b1": qs[:200], "want_b2": False}

def estimate_r2_opus(n, observed):
    obs = {int(m): float(v) for m, v in observed}
    v0 = obs.get(0, 0.0)
    Q = np.array(sorted([m for m in obs.keys() if m != 0]), dtype=np.int64)
    y = np.array([obs[int(m)] - v0 for m in Q], dtype=np.float64)
    W, sizes = build_W_ksii(n)
    N_m = 1 << n

    m1 = np.array([obs[1 << i] - v0 for i in range(n)], dtype=np.float64)
    m2 = np.array([obs[(1 << i) | (1 << j)] - v0 - m1[i] - m1[j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)
    s1 = max(1e-6, float(np.mean(m1 ** 2)))
    s2 = max(1e-3 * s1, float(np.mean(m2 ** 2)))

    gamma = np.zeros(N_m, dtype=np.float64)
    gamma[(sizes >= 1) & (sizes <= 2)] = 1e3 * max(s1, s2)
    gamma[sizes >= 3] = s2 * (0.45 ** (sizes[sizes >= 3] - 2))

    inter_QQ = Q[:, None] & Q[None, :]
    inter_flat = inter_QQ.ravel()
    for _ in range(6):
        zg = zeta_sub(gamma, n)
        K = zg[inter_QQ].copy()
        K[np.diag_indices_from(K)] += max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
        K_inv = np.linalg.inv(K)
        alpha = K_inv @ y
        sc_a = np.zeros(N_m); sc_a[Q] = alpha
        m_hat = gamma * zeta_sup(sc_a, n)
        sc_ki = np.bincount(inter_flat, weights=K_inv.ravel(), minlength=N_m)
        sig_d = np.maximum(0.0, gamma - (gamma ** 2) * zeta_sup(sc_ki, n))
        gamma = np.maximum(1e-12, m_hat ** 2 + sig_d); gamma[0] = 0.0

    zg = zeta_sub(gamma, n)
    K = zg[inter_QQ].copy()
    K[np.diag_indices_from(K)] += max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
    alpha = np.linalg.solve(K, y)
    sc_a = np.zeros(N_m); sc_a[Q] = alpha
    m_hat = gamma * zeta_sup(sc_a, n)
    return (W @ m_hat).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    step = req["step"]
    preds = []
    for it in req["items"]:
        if step == "propose_b1":
            preds.append({"id": it["id"], "prediction": propose_b1(it["id"], int(it["n"]))})
        elif step == "estimate":
            preds.append({"id": it["id"], "prediction": estimate_r2_opus(int(it["n"]), it["observed"])})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
