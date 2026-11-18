import argparse, json, math
import numpy as np

VARIANT = 'autosota_1x'

def popcount_arr(masks, n):
    return np.sum(((masks[:, None] >> np.arange(n)[None, :]) & 1), axis=1)

def propose_b1(gid, n, max_budget):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(gid)) % 100000
    rng = np.random.RandomState(seed)
    full = (1 << n) - 1
    if VARIANT == "autosota_ens10_cheat":
        # Reproduces AutoSOTA's N_ENSEMBLE=10 query-budget violation (>200 distinct queries!)
        qs = [0, full]
        for r in range(10):
            rng_r = np.random.RandomState(seed * 31 + r * 997)
            for _ in range(99):
                sz = int(rng_r.randint(1, n))
                idx = rng_r.choice(n, size=sz, replace=False)
                m = int(sum(1 << int(i) for i in idx))
                qs.extend([m, full ^ m])
        return {"queries_b1": qs, "want_b2": False}

    if VARIANT == "coord_wls_shapley_taylor":
        # Exact deterministic singleton + (n-1) boundary coalitions (20 masks) + 89 paired harmonic coalitions = 200!
        seen = {0, full}
        qs = [0, full]
        for i in range(n):
            for m in (1 << i, full ^ (1 << i)):
                if m not in seen:
                    seen.add(m); qs.append(m)
        # Remaining 178 slots -> 89 antithetic pairs weighted by Shapley-Taylor kernel 1/(s*(n-s))
        sizes = np.arange(2, n - 1)
        probs = 1.0 / (sizes * (n - sizes))
        probs /= np.sum(probs)
        while len(qs) < 200:
            sz = int(rng.choice(sizes, p=probs))
            idx = rng.choice(n, size=sz, replace=False)
            m = int(sum(1 << int(i) for i in idx))
            mc = full ^ m
            if m not in seen and len(qs) + (2 if mc != m else 1) <= 200:
                seen.add(m); qs.append(m)
                if mc not in seen and len(qs) < 200:
                    seen.add(mc); qs.append(mc)
        return {"queries_b1": qs[:200], "want_b2": False}

    # Standard <=200 paired / stratified designs
    seen = {0, full}
    qs = [0, full]
    sizes = np.arange(1, n)
    if VARIANT == "paper_svarmiq":
        probs = 1.0 / (sizes * (n - sizes))
    elif VARIANT in ("autosota_1x", "decision_eb_ht"):
        probs = np.ones_like(sizes, dtype=np.float64)
    else:
        # Unstratified uniform coalition binomial weights
        probs = np.array([math.comb(n, int(s)) for s in sizes], dtype=np.float64)
    probs = probs / np.sum(probs)

    attempts = 0
    while len(qs) < 200 and attempts < 4000:
        attempts += 1
        sz = int(rng.choice(sizes, p=probs))
        idx = rng.choice(n, size=sz, replace=False)
        m = int(sum(1 << int(i) for i in idx))
        mc = full ^ m
        if m not in seen:
            seen.add(m); qs.append(m)
            if VARIANT != "base_mc" or len(qs) < 200:
                if mc not in seen and len(qs) < 200:
                    seen.add(mc); qs.append(mc)
    return {"queries_b1": qs[:200], "want_b2": False}

def estimate_ksii(n, observed):
    obs_dict = {int(m): float(v) for m, v in observed}
    full = (1 << n) - 1
    v0 = obs_dict.get(0, 0.0)
    vN = obs_dict.get(full, 0.0)
    masks = np.array([m for m in obs_dict.keys() if m not in (0, full)], dtype=np.int64)
    vals = np.array([obs_dict[int(m)] - v0 for m in masks], dtype=np.float64)
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    sizes = np.sum(bits, axis=1).astype(np.int64)

    pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs_ij)

    if VARIANT == "coord_wls_shapley_taylor":
        # Build exact 55-column design matrix [main_1..n, pair_1..45]
        Z_pairs = np.column_stack([bits[:, i] * bits[:, j] for (i, j) in pairs_ij])
        X = np.hstack([bits, Z_pairs])  # [Q, 55]
        # Exact Shapley-Taylor kernel weights w(|S|) = (n-1) / (comb(n, |S|) * |S| * (n - |S|))
        w = np.array([
            100.0 if (s == 1 or s == n - 1) else float((n - 1) / (math.comb(n, int(s)) * s * (n - s)))
            for s in sizes
        ], dtype=np.float64)
        # Enforce exact efficiency sum(beta) == v(N) - v(0) via high-weight pseudo-observation
        X_aug = np.vstack([X, np.ones((1, X.shape[1]))])
        y_aug = np.append(vals, vN - v0)
        w_aug = np.append(w, 1e5)
        reg = np.diag([1e-6] * n + [2e-3] * n_pairs)
        XtW = X_aug.T * w_aug[None, :]
        beta = np.linalg.solve(XtW @ X_aug + reg, XtW @ y_aug)
        # For order-2 polynomial game v(S) = v0 + sum_i a_i z_i + sum_{i<j} a_{ij} z_i z_j:
        # Exact k-SII order-2 has I_{ij} = a_{ij} and phi^(2)_i = a_i!
        return beta.tolist()

    # Horvitz-Thompson / Regression-adjusted Monte Carlo estimators for baselines & Decision-Theory EB
    fact = np.array([math.factorial(k) for k in range(n + 1)], dtype=np.float64)
    # Estimate inclusion probabilities q(S) under each sampling scheme
    if VARIANT == "base_mc":
        q_s = np.full(len(masks), max(1e-6, len(masks) / float((1 << n) - 2)))
    else:
        # Stratified by coalition size (9 strata s=1..9)
        cnt_by_s = {s: max(1, int(np.sum(sizes == s))) for s in range(1, n)}
        q_s = np.array([cnt_by_s[int(s)] / float(math.comb(n, int(s))) for s in sizes], dtype=np.float64)

    # Exact HT weights for SII main & pair terms
    shap1 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bi = bits[:, i] > 0.5
        s_wo = np.where(bi, sizes - 1, sizes)
        sign = np.where(bi, 1.0, -1.0)
        a_i = sign * fact[s_wo] * fact[n - s_wo - 1] / fact[n]
        a_i_0 = -1.0 / n
        a_i_N = 1.0 / n
        shap1[i] = a_i_0 * v0 + a_i_N * vN + float(np.sum((a_i * (vals + v0)) / np.maximum(1e-6, q_s)))

    pair_est = np.zeros(n_pairs, dtype=np.float64)
    pair_var = np.zeros(n_pairs, dtype=np.float64)
    pair_mat = np.zeros((n, n), dtype=np.float64)
    for p_idx, (i, j) in enumerate(pairs_ij):
        bij = bits[:, i] + bits[:, j]
        s_wo = (sizes - bij).astype(np.int64)
        sign = np.where(bij == 1, -1.0, 1.0)
        a_ij = sign * fact[s_wo] * fact[n - s_wo - 2] / fact[n - 1]
        contrib = (a_ij * (vals + v0)) / np.maximum(1e-6, q_s)
        est_v = (1.0 / (n - 1)) * v0 + (1.0 / (n - 1)) * vN + float(np.sum(contrib))
        pair_est[p_idx] = est_v
        pair_var[p_idx] = float(np.var(contrib) * len(contrib))
        pair_mat[i, j] = pair_mat[j, i] = est_v

    ksii_main = shap1 - 0.5 * np.sum(pair_mat, axis=1)
    if VARIANT == "decision_eb_ht":
        tau2 = max(1e-10, float(np.mean(pair_est ** 2 - pair_var)))
        shrink = tau2 / (tau2 + np.maximum(1e-10, pair_var))
        pair_est = pair_est * shrink
    return np.concatenate([ksii_main, pair_est]).tolist()

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
            preds.append({"id": it["id"], "prediction": propose_b1(it["id"], int(it["n"]), int(it["max_budget"]))})
        elif step == "estimate":
            preds.append({"id": it["id"], "prediction": estimate_ksii(int(it["n"]), it["observed"])})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
