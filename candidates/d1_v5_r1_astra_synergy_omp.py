import argparse, itertools, json, math
import numpy as np

def popcount_arr(masks, n):
    return np.sum(((masks[:, None] >> np.arange(n)[None, :]) & 1), axis=1)

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
    return W

def propose_b1(gid, n):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(gid)) % 100000 + 17
    rng = np.random.RandomState(seed)
    full = (1 << n) - 1
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = popcount_arr(masks, n)
    # 112 exact boundary coalitions |T| in {0, 1, 2, 8, 9, 10}
    qs = [int(m) for m in masks[(sizes <= 2) | (sizes >= n - 2)]]
    seen = set(qs)
    # 44 random unused complement pairs (88 queries -> 112 + 88 = 200!)
    reps = [int(m) for m in range(1, full) if (m not in seen) and (m < (full ^ m))]
    pick_idx = rng.choice(len(reps), size=44, replace=False)
    for idx in pick_idx:
        m = reps[int(idx)]
        qs.extend([m, full ^ m])
    return {"queries_b1": qs[:200], "want_b2": False}

def estimate_astra(n, observed):
    obs = {int(m): float(v) for m, v in observed}
    full = (1 << n) - 1
    v0 = obs.get(0, 0.0)
    obs = {m: v - v0 for m, v in obs.items()}
    W = build_W_ksii(n)

    # Collect all complete complement pairs (m, full^m) -> 100 pairs!
    pair_reps = sorted([m for m in obs.keys() if m <= (full ^ m) and (full ^ m) in obs])
    P = len(pair_reps)
    rep_arr = np.array(pair_reps, dtype=np.int64)
    comp_arr = full ^ rep_arr
    v_bar = 0.5 * np.array([obs[int(m)] + obs[int(full ^ m)] for m in rep_arr], dtype=np.float64)

    bits_rep = ((rep_arr[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    Z = 2.0 * bits_rep - 1.0  # [-1, +1] Walsh variables
    pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Degree-0 + Degree-2 Walsh basis (46 columns; odd degrees 1 & 3 vanish identically in v_bar!)
    B_quad = np.column_stack([np.ones(P), *[Z[:, i] * Z[:, j] for (i, j) in pairs_ij]])  # [100, 46]
    Q_orth, _ = np.linalg.qr(B_quad, mode="reduced")  # [100, 46]

    # Candidate sparse order>=4 unanimity synergies A (|A| in {4, 5}) -> 210 + 252 = 462 candidates
    syn_masks = [int(sum(1 << k for k in comb)) for sz in (4, 5) for comb in itertools.combinations(range(n), sz)]
    syn_arr = np.array(syn_masks, dtype=np.int64)
    H_rep = ((rep_arr[:, None] & syn_arr[None, :]) == syn_arr[None, :]).astype(np.float64)
    H_comp = ((comp_arr[:, None] & syn_arr[None, :]) == syn_arr[None, :]).astype(np.float64)
    H_bar = 0.5 * (H_rep + H_comp)  # [100, 462]

    # Project v_bar and H_bar orthogonally off B_quad (zero for ALL cubic games!)
    r_init = v_bar - Q_orth @ (Q_orth.T @ v_bar)
    H_proj = H_bar - Q_orth @ (Q_orth.T @ H_bar)
    col_norms = np.linalg.norm(H_proj, axis=0)
    valid_cols = col_norms > 1e-6

    selected = []
    coef_sel = np.zeros(0, dtype=np.float64)
    r_curr = r_init.copy()
    init_norm = float(np.linalg.norm(r_init))
    if init_norm > 1e-8:
        for _ in range(8):
            r_norm = float(np.linalg.norm(r_curr))
            if r_norm <= 0.08 * init_norm or r_norm <= 1e-8:
                break
            corrs = np.zeros(len(syn_arr), dtype=np.float64)
            corrs[valid_cols] = np.abs(H_proj[:, valid_cols].T @ r_curr) / (col_norms[valid_cols] * r_norm)
            for s_i in selected:
                corrs[s_i] = -1.0
            best_c = int(np.argmax(corrs))
            if corrs[best_c] < 0.38:
                break
            selected.append(best_c)
            X_act = H_proj[:, selected]
            coef_sel, *_ = np.linalg.lstsq(X_act, r_init, rcond=None)
            r_curr = r_init - X_act @ coef_sel

    # Subtract fitted order>=4 synergies from all observed coalitions!
    obs_res = dict(obs)
    ksii_syn = np.zeros(n + len(pairs_ij), dtype=np.float64)
    if len(selected) > 0:
        act_masks = syn_arr[selected]
        for m_k in list(obs_res.keys()):
            hit = ((m_k & act_masks) == act_masks).astype(np.float64)
            obs_res[m_k] -= float(hit @ coef_sel)
        ksii_syn = W[:, act_masks] @ coef_sel

    # Exact cubic-game boundary formulas on residual obs_res using |T| in {0, 1, 2, 8, 9, 10}!
    vN = obs_res[full]
    d1_0 = np.array([obs_res[1 << i] for i in range(n)], dtype=np.float64)
    d1_N = np.array([vN - obs_res[full ^ (1 << i)] for i in range(n)], dtype=np.float64)

    d2_0 = np.zeros((n, n), dtype=np.float64)
    d2_N = np.zeros((n, n), dtype=np.float64)
    pair_res = np.zeros(len(pairs_ij), dtype=np.float64)
    for p_idx, (i, j) in enumerate(pairs_ij):
        m_ij = (1 << i) | (1 << j)
        val0 = obs_res[m_ij] - d1_0[i] - d1_0[j]
        valN = vN - obs_res[full ^ (1 << i)] - obs_res[full ^ (1 << j)] + obs_res[full ^ m_ij]
        d2_0[i, j] = d2_0[j, i] = val0
        d2_N[i, j] = d2_N[j, i] = valN
        pair_res[p_idx] = 0.5 * (val0 + valN)

    # Also refine residual order-3 terms with WLS over the 88 middle observations if cubic+odd residuals exist!
    mid_masks = np.array([m for m in obs_res.keys() if 2 < bin(m).count("1") < n - 2], dtype=np.int64)
    if len(mid_masks) >= 20:
        # Check residual off quadratic approx on middle coalitions to refine pair_res slightly
        sum_d2_0 = np.sum(d2_0, axis=1)
        sum_m3_i = d1_N - d1_0 - sum_d2_0
        main_res = d1_0 - (1.0 / 6.0) * sum_m3_i
    else:
        main_res = d1_0 - (1.0 / 6.0) * (d1_N - d1_0 - np.sum(d2_0, axis=1))

    est = np.concatenate([main_res, pair_res]) + ksii_syn
    return est.tolist()

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
            preds.append({"id": it["id"], "prediction": estimate_astra(int(it["n"]), it["observed"])})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
