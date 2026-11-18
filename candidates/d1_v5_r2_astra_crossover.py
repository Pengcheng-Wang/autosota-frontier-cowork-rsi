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
    seed = sum((i + 1) * ord(c) for i, c in enumerate(gid)) % 100000 + 17
    rng = np.random.RandomState(seed)
    full = (1 << n) - 1
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = popcount_arr(masks, n)
    qs = [int(m) for m in masks[(sizes <= 2) | (sizes >= n - 2)]]
    seen = set(qs)
    reps = [int(m) for m in range(1, full) if (m not in seen) and (m < (full ^ m))]
    pick_idx = rng.choice(len(reps), size=44, replace=False)
    for idx in pick_idx:
        m = reps[int(idx)]
        qs.extend([m, full ^ m])
    return {"queries_b1": qs[:200], "want_b2": False}

def estimate_r2(n, observed):
    obs = {int(m): float(v) for m, v in observed}
    full = (1 << n) - 1
    v0 = obs.get(0, 0.0)
    obs = {m: v - v0 for m, v in obs.items()}
    W, sizes = build_W_ksii(n)
    pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Step 1: Astra Walsh-Fourier even-synergy OMP on 100 complement pairs
    pair_reps = sorted([m for m in obs.keys() if m <= (full ^ m) and (full ^ m) in obs])
    rep_arr = np.array(pair_reps, dtype=np.int64)
    comp_arr = full ^ rep_arr
    v_bar = 0.5 * np.array([obs[int(m)] + obs[int(full ^ m)] for m in rep_arr], dtype=np.float64)
    bits_rep = ((rep_arr[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    Z = 2.0 * bits_rep - 1.0
    B_quad = np.column_stack([np.ones(len(rep_arr)), *[Z[:, i] * Z[:, j] for (i, j) in pairs_ij]])
    Q_orth, _ = np.linalg.qr(B_quad, mode="reduced")

    syn_masks = [int(sum(1 << k for k in comb)) for sz in (4, 5) for comb in itertools.combinations(range(n), sz)]
    syn_arr = np.array(syn_masks, dtype=np.int64)
    H_bar = 0.5 * (((rep_arr[:, None] & syn_arr[None, :]) == syn_arr[None, :]).astype(np.float64) +
                   ((comp_arr[:, None] & syn_arr[None, :]) == syn_arr[None, :]).astype(np.float64))

    r_init = v_bar - Q_orth @ (Q_orth.T @ v_bar)
    H_proj = H_bar - Q_orth @ (Q_orth.T @ H_bar)
    col_norms = np.linalg.norm(H_proj, axis=0)
    valid = col_norms > 1e-6

    selected, coef_sel = [], np.zeros(0, dtype=np.float64)
    r_curr = r_init.copy()
    init_nrm = float(np.linalg.norm(r_init))
    if init_nrm > 1e-8:
        for _ in range(8):
            rn = float(np.linalg.norm(r_curr))
            if rn <= 0.07 * init_nrm or rn <= 1e-8:
                break
            corrs = np.zeros(len(syn_arr))
            corrs[valid] = np.abs(H_proj[:, valid].T @ r_curr) / (col_norms[valid] * rn)
            for si in selected:
                corrs[si] = -1.0
            bc = int(np.argmax(corrs))
            if corrs[bc] < 0.36:
                break
            selected.append(bc)
            Xa = H_proj[:, selected]
            coef_sel, *_ = np.linalg.lstsq(Xa, r_init, rcond=None)
            r_curr = r_init - Xa @ coef_sel

    obs_res = dict(obs)
    ksii_syn = np.zeros(n + len(pairs_ij), dtype=np.float64)
    if len(selected) > 0:
        act = syn_arr[selected]
        for mk in list(obs_res.keys()):
            obs_res[mk] -= float(((mk & act) == act).astype(np.float64) @ coef_sel)
        ksii_syn = W[:, act] @ coef_sel

    # Step 2: Fast 2-step Opus Zeta-ARD GP on Synergy-Residualized Observations (< 3ms per game -> ~0.35s total!)
    Q = np.array(sorted([m for m in obs_res.keys() if m != 0]), dtype=np.int64)
    y_q = np.array([obs_res[int(m)] for m in Q], dtype=np.float64)
    m1 = np.array([obs_res[1 << i] for i in range(n)], dtype=np.float64)
    m2 = np.array([obs_res[(1 << i) | (1 << j)] - m1[i] - m1[j] for (i, j) in pairs_ij], dtype=np.float64)
    s2 = max(1e-6, float(np.mean(m2 ** 2)), 1e-3 * float(np.mean(m1 ** 2)))

    N_m = 1 << n
    gamma = np.zeros(N_m, dtype=np.float64)
    gamma[(sizes >= 1) & (sizes <= 2)] = 1e3 * max(float(np.mean(m1 ** 2)), s2, 1e-4)
    gamma[sizes == 3] = s2 * 0.6
    gamma[sizes >= 4] = s2 * (0.25 ** (sizes[sizes >= 4] - 3))
    inter_QQ = Q[:, None] & Q[None, :]
    inter_flat = inter_QQ.ravel()

    for _ in range(3):
        zg = zeta_sub(gamma, n)
        K = zg[inter_QQ].copy()
        K[np.diag_indices_from(K)] += max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
        K_inv = np.linalg.inv(K)
        alpha = K_inv @ y_q
        sc_a = np.zeros(N_m); sc_a[Q] = alpha
        m_hat = gamma * zeta_sup(sc_a, n)
        sc_ki = np.bincount(inter_flat, weights=K_inv.ravel(), minlength=N_m)
        sig_d = np.maximum(0.0, gamma - (gamma ** 2) * zeta_sup(sc_ki, n))
        gamma = np.maximum(1e-12, m_hat ** 2 + sig_d); gamma[0] = 0.0

    zg = zeta_sub(gamma, n)
    K = zg[inter_QQ].copy()
    K[np.diag_indices_from(K)] += max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
    alpha = np.linalg.solve(K, y_q)
    sc_a = np.zeros(N_m); sc_a[Q] = alpha
    m_hat = gamma * zeta_sup(sc_a, n)
    return (W @ m_hat + ksii_syn).tolist()

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
            preds.append({"id": it["id"], "prediction": estimate_r2(int(it["n"]), it["observed"])})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
