import argparse, json, math
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
        w_main = (1.0 / k) - (0.5 if k >= 2 else 0.0)
        for i in range(n):
            if (a >> i) & 1:
                W[i, a] = w_main
        if k >= 2:
            w_pair = 1.0 / (k - 1.0)
            for p_idx, (i, j) in enumerate(pairs):
                if ((a >> i) & 1) and ((a >> j) & 1):
                    W[n + p_idx, a] = w_pair
    return W, sizes

def fit_bemc_gp(n, observed, n_em=5):
    obs_dict = {int(m): float(v) for m, v in observed}
    v0 = obs_dict.get(0, 0.0)
    Q = np.array(sorted([m for m in obs_dict.keys() if m != 0]), dtype=np.int64)
    y = np.array([obs_dict[int(m)] - v0 for m in Q], dtype=np.float64)
    W, sizes = build_W_ksii(n)
    N_m = 1 << n

    # Exact border Mobius terms m_i and m_{ij} from |T| <= 2
    m1 = np.array([obs_dict.get(1 << i, v0) - v0 for i in range(n)], dtype=np.float64)
    m2 = []
    for i in range(n):
        for j in range(i + 1, n):
            mij = obs_dict.get((1 << i) | (1 << j), v0) - v0 - m1[i] - m1[j]
            m2.append(mij)
    m2 = np.array(m2, dtype=np.float64)
    s1_sq = max(1e-6, float(np.mean(m1 ** 2)))
    s2_sq = max(1e-3 * s1_sq, float(np.mean(m2 ** 2)))

    gamma = np.zeros(N_m, dtype=np.float64)
    for a in range(1, N_m):
        k = int(sizes[a])
        if k <= 2:
            gamma[a] = 1e3 * max(s1_sq, s2_sq)
        else:
            gamma[a] = s2_sq * (0.5 ** (k - 2))

    inter_QQ = Q[:, None] & Q[None, :]
    inter_flat = inter_QQ.ravel()

    for _ in range(n_em):
        zg = zeta_sub(gamma, n)
        K = zg[inter_QQ].copy()
        eps = max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
        K[np.diag_indices_from(K)] += eps
        K_inv = np.linalg.inv(K)
        alpha = K_inv @ y

        sc_a = np.zeros(N_m, dtype=np.float64)
        sc_a[Q] = alpha
        m_hat = gamma * zeta_sup(sc_a, n)

        sc_kinv = np.bincount(inter_flat, weights=K_inv.ravel(), minlength=N_m)
        sig_diag = np.maximum(0.0, gamma - (gamma ** 2) * zeta_sup(sc_kinv, n))
        gamma = np.maximum(1e-12, m_hat ** 2 + sig_diag)
        gamma[0] = 0.0

    zg = zeta_sub(gamma, n)
    K = zg[inter_QQ].copy()
    eps = max(1e-10, 1e-9 * float(np.mean(np.diag(K))))
    K[np.diag_indices_from(K)] += eps
    K_inv = np.linalg.inv(K)
    alpha = K_inv @ y
    sc_a = np.zeros(N_m, dtype=np.float64)
    sc_a[Q] = alpha
    m_hat = gamma * zeta_sup(sc_a, n)
    phi_hat = W @ m_hat
    return phi_hat, gamma, Q, K, K_inv, W

def propose_b1(gid, n):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(gid)) % 100000
    rng = np.random.RandomState(seed)
    full = (1 << n) - 1
    masks = np.arange(1 << n, dtype=np.int64)
    sizes = popcount_arr(masks, n)
    qs = [int(m) for m in masks[(sizes <= 2) | (sizes >= n - 2)]]
    seen = set(qs)
    mid = [int(m) for m in masks[(sizes >= 3) & (sizes <= 5)]]
    rng.shuffle(mid)
    for m in mid:
        mc = full ^ m
        if m not in seen and mc not in seen and len(qs) + 2 <= 140:
            seen.add(m); seen.add(mc)
            qs.extend([m, mc])
    return {"queries_b1": qs[:140], "want_b2": True}

def propose_b2(gid, n, observed, rem):
    if rem <= 0:
        return {"queries_b2": []}
    phi_hat, gamma, Q, K, K_inv, W = fit_bemc_gp(n, observed, n_em=5)
    N_m = 1 << n
    seen = {int(m) for m, _ in observed}
    unqueried = np.array([m for m in range(1, N_m - 1) if m not in seen], dtype=np.int64)

    # Exact subset inclusion matrix Z_all[t, a] = 1[a subseteq t]
    all_a = np.arange(N_m, dtype=np.int64)
    Z_Q = ((Q[:, None] & all_a[None, :]) == all_a[None, :]).astype(np.float64)  # [|Q|, 1024]
    WG = W * gamma[None, :]  # [55, 1024]
    C_WQ = WG @ Z_Q.T  # [55, |Q|] = Cov(Phi, v_Q)
    prior_var_phi = np.sum(W * WG, axis=1)
    V_e = np.maximum(1e-12, prior_var_phi - np.sum((C_WQ @ K_inv) * C_WQ, axis=1))

    abs_phi = np.sort(np.abs(phi_hat))[::-1]
    tau = 0.5 * (abs_phi[9] + abs_phi[10])
    omega = np.exp(-((np.abs(phi_hat) - tau) ** 2) / (2.0 * V_e))  # [55]

    # Candidate covariance Cov(Phi, v_T) and K_{Q, T} for all unqueried T
    Z_U = ((unqueried[:, None] & all_a[None, :]) == all_a[None, :]).astype(np.float64)  # [U, 1024]
    zg = zeta_sub(gamma, n)
    K_QU = zg[Q[:, None] & unqueried[None, :]]  # [|Q|, U]
    var_U = np.maximum(1e-10, zg[unqueried] - np.sum((K_inv @ K_QU) * K_QU, axis=0))  # [U]
    C_WU = WG @ Z_U.T - C_WQ @ (K_inv @ K_QU)  # [55, U] = posterior Cov(Phi, v_T | y_Q)

    # Greedy Batch-2 acquisition with exact rank-1 posterior covariance downdates!
    K_UU = zg[unqueried[:, None] & unqueried[None, :]] - K_QU.T @ (K_inv @ K_QU)  # [U, U]
    avail = np.ones(len(unqueried), dtype=bool)
    picked = []
    for _ in range(min(rem, len(unqueried))):
        scores = (omega @ (C_WU ** 2)) / np.maximum(1e-12, var_U)
        scores[~avail] = -1.0
        idx = int(np.argmax(scores))
        avail[idx] = False
        picked.append(int(unqueried[idx]))
        # Exact rank-1 conditioning on v_{U[idx]}!
        v_piv = max(1e-12, float(var_U[idx]))
        k_col = K_UU[:, idx].copy()  # [U]
        c_col = C_WU[:, idx].copy()  # [55]
        C_WU = C_WU - np.outer(c_col, k_col) / v_piv
        K_UU = K_UU - np.outer(k_col, k_col) / v_piv
        var_U = np.maximum(1e-12, np.diag(K_UU))
    return {"queries_b2": picked[:rem]}

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
        elif step == "propose_b2":
            preds.append({"id": it["id"], "prediction": propose_b2(it["id"], int(it["n"]), it["observed"], int(it["remaining_budget"]))})
        elif step == "estimate":
            phi_hat, *_ = fit_bemc_gp(int(it["n"]), it["observed"], n_em=6)
            preds.append({"id": it["id"], "prediction": phi_hat.tolist()})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
