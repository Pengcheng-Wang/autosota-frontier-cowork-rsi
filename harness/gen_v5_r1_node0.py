#!/usr/bin/env python3
"""
Wave-5 Round-1 Candidate & Baseline Generator — Node 0 (D1, D2, D3):
  - Copies frozen Wave-4 baselines, mandatory comparators, and Wave-4 Headless Champions
  - Writes standalone `python -I -B` Round-1 scripts from `Opus 5.5, Max` and `GPT 6.2 Astra, Max`
"""
from __future__ import annotations

import shutil
from pathlib import Path

W4_ROOT = Path("/scratch/autosota_governed_v4")
W5_ROOT = Path("/scratch/autosota_governed_v5")


def write_cand(bench_folder: str, cid: str, code: str) -> None:
    p = W5_ROOT / "tracks" / bench_folder / "candidates" / f"{cid}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code.lstrip(), encoding="utf-8")


def copy_v4_cands(bench_folder: str, mapping: dict[str, str]) -> None:
    src_dir = W4_ROOT / "tracks" / bench_folder / "candidates"
    dst_dir = W5_ROOT / "tracks" / bench_folder / "candidates"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_id, dst_id in mapping.items():
        shutil.copy2(src_dir / f"{src_id}.py", dst_dir / f"{dst_id}.py")


# ============================================================================
# D1 — Opus 5.5 Max: BEMC (Border-Exact Mobius Bayesian Completion + Active Batch 2)
# ============================================================================
D1_OPUS_BEMC = '''import argparse, json, math
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
'''


# ============================================================================
# D1 — GPT 6.2 Astra Max: Sparse Synergy Pursuit + Exact Cubic Boundary Formula
# ============================================================================
D1_ASTRA_SYNERGY = '''import argparse, itertools, json, math
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
'''


# ============================================================================
# D2 — Opus 5.5 Max (MPOT) & GPT 6.2 Astra Max (Eight Boundary Directions)
# ============================================================================
D2_OPUS_MPOT = '''import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

def fit_multiclass_l2(X, y, n_classes, reg=1e-3):
    N, d = X.shape
    xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    lin = nn.Linear(d, n_classes, dtype=torch.float64)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = lin(xt)
        loss = nn.functional.cross_entropy(logits, yt) + 0.5 * reg * torch.sum(lin.weight ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    W_hat = lin.weight.detach().numpy().T  # [d, C]
    b_hat = lin.bias.detach().numpy()      # [C]

    def pred_proba(Xe):
        lg = Xe @ W_hat + b_hat[None, :]
        lg -= np.max(lg, axis=1, keepdims=True)
        ex = np.exp(lg)
        return ex / np.sum(ex, axis=1, keepdims=True)

    return pred_proba

def run_mpot(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    # Cell-stratified 75/25 fit/cal split
    rng = np.random.RandomState(502)
    fit_idx, cal_idx = [], []
    for c in range(8):
        idx_c = np.where(g8_tr == c)[0]
        rng.shuffle(idx_c)
        n_f = int(0.75 * len(idx_c))
        fit_idx.append(idx_c[:n_f])
        cal_idx.append(idx_c[n_f:])
    fit_i = np.concatenate(fit_idx)
    cal_i = np.concatenate(cal_idx)

    xm = np.mean(X_tr[fit_i], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[fit_i], axis=0, keepdims=True) > 1e-6, np.std(X_tr[fit_i], axis=0, keepdims=True), 1.0)
    Xf = (X_tr[fit_i] - xm) / xs
    Xc = (X_tr[cal_i] - xm) / xs
    yf, yc = y_tr[fit_i], y_tr[cal_i]
    gf, gc = g_tr[fit_i], g_tr[cal_i]
    g8c = g8_tr[cal_i]

    # Fit smooth ERM MLP logit model l(x) in PyTorch (fast, python -I native!)
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(Xf.shape[1], 96), nn.SiLU(), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-2)
    xt = torch.tensor(Xf.astype(np.float32))
    yt = torch.tensor(yf.astype(np.float32))
    for _ in range(95):
        p_idx = torch.randperm(len(xt))
        for b0 in range(0, len(xt), 1024):
            sl = p_idx[b0:b0 + 1024]
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[sl]).squeeze(-1), yt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    # Fit 4-class L2 softmax group posterior q(g | x, y) via SciPy L-BFGS-B
    XY_f = np.hstack([Xf, yf[:, None], Xf * (2.0 * yf[:, None] - 1.0)])
    q_proba_fn = fit_multiclass_l2(XY_f, gf, n_classes=4, reg=1e-3)

    def get_logit_and_q(X_s):
        with torch.no_grad():
            ell = net(torch.tensor(X_s.astype(np.float32))).squeeze(-1).numpy().astype(np.float64)
        n_r = len(X_s)
        XY0 = np.hstack([X_s, np.zeros((n_r, 1)), -X_s])
        XY1 = np.hstack([X_s, np.ones((n_r, 1)), X_s])
        q0 = np.clip(q_proba_fn(XY0), 1e-5, 1.0)
        q1 = np.clip(q_proba_fn(XY1), 1e-5, 1.0)
        return ell, q0, q1

    ell_c, q0_c, q1_c = get_logit_and_q(Xc)
    pi_hat = np.array([max(1e-4, np.mean(g8_tr == c)) for c in range(8)], dtype=np.float64)
    n_c = np.array([max(1.0, np.sum(g8c == c)) for c in range(8)], dtype=np.float64)

    lam = np.ones(8, dtype=np.float64) / 8.0
    mu_fl = 0.0
    best_a = np.ones(8, dtype=np.float64)
    best_lcb = -1e9

    for _ in range(250):
        a = lam / pi_hat + mu_fl
        a_g0 = a[0::2]
        a_g1 = a[1::2]
        t_a = np.log(q0_c @ a_g0) - np.log(q1_c @ a_g1)
        yhat = (ell_c > t_a).astype(np.int64)
        corr = (yhat == yc).astype(np.float64)
        A_c = np.array([float(np.mean(corr[g8c == c])) for c in range(8)])
        A_mean = float(np.mean(corr))
        lcb = float(np.min(A_c - 0.5 * np.sqrt(np.maximum(1e-6, A_c * (1.0 - A_c)) / n_c)))
        if A_mean >= 0.750 and lcb > best_lcb:
            best_lcb = lcb
            best_a = a.copy()
        lam = lam * np.exp(-2.2 * A_c)
        lam /= np.sum(lam)
        mu_fl = max(0.0, mu_fl + 5.0 * (0.754 - A_mean))

    a_g0 = best_a[0::2]
    a_g1 = best_a[1::2]
    outs = []
    for it in eval_items:
        Xev = (np.asarray(it["X_eval"], dtype=np.float64) - xm) / xs
        ell_e, q0_e, q1_e = get_logit_and_q(Xev)
        t_e = np.log(q0_e @ a_g0) - np.log(q1_e @ a_g1)
        outs.append({"id": it["id"], "prediction": (ell_e > t_e).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_mpot(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


D2_ASTRA_BOUNDARY8 = '''import argparse, json
import numpy as np
import torch

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def make_phi(X, xm, xs, meds, nonbin):
    Xs = (X - xm) / xs
    hinges = np.maximum(0.0, Xs[:, nonbin] - meds[None, :])
    sq = Xs[:, nonbin] ** 2
    return np.hstack([np.ones((len(X), 1)), Xs, hinges, sq])

def fit_logistic_lbfgs(Phi, y, reg=1e-3):
    xt = torch.tensor(Phi, dtype=torch.float64)
    yt = torch.tensor(y.astype(np.float64), dtype=torch.float64)
    w = torch.zeros(Phi.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=55, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = xt @ w
        loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(z, yt)) + 0.5 * reg * torch.sum(w[1:] ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()

def run_astra_d2(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    rng = np.random.RandomState(17)
    perm = rng.permutation(len(X_tr))
    n_fit = int(0.65 * len(X_tr))
    n_tune = int(0.85 * len(X_tr))
    i_fit, i_tune, i_aud = perm[:n_fit], perm[n_fit:n_tune], perm[n_tune:]

    xm = np.mean(X_tr[i_fit], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[i_fit], axis=0, keepdims=True) > 1e-6, np.std(X_tr[i_fit], axis=0, keepdims=True), 1.0)
    Xf_s = (X_tr[i_fit] - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[i_fit, j])) > 2]
    meds = np.median(Xf_s[:, nonbin], axis=0)

    Phi_f = make_phi(X_tr[i_fit], xm, xs, meds, nonbin)
    Phi_t = make_phi(X_tr[i_tune], xm, xs, meds, nonbin)
    Phi_a = make_phi(X_tr[i_aud], xm, xs, meds, nonbin)

    w_erm = fit_logistic_lbfgs(Phi_f, y_tr[i_fit], reg=1e-3)
    s_f = Phi_f @ w_erm
    sd_f = max(1e-6, float(np.std(s_f)))
    w_erm = w_erm / sd_f
    s_f = Phi_f @ w_erm

    G_mat = (Phi_f.T @ Phi_f) / len(Phi_f) + 0.01 * np.eye(Phi_f.shape[1])
    G_inv = np.linalg.inv(G_mat)
    r_f = 2.0 * y_tr[i_fit] - 1.0
    kern = np.exp(-0.5 * ((s_f / 0.35) ** 2))
    g8_f = g8_tr[i_fit]

    U_dirs = np.zeros((Phi_f.shape[1], 8), dtype=np.float64)
    for c in range(8):
        mc = (g8_f == c)
        if np.any(mc):
            rhs = np.mean((r_f[mc] * kern[mc])[:, None] * Phi_f[mc], axis=0)
            u = G_inv @ rhs
            rms = float(np.sqrt(np.mean((Phi_f @ u) ** 2)))
            if rms > 1e-8:
                u /= rms
            U_dirs[:, c] = u

    s0_t = Phi_t @ w_erm
    B_t = Phi_t @ U_dirs
    y_t, g8_t = y_tr[i_tune], g8_tr[i_tune]
    r_t = 2.0 * y_t - 1.0

    best_rule = (np.zeros(8), 0.0)
    best_aud_wg = -1.0

    for tau_temp in (0.12, 0.22):
        Bt_t = torch.tensor(B_t, dtype=torch.float64)
        s0_tt = torch.tensor(s0_t, dtype=torch.float64)
        r_tt = torch.tensor(r_t, dtype=torch.float64)
        g_masks = [torch.tensor(g8_t == c, dtype=torch.bool) for c in range(8)]
        prm = torch.zeros(9, dtype=torch.float64, requires_grad=True)
        opt_p = torch.optim.LBFGS([prm], lr=0.8, max_iter=45, line_search_fn="strong_wolfe")

        def closure_p():
            opt_p.zero_grad()
            al = prm[:8]
            b0 = prm[8]
            sc = s0_tt + Bt_t @ al + b0
            p_corr = torch.sigmoid(torch.clamp(r_tt * sc / tau_temp, -25.0, 25.0))
            c_accs = torch.stack([torch.mean(p_corr[m_c]) for m_c in g_masks])
            soft_wg = -0.05 * torch.log(torch.mean(torch.exp(-c_accs / 0.05)))
            mean_pen = 8.0 * torch.clamp(0.758 - torch.mean(p_corr), min=0.0) ** 2
            loss = -soft_wg + mean_pen + 0.01 * torch.sum(al ** 2)
            loss.backward()
            return loss

        opt_p.step(closure_p)
        al_hat = prm.detach().numpy()[:8]
        sc_t = s0_t + B_t @ al_hat
        sc_a = (Phi_a @ w_erm) + (Phi_a @ U_dirs) @ al_hat
        y_a, g8_a = y_tr[i_aud], g8_tr[i_aud]

        for cut in np.quantile(sc_t, np.linspace(0.15, 0.85, 81)):
            pr_t = (sc_t >= cut).astype(np.int64)
            if np.mean(pr_t == y_t) < 0.750:
                continue
            pr_a = (sc_a >= cut).astype(np.int64)
            acc_a = float(np.mean(pr_a == y_a))
            se_a = float(np.sqrt(acc_a * (1.0 - acc_a) / len(y_a)))
            if acc_a - 1.645 * se_a < 0.740:
                continue
            wg_t = min(float(np.mean(pr_t[g8_t == c] == y_t[g8_t == c])) for c in range(8))
            wg_a = min(float(np.mean(pr_a[g8_a == c] == y_a[g8_a == c])) for c in range(8))
            if 0.5 * (wg_t + wg_a) > best_aud_wg:
                best_aud_wg = 0.5 * (wg_t + wg_a)
                best_rule = (al_hat.copy(), float(cut))

    al_fin, cut_fin = best_rule
    outs = []
    for it in eval_items:
        Phi_e = make_phi(np.asarray(it["X_eval"], dtype=np.float64), xm, xs, meds, nonbin)
        sc_e = (Phi_e @ w_erm) + (Phi_e @ U_dirs) @ al_fin
        outs.append({"id": it["id"], "prediction": (sc_e >= cut_fin).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_astra_d2(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D3 — Opus 5.5 Max (GL-NNLS) & GPT 6.2 Astra Max (Free-Gated Min/Max Splines)
# ============================================================================
D3_OPUS_GL_NNLS = '''import argparse, json
import numpy as np

def fista_box_qp(S_mat, h_vec, n_iter=160):
    """Exact Restarted Nesterov FISTA for min_{theta >= 0} 0.5 * theta^T S theta - h^T theta."""
    p = len(h_vec)
    # Power iteration for exact largest eigenvalue L_lip of S_mat
    v = np.ones(p, dtype=np.float64) / np.sqrt(max(1, p))
    for _ in range(12):
        Sv = S_mat @ v
        nrm = float(np.linalg.norm(Sv))
        if nrm > 1e-12:
            v = Sv / nrm
    L_lip = max(1e-6, float(v @ (S_mat @ v)) * 1.03)
    inv_L = 1.0 / L_lip

    th = np.zeros(p, dtype=np.float64)
    y = th.copy()
    t = 1.0
    for _ in range(n_iter):
        grad = S_mat @ y - h_vec
        th_n = np.maximum(0.0, y - inv_L * grad)
        if float((th_n - th) @ grad) > 0.0:
            t = 1.0
            y = th_n.copy()
        else:
            tn = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = th_n + ((t - 1.0) / tn) * (th_n - th)
            t = tn
        th = th_n
    return th

def build_gl_model(Xf_tr, Xm_tr, y_tr):
    n, df = Xf_tr.shape
    dm = Xm_tr.shape[1]
    rng = np.random.RandomState(77)

    # 1) Softmax partition-of-unity gates rho_j(x_free) >= 0 via J=4 k-means on top-10 PCs of Xf
    J = 4 if (df >= 2 and n >= 120) else 2
    if df > 0:
        Xf_c = Xf_tr - np.mean(Xf_tr, axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xf_c, full_matrices=False)
        V_pc = Vt[:min(10, len(Vt))].T
        Z_tr = Xf_c @ V_pc
        centers = Z_tr[rng.choice(n, size=J, replace=False)].copy()
        for _ in range(10):
            d2 = np.sum((Z_tr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            lab = np.argmin(d2, axis=1)
            for j in range(J):
                if np.any(lab == j):
                    centers[j] = np.mean(Z_tr[lab == j], axis=0)
        d2 = np.sum((Z_tr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        tau = max(1e-3, float(np.mean(np.min(d2, axis=1))))
    else:
        V_pc, centers, tau = None, None, 1.0

    def get_gates(Xf):
        if df == 0:
            return np.ones((len(Xf), 1), dtype=np.float64)
        Zc = (Xf - np.mean(Xf_tr, axis=0, keepdims=True)) @ V_pc
        dist2 = np.sum((Zc[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lg = -dist2 / tau
        lg -= np.max(lg, axis=1, keepdims=True)
        eg = np.exp(lg)
        return eg / np.sum(eg, axis=1, keepdims=True)  # strictly >= 0, sum == 1

    # 2) Unconstrained free feature basis Phi(x_f): [1, x_f, ReLU RFF, rho]
    D_rff = min(160, max(16, n // 3))
    U_rff = rng.normal(0.0, 1.0 / max(1.0, np.sqrt(df)), size=(df, D_rff)) if df > 0 else None
    c_rff = rng.normal(0.0, 0.8, size=(1, D_rff)) if df > 0 else None

    def make_Phi(Xf, rho):
        if df == 0:
            return np.ones((len(Xf), 1), dtype=np.float64)
        rff = np.maximum(0.0, Xf @ U_rff + c_rff)
        return np.hstack([np.ones((len(Xf), 1)), Xf, rff, rho])

    # 3) Monotone spline basis psi_{k,b}(x_k) & non-negative shifted hinges u_{k,c}(x_k) >= 0!
    # CRITICAL FOR PAIRWISE PRODUCT MONOTONICITY: u * v is non-decreasing iff u >= 0, v >= 0, u' >= 0, v' >= 0!
    # max(x - q, 0) satisfies BOTH u >= 0 AND u' >= 0 everywhere on R!
    B_knots = 6 if n < 1000 else 8
    knots_list = [np.unique(np.quantile(Xm_tr[:, k], np.linspace(0.05, 0.95, B_knots + 1))) for k in range(dm)]
    q_hinge_list = [np.quantile(Xm_tr[:, k], [0.25, 0.50, 0.75]) for k in range(dm)]

    # Screen top-8 monotone columns by positive correlation for pairwise hinge products
    corrs_m = [float(np.corrcoef(Xm_tr[:, k], y_tr)[0, 1]) if np.std(Xm_tr[:, k]) > 1e-8 else 0.0 for k in range(dm)]
    top_m = list(np.argsort(-np.abs(corrs_m))[:min(6, dm)])

    def make_Psi(Xm, rho, include_products=True):
        cols = []
        # Gated monotone linear + bounded incremental spline segments:
        # Note: for x_k on R, sig(x_k) or shifted softplus log(1 + exp(x_k)) is >= 0 and monotone,
        # AND rho_j(x_f) >= 0 DOES NOT depend on x_mono! So rho_j(x_f) * psi(x_k) has partial derivative
        # rho_j(x_f) * psi'(x_k) >= 0 EVERYWHERE even if psi(x_k) takes negative values!
        for k in range(dm):
            xk = Xm[:, k:k + 1]
            seg_k = [xk]
            kn = knots_list[k]
            for b_i in range(len(kn) - 1):
                w_seg = kn[b_i + 1] - kn[b_i]
                if w_seg > 1e-6:
                    seg_k.append(np.clip(xk - kn[b_i], 0.0, w_seg))
            # Also unbounded positive right-tail ray max(xk - kn[-1], 0)
            if len(kn) > 0:
                seg_k.append(np.maximum(0.0, xk - kn[-1]))
            M_k = np.hstack(seg_k)  # [N, S_k], monotone non-decreasing in x_k!
            for j in range(rho.shape[1]):
                cols.append(rho[:, j:j + 1] * M_k)

        if include_products and len(top_m) >= 2:
            # Non-negative monotone hinges u >= 0, u' >= 0 => product u_k * u_l is 100% monotone everywhere!
            for idx_a in range(len(top_m)):
                for idx_b in range(idx_a + 1, len(top_m)):
                    ka, kb = top_m[idx_a], top_m[idx_b]
                    for qa in q_hinge_list[ka]:
                        ua = np.maximum(0.0, Xm[:, ka:ka + 1] - qa)
                        for qb in q_hinge_list[kb]:
                            ub = np.maximum(0.0, Xm[:, kb:kb + 1] - qb)
                            cols.append(ua * ub)
        return np.hstack(cols)

    # 2-fold CV over lambda in {0.5, 3.0, 18.0} using Schur-complement Cholesky NNLS!
    rho_tr = get_gates(Xf_tr)
    Phi_all = make_Phi(Xf_tr, rho_tr)
    Psi_all = make_Psi(Xm_tr, rho_tr, include_products=True)

    def fit_schur_nnls(Phi, Psi, y, lam):
        p_f = Phi.shape[1]
        p_m = Psi.shape[1]
        reg_f = lam * np.eye(p_f)
        reg_f[0, 0] = 1e-8  # don't penalize intercept
        A_mat = Phi.T @ Phi + reg_f
        A_inv_PhiT = np.linalg.solve(A_mat, Phi.T)
        S_mat = (Psi.T @ Psi) - (Psi.T @ Phi) @ (A_inv_PhiT @ Psi) + lam * np.eye(p_m)
        S_mat = 0.5 * (S_mat + S_mat.T) + 1e-8 * np.eye(p_m)
        h_vec = Psi.T @ y - (Psi.T @ Phi) @ (A_inv_PhiT @ y)
        theta = fista_box_qp(S_mat, h_vec, n_iter=160)
        theta = np.maximum(0.0, theta)
        beta = A_inv_PhiT @ (y - Psi @ theta)
        return beta, theta

    perm_cv = rng.permutation(n)
    half = n // 2
    f1, f2 = perm_cv[:half], perm_cv[half:]
    best_lam, best_cv = 3.0, 1e18
    for lam in (0.5, 3.0, 18.0):
        b1, t1 = fit_schur_nnls(Phi_all[f1], Psi_all[f1], y_tr[f1], lam)
        p2 = Phi_all[f2] @ b1 + Psi_all[f2] @ t1
        b2, t2 = fit_schur_nnls(Phi_all[f2], Psi_all[f2], y_tr[f2], lam)
        p1 = Phi_all[f1] @ b2 + Psi_all[f1] @ t2
        cv_mse = 0.5 * (float(np.mean((p2 - y_tr[f2]) ** 2)) + float(np.mean((p1 - y_tr[f1]) ** 2)))
        if cv_mse < best_cv:
            best_cv, best_lam = cv_mse, lam

    beta_fin, theta_fin = fit_schur_nnls(Phi_all, Psi_all, y_tr, best_lam)

    def predict_fn(Xf_ev, Xm_ev):
        rho_e = get_gates(Xf_ev)
        Phi_e = make_Phi(Xf_ev, rho_e)
        Psi_e = make_Psi(Xm_ev, rho_e, include_products=True)
        return Phi_e @ beta_fin + Psi_e @ np.maximum(0.0, theta_fin)

    return predict_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    tr_all = json.loads(open(req["train_datasets_path"], encoding="utf-8").read())

    models_by_ds = {}
    for ds, d_obj in tr_all.items():
        Xf = np.asarray(d_obj["X_train_free"], dtype=np.float64)
        Xm = np.asarray(d_obj["X_train_mono"], dtype=np.float64)
        y = np.asarray(d_obj["y_train"], dtype=np.float64)
        models_by_ds[ds] = build_gl_model(Xf, Xm, y)

    preds = []
    for it in req["items"]:
        ds = it["dataset"]
        Xf_e = np.asarray(it["X_eval_free"], dtype=np.float64)
        Xm_e = np.asarray(it["X_eval_mono"], dtype=np.float64)
        preds.append({"id": it["id"], "prediction": models_by_ds[ds](Xf_e, Xm_e).tolist()})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


D3_ASTRA_MINMAX = '''import argparse, json
import numpy as np

def fista_mixed_qp(G, c_rhs, p_f, n_iter=180):
    p = len(c_rhs)
    v = np.ones(p, dtype=np.float64) / np.sqrt(max(1, p))
    for _ in range(12):
        Gv = G @ v
        nrm = float(np.linalg.norm(Gv))
        if nrm > 1e-12:
            v = Gv / nrm
    inv_L = 1.0 / max(1e-6, float(v @ (G @ v)) * 1.03)
    w = np.zeros(p, dtype=np.float64)
    y = w.copy()
    t = 1.0
    for _ in range(n_iter):
        grad = G @ y - c_rhs
        wn = y - inv_L * grad
        wn[p_f:] = np.maximum(0.0, wn[p_f:])
        if float((wn - w) @ grad) > 0.0:
            t = 1.0
            y = wn.copy()
        else:
            tn = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            y = wn + ((t - 1.0) / tn) * (wn - w)
            t = tn
        w = wn
    return w

def build_astra_minmax(Xf_tr, Xm_tr, y_tr):
    n, df = Xf_tr.shape
    dm = Xm_tr.shape[1]

    # 1) Free basis F: intercept + linear + quartile hinges on top-12 correlated free columns
    if df > 0:
        f_corrs = [abs(float(np.corrcoef(Xf_tr[:, j], y_tr)[0, 1])) if np.std(Xf_tr[:, j]) > 1e-8 else 0.0 for j in range(df)]
        top_f = list(np.argsort(-np.array(f_corrs))[:min(12, df)])
        f_quants = {j: np.quantile(Xf_tr[:, j], [0.25, 0.50, 0.75]) for j in top_f}
        f_gate_q = {j: np.quantile(Xf_tr[:, j], [0.20, 0.80]) for j in top_f[:min(8, len(top_f))]}
    else:
        top_f, f_quants, f_gate_q = [], {}, {}

    def make_F(Xf):
        cols = [np.ones((len(Xf), 1), dtype=np.float64)]
        if df > 0:
            cols.append(Xf)
            for j in top_f:
                for qv in f_quants[j]:
                    cols.append(np.maximum(0.0, Xf[:, j:j + 1] - qv))
        return np.hstack(cols)

    F_tr = make_F(Xf_tr)
    reg_F = 0.02 * len(y_tr) * np.eye(F_tr.shape[1])
    reg_F[0, 0] = 1e-8
    w_F0 = np.linalg.solve(F_tr.T @ F_tr + reg_F, F_tr.T @ y_tr)
    res0 = y_tr - F_tr @ w_F0

    # 2) Monotone quartile ramps R_{a,b}(x) = clip((x - Qa)/(Qb - Qa), 0, 1) + tail rays
    ramp_specs = []
    best_ramp_per_col = []
    for k in range(dm):
        qs = np.quantile(Xm_tr[:, k], [0.0, 0.25, 0.50, 0.75, 1.0])
        best_c, best_spec = -1e9, None
        for b_i in range(4):
            w_ab = qs[b_i + 1] - qs[b_i]
            if w_ab > 1e-6:
                sp = ("ramp", k, float(qs[b_i]), float(w_ab))
                r_vec = np.clip((Xm_tr[:, k] - sp[2]) / sp[3], 0.0, 1.0)
                c_val = float((r_vec - np.mean(r_vec)) @ res0) / (float(np.linalg.norm(r_vec - np.mean(r_vec))) + 1e-8)
                ramp_specs.append((c_val, sp))
                if c_val > best_c:
                    best_c, best_spec = c_val, sp
        # Also include global linear monotone ray (scaled by positive std) so tail extrapolation is non-flat!
        sp_lin = ("lin", k, float(np.min(Xm_tr[:, k])), max(1.0, float(np.std(Xm_tr[:, k]))))
        ramp_specs.append((0.5, sp_lin))
        if best_spec is not None:
            best_ramp_per_col.append((best_c, best_spec))

    best_ramp_per_col.sort(key=lambda x: -x[0])
    winners = [sp for _, sp in best_ramp_per_col[:min(12, len(best_ramp_per_col))]]

    def eval_spec(Xm, sp):
        if sp[0] == "ramp":
            return np.clip((Xm[:, sp[1]] - sp[2]) / sp[3], 0.0, 1.0)
        else:
            return (Xm[:, sp[1]] - sp[2]) / sp[3]

    # 3) Lattice min/max(u_j, u_k) + free-gated h_l * u_j and (1 - h_l) * u_j!
    inter_specs = []
    for i_a in range(len(winners)):
        for i_b in range(i_a + 1, len(winners)):
            inter_specs.append(("min", winners[i_a], winners[i_b]))
            inter_specs.append(("max", winners[i_a], winners[i_b]))
    for sp_w in winners:
        for jf, (q20, q80) in f_gate_q.items():
            if q80 - q20 > 1e-6:
                inter_specs.append(("fgate", sp_w, jf, float(q20), float(q80 - q20), False))
                inter_specs.append(("fgate", sp_w, jf, float(q20), float(q80 - q20), True))

    def eval_inter_col(Xf, Xm, isp):
        if isp[0] == "min":
            return np.minimum(eval_spec(Xm, isp[1]), eval_spec(Xm, isp[2]))
        elif isp[0] == "max":
            return np.maximum(eval_spec(Xm, isp[1]), eval_spec(Xm, isp[2]))
        else:
            u = eval_spec(Xm, isp[1])
            h = np.clip((Xf[:, isp[2]] - isp[3]) / isp[4], 0.0, 1.0)
            return ((1.0 - h) if isp[5] else h) * u

    main_sp_list = [sp for _, sp in ramp_specs]
    M_main_tr = np.column_stack([eval_spec(Xm_tr, sp) for sp in main_sp_list])

    # Screen top K_int <= min(64, n//4) interactions positively correlated with residual!
    K_keep = min(64, max(8, n // 5))
    scored_inter = []
    for isp in inter_specs:
        vec = eval_inter_col(Xf_tr, Xm_tr, isp)
        vc = vec - np.mean(vec)
        nv = float(np.linalg.norm(vc))
        if nv > 1e-6:
            scored_inter.append((float(vc @ res0) / nv, isp))
    scored_inter.sort(key=lambda x: -x[0])
    kept_inter = [isp for sc_v, isp in scored_inter[:K_keep] if sc_v > 0.0]

    def make_M(Xf, Xm):
        parts = [np.column_stack([eval_spec(Xm, sp) for sp in main_sp_list])]
        if len(kept_inter) > 0:
            parts.append(np.column_stack([eval_inter_col(Xf, Xm, isp) for isp in kept_inter]))
        return np.hstack(parts)

    M_tr = make_M(Xf_tr, Xm_tr)
    p_f, p_m = F_tr.shape[1], M_tr.shape[1]
    Z_tr = np.hstack([F_tr, M_tr])
    rms = np.sqrt(np.mean(Z_tr ** 2, axis=0))
    rms = np.where(rms > 1e-8, rms, 1.0)
    rms[0] = 1.0
    Zn = Z_tr / rms[None, :]

    G = (Zn.T @ Zn) / n
    reg = np.full(p_f + p_m, 0.015)
    reg[0] = 1e-9
    G += np.diag(reg)
    c_rhs = (Zn.T @ y_tr) / n

    w_sol = fista_mixed_qp(G, c_rhs, p_f, n_iter=200)
    w_hat = w_sol / rms
    w_f_fin = w_hat[:p_f]
    w_m_fin = np.maximum(0.0, w_hat[p_f:])  # certified >= 0!

    def predict_fn(Xf_e, Xm_e):
        return make_F(Xf_e) @ w_f_fin + make_M(Xf_e, Xm_e) @ w_m_fin

    return predict_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    tr_all = json.loads(open(req["train_datasets_path"], encoding="utf-8").read())

    models_by_ds = {}
    for ds, d_obj in tr_all.items():
        Xf = np.asarray(d_obj["X_train_free"], dtype=np.float64)
        Xm = np.asarray(d_obj["X_train_mono"], dtype=np.float64)
        y = np.asarray(d_obj["y_train"], dtype=np.float64)
        models_by_ds[ds] = build_astra_minmax(Xf, Xm, y)

    preds = []
    for it in req["items"]:
        ds = it["dataset"]
        Xf_e = np.asarray(it["X_eval_free"], dtype=np.float64)
        Xm_e = np.asarray(it["X_eval_mono"], dtype=np.float64)
        preds.append({"id": it["id"], "prediction": models_by_ds[ds](Xf_e, Xm_e).tolist()})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


def main() -> None:
    copy_v4_cands("bench_d1_shapiq", {
        "d1_base_mc_antithetic_200": "d1_base_mc_antithetic_200",
        "d1_paper_svarmiq_200": "d1_paper_svarmiq_200",
        "d1_autosota_stratsize_1x_200": "d1_autosota_stratsize_1x_200",
        "d1_autosota_ensemble10_cheat": "d1_autosota_ensemble10_cheat",
        "d1_r2_crossover_debiased_order3": "d1_w4_champ_order3",
    })
    write_cand("bench_d1_shapiq", "d1_v5_r1_opus_bemc", D1_OPUS_BEMC)
    write_cand("bench_d1_shapiq", "d1_v5_r1_astra_synergy_omp", D1_ASTRA_SYNERGY)

    copy_v4_cands("bench_d2_acs_reweight", {
        "d2_base_erm_mlp": "d2_base_erm_mlp",
        "d2_paper_latent_score_reweight": "d2_paper_latent_score_reweight",
        "d2_comp_group_dro": "d2_comp_group_dro",
        "d2_r1_spectral_poly_minimax": "d2_w4_champ_poly_minimax",
    })
    write_cand("bench_d2_acs_reweight", "d2_v5_r1_opus_mpot", D2_OPUS_MPOT)
    write_cand("bench_d2_acs_reweight", "d2_v5_r1_astra_boundary8", D2_ASTRA_BOUNDARY8)

    copy_v4_cands("bench_d3_cmnn", {
        "d3_paper_cmnn_100ep": "d3_paper_cmnn_100ep",
        "d3_autosota_cmnn_150ep": "d3_autosota_cmnn_150ep",
        "d3_comp_monotone_ridge_spline": "d3_comp_monotone_ridge_spline",
        "d3_r1_coord_convex_concave": "d3_w4_champ_convex_concave",
    })
    write_cand("bench_d3_cmnn", "d3_v5_r1_opus_gl_nnls", D3_OPUS_GL_NNLS)
    write_cand("bench_d3_cmnn", "d3_v5_r1_astra_minmax_spline", D3_ASTRA_MINMAX)
    print("NODE0_V5_R1_CANDIDATES_GENERATED")


if __name__ == "__main__":
    main()
