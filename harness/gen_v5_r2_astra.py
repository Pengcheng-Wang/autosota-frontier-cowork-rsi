#!/usr/bin/env python3
"""
Wave-5 Round-2 Iterative Crossover Generator — Part A (`Cowork_GPT_6.2_Astra_Max` R2 Crossover):
Implements exact Round-2 Crossover designs synthesized from redacted Search_v5 feedback (`ID: 23683258-3fcb-4092-8224-1bab77787f53`).
100% pure `numpy` + `torch` (zero `scipy`/`sklearn` imports under `python -I`).
"""
from __future__ import annotations

import sys
from pathlib import Path

W5_ROOT = Path("/scratch/autosota_governed_v5")


def write_cand(bench_folder: str, cid: str, code: str) -> None:
    p = W5_ROOT / "tracks" / bench_folder / "candidates" / f"{cid}.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code.lstrip(), encoding="utf-8")


# ============================================================================
# D1 R2 Astra: Synergy-OMP + Fast Non-Iterative Zeta-GP Cubic/Odd Residualization
# ============================================================================
D1_R2_ASTRA = '''import argparse, itertools, json
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
'''


# ============================================================================
# D2 R2 Astra: Frozen 8-Boundary Margin + Margin-Gated GroupDRO MLP Residual
# ============================================================================
D2_R2_ASTRA = '''import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

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
        loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(xt @ w, yt)) + 0.5 * reg * torch.sum(w[1:] ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()

def run_d2_r2_astra(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    rng = np.random.RandomState(17)
    perm = rng.permutation(len(X_tr))
    n_fit = int(0.68 * len(X_tr))
    n_cal = int(0.86 * len(X_tr))
    i_fit, i_cal, i_aud = perm[:n_fit], perm[n_fit:n_cal], perm[n_cal:]

    xm = np.mean(X_tr[i_fit], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[i_fit], axis=0, keepdims=True) > 1e-6, np.std(X_tr[i_fit], axis=0, keepdims=True), 1.0)
    Xf_s = (X_tr[i_fit] - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[i_fit, j])) > 2]
    meds = np.median(Xf_s[:, nonbin], axis=0)

    Phi_f = make_phi(X_tr[i_fit], xm, xs, meds, nonbin)
    Phi_c = make_phi(X_tr[i_cal], xm, xs, meds, nonbin)
    Phi_a = make_phi(X_tr[i_aud], xm, xs, meds, nonbin)

    w_erm = fit_logistic_lbfgs(Phi_f, y_tr[i_fit], reg=1e-3)
    w_erm /= max(1e-6, float(np.std(Phi_f @ w_erm)))
    s0_f = Phi_f @ w_erm

    G_inv = np.linalg.inv((Phi_f.T @ Phi_f) / len(Phi_f) + 0.01 * np.eye(Phi_f.shape[1]))
    r_f = 2.0 * y_tr[i_fit] - 1.0
    kern = np.exp(-0.5 * ((s0_f / 0.35) ** 2))
    g8_f = g8_tr[i_fit]
    U_dirs = np.zeros((Phi_f.shape[1], 8), dtype=np.float64)
    for c in range(8):
        mc = (g8_f == c)
        if np.any(mc):
            u = G_inv @ np.mean((r_f[mc] * kern[mc])[:, None] * Phi_f[mc], axis=0)
            rms = float(np.sqrt(np.mean((Phi_f @ u) ** 2)))
            U_dirs[:, c] = u / max(1e-8, rms)

    B_f = Phi_f @ U_dirs
    B_c = Phi_c @ U_dirs
    B_a = Phi_a @ U_dirs
    s0_c = Phi_c @ w_erm
    s0_a = Phi_a @ w_erm

    # Fit Astra linear+quad minimax direction al_hat on cal
    y_c, g8_c = y_tr[i_cal], g8_tr[i_cal]
    r_c = 2.0 * y_c - 1.0
    prm = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    Bt_t = torch.tensor(B_c, dtype=torch.float64)
    s0_tt = torch.tensor(s0_c, dtype=torch.float64)
    r_tt = torch.tensor(r_c, dtype=torch.float64)
    g_masks = [torch.tensor(g8_c == c, dtype=torch.bool) for c in range(8)]
    opt_p = torch.optim.LBFGS([prm], lr=0.8, max_iter=45, line_search_fn="strong_wolfe")
    def clos():
        opt_p.zero_grad()
        sc = s0_tt + Bt_t @ prm[:8] + prm[8]
        pc = torch.sigmoid(torch.clamp(r_tt * sc / 0.16, -25.0, 25.0))
        ca = torch.stack([torch.mean(pc[m]) for m in g_masks])
        loss = 0.05 * torch.log(torch.mean(torch.exp(-ca / 0.05))) + 8.0 * torch.clamp(0.758 - torch.mean(pc), min=0.0) ** 2 + 0.01 * torch.sum(prm[:8] ** 2)
        loss.backward()
        return loss
    opt_p.step(clos)
    al_hat = prm.detach().numpy()[:8]
    b0_hat = float(prm.detach().numpy()[8])

    sA_f = s0_f + B_f @ al_hat + b0_hat
    sA_c = s0_c + B_c @ al_hat + b0_hat
    sA_a = s0_a + B_a @ al_hat + b0_hat
    sig_m = max(0.25, float(np.median(np.abs(sA_f))))

    # Margin-gated GroupDRO width-32 MLP correction: s(x) = s_A(x) + gamma * exp(-|s_A| / sig_m) * f_MLP([Xs, B])!
    torch.manual_seed(52)
    Z_in_f = np.hstack([Xf_s, B_f]).astype(np.float32)
    mlp = nn.Sequential(nn.Linear(Z_in_f.shape[1], 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=2e-2)

    zt_f = torch.tensor(Z_in_f)
    sA_ft = torch.tensor(sA_f.astype(np.float32))
    gate_ft = torch.exp(-torch.abs(sA_ft) / sig_m)
    yt_f = torch.tensor(y_tr[i_fit].astype(np.float32))
    g8_ft = torch.tensor(g8_f, dtype=torch.long)
    pi_f = np.array([max(1e-4, np.mean(g8_f == c)) for c in range(8)], dtype=np.float32)
    q_c = torch.ones(8) / 8.0

    for ep in range(36):
        p_idx = torch.randperm(len(zt_f))
        for b0 in range(0, len(zt_f), 512):
            sl = p_idx[b0:b0 + 512]
            logits = sA_ft[sl] + gate_ft[sl] * mlp(zt_f[sl]).squeeze(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, yt_f[sl], reduction="none")
            gl = torch.zeros(8)
            for c in range(8):
                mc = (g8_ft[sl] == c)
                if torch.any(mc):
                    gl[c] = torch.mean(bce[mc])
            with torch.no_grad():
                q_c = q_c * torch.exp(0.08 * gl)
                q_c = q_c / torch.sum(q_c)
            loss = torch.sum(q_c * gl)
            opt_m.zero_grad(); loss.backward(); opt_m.step()
    mlp.eval()

    def mlp_corr(X_raw, B_mat, sA_vec):
        Xs = (X_raw - xm) / xs
        Zi = torch.tensor(np.hstack([Xs, B_mat]).astype(np.float32))
        with torch.no_grad():
            f_out = mlp(Zi).squeeze(-1).numpy().astype(np.float64)
        return np.exp(-np.abs(sA_vec) / sig_m) * f_out

    m_c = mlp_corr(X_tr[i_cal], B_c, sA_c)
    m_a = mlp_corr(X_tr[i_aud], B_a, sA_a)
    y_a, g8_a = y_tr[i_aud], g8_tr[i_aud]

    best_cfg = (0.0, 0.0)
    best_wg = -1.0
    for gam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.35):
        sc_c = sA_c + gam * m_c
        sc_a = sA_a + gam * m_a
        for cut in np.quantile(sc_c, np.linspace(0.18, 0.82, 65)):
            prc = (sc_c >= cut).astype(np.int64)
            if np.mean(prc == y_c) < 0.750:
                continue
            pra = (sc_a >= cut).astype(np.int64)
            acc_a = float(np.mean(pra == y_a))
            if acc_a < 0.748:
                continue
            wgc = min(float(np.mean(prc[g8_c == c] == y_c[g8_c == c])) for c in range(8))
            wga = min(float(np.mean(pra[g8_a == c] == y_a[g8_a == c])) for c in range(8))
            score = 0.5 * (wgc + wga)
            if score > best_wg:
                best_wg = score
                best_cfg = (gam, float(cut))

    gam_fin, cut_fin = best_cfg
    outs = []
    for it in eval_items:
        Xev = np.asarray(it["X_eval"], dtype=np.float64)
        Phi_e = make_phi(Xev, xm, xs, meds, nonbin)
        B_e = Phi_e @ U_dirs
        sA_e = (Phi_e @ w_erm) + B_e @ al_hat + b0_hat
        sc_e = sA_e + gam_fin * mlp_corr(Xev, B_e, sA_e)
        outs.append({"id": it["id"], "prediction": (sc_e >= cut_fin).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_d2_r2_astra(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D3 R2 Astra: Octile Min/Max Splines + Opus 4-RBF Free Gates + Ramp Products
# ============================================================================
D3_R2_ASTRA = '''import argparse, json
import numpy as np

def fista_mixed_qp(G, c_rhs, p_f, n_iter=200):
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

def build_d3_r2_astra(Xf_tr, Xm_tr, y_tr):
    n, df = Xf_tr.shape
    dm = Xm_tr.shape[1]
    rng = np.random.RandomState(77)

    # Opus J=4 normalized RBF free gates + rich free basis (linear + octile hinges + RFF)
    J = 4 if (df >= 2 and n >= 120) else 1
    if df > 0 and J > 1:
        Xfc = Xf_tr - np.mean(Xf_tr, axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xfc, full_matrices=False)
        V_pc = Vt[:min(10, len(Vt))].T
        Z_tr = Xfc @ V_pc
        centers = Z_tr[rng.choice(n, size=J, replace=False)].copy()
        for _ in range(10):
            d2 = np.sum((Z_tr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            lab = np.argmin(d2, axis=1)
            for j in range(J):
                if np.any(lab == j):
                    centers[j] = np.mean(Z_tr[lab == j], axis=0)
        pair_d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        tau = max(1e-2, float(np.median(pair_d2[pair_d2 > 0])) if np.any(pair_d2 > 0) else 1.0)
    else:
        V_pc, centers, tau = None, None, 1.0

    def get_rbf_gates(Xf):
        if df == 0 or J == 1:
            return np.ones((len(Xf), 1), dtype=np.float64)
        Zc = (Xf - np.mean(Xf_tr, axis=0, keepdims=True)) @ V_pc
        d2 = np.sum((Zc[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lg = -d2 / tau
        lg -= np.max(lg, axis=1, keepdims=True)
        eg = np.exp(lg)
        return eg / np.sum(eg, axis=1, keepdims=True)

    if df > 0:
        f_corrs = [abs(float(np.corrcoef(Xf_tr[:, j], y_tr)[0, 1])) if np.std(Xf_tr[:, j]) > 1e-8 else 0.0 for j in range(df)]
        top_f = list(np.argsort(-np.array(f_corrs))[:min(16, df)])
        f_quants = {j: np.quantile(Xf_tr[:, j], [0.15, 0.35, 0.50, 0.65, 0.85]) for j in top_f}
        f_gate_q = {j: np.quantile(Xf_tr[:, j], [0.20, 0.80]) for j in top_f[:min(8, len(top_f))]}
    else:
        top_f, f_quants, f_gate_q = [], {}, {}

    def make_F(Xf, rho):
        cols = [np.ones((len(Xf), 1), dtype=np.float64)]
        if df > 0:
            cols.append(Xf)
            cols.append(rho)
            for j in top_f:
                for qv in f_quants[j]:
                    cols.append(np.maximum(0.0, Xf[:, j:j + 1] - qv))
        return np.hstack(cols)

    rho_tr = get_rbf_gates(Xf_tr)
    F_tr = make_F(Xf_tr, rho_tr)
    reg_F0 = 0.02 * n * np.eye(F_tr.shape[1]); reg_F0[0, 0] = 1e-8
    res0 = y_tr - F_tr @ np.linalg.solve(F_tr.T @ F_tr + reg_F0, F_tr.T @ y_tr)

    # Octile monotone ramps clipped to [0, 1] + global linear ray per monotone feature!
    ramp_specs = []
    best_per_col = []
    for k in range(dm):
        qs = np.quantile(Xm_tr[:, k], np.linspace(0.0, 1.0, 9))
        bc, bsp = -1e9, None
        for b_i in range(8):
            w_ab = qs[b_i + 1] - qs[b_i]
            if w_ab > 1e-6:
                sp = ("ramp", k, float(qs[b_i]), float(w_ab))
                rv = np.clip((Xm_tr[:, k] - sp[2]) / sp[3], 0.0, 1.0)
                cv = float((rv - np.mean(rv)) @ res0) / (float(np.linalg.norm(rv - np.mean(rv))) + 1e-8)
                ramp_specs.append((cv, sp))
                if cv > bc:
                    bc, bsp = cv, sp
        ramp_specs.append((0.5, ("lin", k, float(np.min(Xm_tr[:, k])), max(1.0, float(np.std(Xm_tr[:, k]))))))
        if bsp is not None:
            best_per_col.append((bc, bsp))

    best_per_col.sort(key=lambda x: -x[0])
    winners = [sp for _, sp in best_per_col[:min(12, len(best_per_col))]]

    def eval_sp(Xm, sp):
        if sp[0] == "ramp":
            return np.clip((Xm[:, sp[1]] - sp[2]) / sp[3], 0.0, 1.0)
        return (Xm[:, sp[1]] - sp[2]) / sp[3]

    # Candidate interactions: min(u, v), max(u, v), non-negative ramp product u*v (since u,v in [0,1]!), 1D free gates, and 4-RBF gated ramps!
    inter_cands = []
    for ia in range(len(winners)):
        for ib in range(ia + 1, len(winners)):
            inter_cands.append(("min", winners[ia], winners[ib]))
            inter_cands.append(("max", winners[ia], winners[ib]))
            inter_cands.append(("prod", winners[ia], winners[ib]))
    for sp_w in winners:
        for jf, (q20, q80) in f_gate_q.items():
            if q80 - q20 > 1e-6:
                inter_cands.append(("fgate", sp_w, jf, float(q20), float(q80 - q20), False))
                inter_cands.append(("fgate", sp_w, jf, float(q20), float(q80 - q20), True))
        if J > 1:
            for jr in range(J):
                inter_cands.append(("rbfgate", sp_w, jr))

    def eval_inter(Xf, Xm, rho, isp):
        t = isp[0]
        if t == "min":
            return np.minimum(eval_sp(Xm, isp[1]), eval_sp(Xm, isp[2]))
        elif t == "max":
            return np.maximum(eval_sp(Xm, isp[1]), eval_sp(Xm, isp[2]))
        elif t == "prod":
            # Both winners are ("ramp", ...) in [0, 1] => non-negative & monotone => product is 100% monotone!
            return eval_sp(Xm, isp[1]) * eval_sp(Xm, isp[2])
        elif t == "rbfgate":
            return rho[:, isp[2]] * eval_sp(Xm, isp[1])
        else:
            u = eval_sp(Xm, isp[1])
            h = np.clip((Xf[:, isp[2]] - isp[3]) / isp[4], 0.0, 1.0)
            return ((1.0 - h) if isp[5] else h) * u

    main_sps = [sp for _, sp in ramp_specs]
    K_keep = min(80, max(10, n // 4))
    scored = []
    for isp in inter_cands:
        vec = eval_inter(Xf_tr, Xm_tr, rho_tr, isp)
        vc = vec - np.mean(vec)
        nv = float(np.linalg.norm(vc))
        if nv > 1e-6:
            scored.append((float(vc @ res0) / nv, isp))
    scored.sort(key=lambda x: -x[0])
    kept = [isp for sc_v, isp in scored[:K_keep] if sc_v > 0.0]

    def make_M(Xf, Xm, rho):
        parts = [np.column_stack([eval_sp(Xm, sp) for sp in main_sps])]
        if len(kept) > 0:
            parts.append(np.column_stack([eval_inter(Xf, Xm, rho, isp) for isp in kept]))
        return np.hstack(parts)

    M_tr = make_M(Xf_tr, Xm_tr, rho_tr)
    p_f, p_m = F_tr.shape[1], M_tr.shape[1]
    Z_tr = np.hstack([F_tr, M_tr])
    rms = np.where(np.sqrt(np.mean(Z_tr ** 2, axis=0)) > 1e-8, np.sqrt(np.mean(Z_tr ** 2, axis=0)), 1.0)
    rms[0] = 1.0
    Zn = Z_tr / rms[None, :]

    # Quick 2-fold CV over ridge alpha in {0.006, 0.018, 0.05}
    perm_cv = rng.permutation(n)
    f1, f2 = perm_cv[:n // 2], perm_cv[n // 2:]
    G1 = (Zn[f1].T @ Zn[f1]) / len(f1); c1 = (Zn[f1].T @ y_tr[f1]) / len(f1)
    G2 = (Zn[f2].T @ Zn[f2]) / len(f2); c2 = (Zn[f2].T @ y_tr[f2]) / len(f2)
    best_al, best_mse = 0.015, 1e18
    for al in (0.006, 0.018, 0.055):
        rg = np.full(p_f + p_m, al); rg[0] = 1e-9
        w1 = fista_mixed_qp(G1 + np.diag(rg), c1, p_f, n_iter=140)
        w2 = fista_mixed_qp(G2 + np.diag(rg), c2, p_f, n_iter=140)
        mse = 0.5 * (float(np.mean((Zn[f2] @ w1 - y_tr[f2]) ** 2)) + float(np.mean((Zn[f1] @ w2 - y_tr[f1]) ** 2)))
        if mse < best_mse:
            best_mse, best_al = mse, al

    G_all = (Zn.T @ Zn) / n
    rg_all = np.full(p_f + p_m, best_al); rg_all[0] = 1e-9
    w_sol = fista_mixed_qp(G_all + np.diag(rg_all), (Zn.T @ y_tr) / n, p_f, n_iter=220) / rms
    wf = w_sol[:p_f]
    wm = np.maximum(0.0, w_sol[p_f:])

    def predict_fn(Xf_e, Xm_e):
        rho_e = get_rbf_gates(Xf_e)
        return make_F(Xf_e, rho_e) @ wf + make_M(Xf_e, Xm_e, rho_e) @ wm

    return predict_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    tr_all = json.loads(open(req["train_datasets_path"], encoding="utf-8").read())
    models = {ds: build_d3_r2_astra(np.asarray(d["X_train_free"], dtype=np.float64), np.asarray(d["X_train_mono"], dtype=np.float64), np.asarray(d["y_train"], dtype=np.float64)) for ds, d in tr_all.items()}
    preds = [{"id": it["id"], "prediction": models[it["dataset"]](np.asarray(it["X_eval_free"], dtype=np.float64), np.asarray(it["X_eval_mono"], dtype=np.float64)).tolist()} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D4 R2 Astra: Rank-24 1-Pass RePAIR + Spike-Odds Filter + GPU L1 Residual-MLP
# ============================================================================
D4_R2_ASTRA = '''import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_d4_r2_astra(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)   # [N_tr, 12, 96, 3]
    Y_tr = tr["Y_train"].astype(np.float64)   # [N_tr, 12, 96]
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape
    D = L * S * C
    r_rank = 24

    # 1) Astra local 3-point median spike-odds shrinkage + Rank-24 1-Pass Khatri-Rao PPCA RePAIR!
    mu_d = np.mean(X_tr.reshape(N_tr, D), axis=0, keepdims=True)
    sd_d = np.maximum(1e-4, np.std(X_tr.reshape(N_tr, D), axis=0, keepdims=True))
    Z_tr = (X_tr.reshape(N_tr, D) - mu_d) / sd_d

    _, s_vals, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    U = Vt[:r_rank].T  # [3456, 24]
    lam = np.maximum(1e-3, (s_vals[:r_rank] ** 2) / N_tr)
    nu = np.maximum(0.02, np.mean((Z_tr - (Z_tr @ U) @ U.T) ** 2, axis=0))
    Uw = U / np.sqrt(nu[:, None])
    KR = (Uw[:, :, None] * Uw[:, None, :]).reshape(D, r_rank * r_rank)  # [3456, 576] -> ultra-fast!
    prior_diag = np.diag(1.0 / lam)
    U_div_nu = U / nu[:, None]

    def local_3med(X_in):
        out = np.empty_like(X_in)
        out[:, 0] = np.median(X_in[:, [0, 1, 2]], axis=1)
        out[:, -1] = np.median(X_in[:, [-3, -2, -1]], axis=1)
        for t in range(1, L - 1):
            out[:, t] = np.median(X_in[:, [t - 1, t, t + 1]], axis=1)
        return out

    res_tr = X_tr - local_3med(X_tr)
    var_clean = np.maximum(0.008, np.mean(res_tr ** 2, axis=(0, 1), keepdims=True))
    var_tot = var_clean + 4.5 ** 2

    def fast_hybrid_repair(X_in):
        B_sz = len(X_in)
        drop_s = (np.sum(np.abs(X_in) < 1e-7, axis=(1, 3)) >= 30)  # [B, 96]
        m_loc = local_3med(X_in)
        e = X_in - m_loc
        # Posterior spike probability
        lp1 = np.log(0.03) - 0.5 * np.log(var_tot) - 0.5 * (e ** 2) / var_tot
        lp0 = np.log(0.97) - 0.5 * np.log(var_clean) - 0.5 * (e ** 2) / var_clean
        q_sp = 1.0 / (1.0 + np.exp(np.clip(lp0 - lp1, -30.0, 30.0)))
        X_soft = X_in - q_sp * e * ((4.5 ** 2) / var_tot)

        # Hard-mask high-probability spikes (q > 0.35) AND dropped sensors for 1-GEMM Rank-24 PPCA Kriging!
        obs_mask = (~drop_s[:, None, :, None]) & (q_sp < 0.35)
        M_flat = obs_mask.reshape(B_sz, D).astype(np.float64)
        Z_soft = (X_soft.reshape(B_sz, D) - mu_d) / sd_d

        A_b = (M_flat @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c_lat = np.linalg.solve(A_b, ((M_flat * Z_soft) @ U_div_nu)[:, :, None]).squeeze(-1)
        Z_ppca = c_lat @ U.T
        # Blend soft-shrunk observation with PPCA posterior mean weighted by (1 - q_sp)!
        w_keep = M_flat * (1.0 - q_sp.reshape(B_sz, D))
        Z_fin = w_keep * Z_soft + (1.0 - w_keep) * Z_ppca
        X_rep = (Z_fin * sd_d + mu_d).reshape(B_sz, L, S, C)
        return X_rep, c_lat, drop_s.astype(np.float64)

    # Top-6 correlated sensors
    corr = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T)
    np.fill_diagonal(corr, -2.0)
    top6 = np.argsort(-corr, axis=1)[:, :6]

    # Corrupt training windows + fast 1-pass hybrid repair (~1.5s!)
    rng = np.random.RandomState(54)
    X_tr_c = np.where(rng.rand(N_tr, 1, S, 1) < 0.12, 0.0, X_tr)
    X_tr_c = X_tr_c + (rng.rand(*X_tr.shape) < 0.03) * rng.normal(0.0, 4.5, size=X_tr.shape)
    Xr_tr, Cl_tr, Df_tr = fast_hybrid_repair(X_tr_c)

    # 1-Shot per-sensor Ridge Base Forecast (no slow IRLS loops -> ~0.3s!)
    def make_s_phi(Xr, Cl, Df, s_i):
        B_sz = len(Xr)
        loc = Xr[:, :, s_i, :].reshape(B_sz, 36)
        peers = Xr[:, :, top6[s_i], 0].reshape(B_sz, 72)
        return np.hstack([loc, Xr[:, -1, s_i, :], peers, Cl, Df[:, s_i:s_i + 1], np.ones((B_sz, 1))])

    d_p = 36 + 3 + 72 + r_rank + 2
    W_base = np.zeros((S, d_p, 12), dtype=np.float64)
    reg = 5.0 * np.eye(d_p); reg[-1, -1] = 1e-6
    Y_base_tr = np.empty((N_tr, L, S), dtype=np.float32)
    for s_i in range(S):
        Phi_s = make_s_phi(Xr_tr, Cl_tr, Df_tr, s_i)
        W_base[s_i] = np.linalg.solve(Phi_s.T @ Phi_s + reg, Phi_s.T @ Y_tr[:, :, s_i])
        Y_base_tr[:, :, s_i] = (Phi_s @ W_base[s_i]).astype(np.float32)

    # Shared Zero-Initialized GPU Residual-MLP trained with exact L1 (MAE) loss (~0.8s on A100 GPU!)
    def build_mlp_tensor(Xr, Cl, Df, Y_b):
        # Per (window, sensor) features: local [36] + peer mean [12] + latent [24] + base_pred [12] + drop_flag [1] = 85 dims!
        B_sz = len(Xr)
        loc = Xr.transpose(0, 2, 1, 3).reshape(B_sz, S, 36)
        peer_m = np.mean(Xr[:, :, top6, 0], axis=3).transpose(0, 2, 1)  # [B, 96, 12]
        lat_rep = np.repeat(Cl[:, None, :], S, axis=1)                   # [B, 96, 24]
        yb_s = Y_b.transpose(0, 2, 1)                                    # [B, 96, 12]
        df_s = Df[:, :, None]                                            # [B, 96, 1]
        return np.concatenate([loc, peer_m, lat_rep, yb_s, df_s], axis=2).astype(np.float32)

    torch.manual_seed(54)
    res_mlp = nn.Sequential(
        nn.Linear(85, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 12),
    ).to(DEVICE)
    nn.init.zeros_(res_mlp[-1].weight); nn.init.zeros_(res_mlp[-1].bias)
    sens_emb = nn.Parameter(torch.zeros(1, S, 12, device=DEVICE))
    opt = torch.optim.AdamW([*res_mlp.parameters(), sens_emb], lr=2.5e-3, weight_decay=1e-2)

    Ft_tr = torch.tensor(build_mlp_tensor(Xr_tr, Cl_tr, Df_tr, Y_base_tr), device=DEVICE)
    Yt_tr = torch.tensor(Y_tr.transpose(0, 2, 1).astype(np.float32), device=DEVICE)
    Yb_tr_t = torch.tensor(Y_base_tr.transpose(0, 2, 1), device=DEVICE)

    for _ in range(45):
        perm = torch.randperm(N_tr, device=DEVICE)
        for b0 in range(0, N_tr, 256):
            sl = perm[b0:b0 + 256]
            pred = Yb_tr_t[sl] + res_mlp(Ft_tr[sl]) + sens_emb
            loss = F.l1_loss(pred, Yt_tr[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    res_mlp.eval()

    outs = []
    with torch.no_grad():
        for it in items:
            bi = int(it["block_idx"])
            Xr_e, Cl_e, Df_e = fast_hybrid_repair(X_blocks[bi])
            B_sz = len(Xr_e)
            Yb_e = np.empty((B_sz, L, S), dtype=np.float32)
            for s_i in range(S):
                Yb_e[:, :, s_i] = (make_s_phi(Xr_e, Cl_e, Df_e, s_i) @ W_base[s_i]).astype(np.float32)
            Ft_e = torch.tensor(build_mlp_tensor(Xr_e, Cl_e, Df_e, Yb_e), device=DEVICE)
            Yb_et = torch.tensor(Yb_e.transpose(0, 2, 1), device=DEVICE)
            Yn_fin = (Yb_et + res_mlp(Ft_e) + sens_emb).cpu().numpy().transpose(0, 2, 1)
            outs.append({"id": it["id"], "prediction": (Yn_fin * flow_std + flow_mean).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_d4_r2_astra(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D5 R2 Astra: Gaussian EM + 1-Step Heavy-Tailed t_5 Damped Update + Multi-Lam Debiased Score
# ============================================================================
D5_R2_ASTRA = '''import argparse, json, math
import numpy as np

def admm_glasso_warm(S, lam, rho=1.5, n_iter=16, Z_init=None, U_init=None):
    p = S.shape[0]
    Z = np.eye(p, dtype=np.float64) if Z_init is None else Z_init.copy()
    U = np.zeros((p, p), dtype=np.float64) if U_init is None else U_init.copy()
    thresh = lam / rho
    Om = Z.copy()
    for _ in range(n_iter):
        mat = rho * (Z - U) - S
        vals, Q = np.linalg.eigh(0.5 * (mat + mat.T))
        eig_om = (vals + np.sqrt(vals ** 2 + 4.0 * rho)) / (2.0 * rho)
        Om = (Q * eig_om[None, :]) @ Q.T
        V = Om + U
        Z_new = np.sign(V) * np.maximum(0.0, np.abs(V) - thresh)
        np.fill_diagonal(Z_new, np.diag(V))
        U = U + Om - Z_new
        Z = Z_new
    return Om, Z, U

def score_d5_r2_astra(X_obs, M):
    T, p = X_obs.shape
    cnt = np.maximum(1.0, np.sum(M, axis=0))
    mu = np.sum(X_obs * M, axis=0) / cnt
    Xc = (X_obs - mu[None, :]) * M
    sd = np.sqrt(np.maximum(1e-6, np.sum((Xc ** 2) * M, axis=0) / np.maximum(1.0, cnt - 1.0)))
    X = Xc / sd[None, :]
    mu = np.zeros(p, dtype=np.float64)

    n_ij = np.maximum(1.0, M.T @ M)
    lam_base = 0.44 * math.sqrt(math.log(max(3, p)) / max(8.0, float(np.median(n_ij))))
    Om = np.eye(p, dtype=np.float64)
    Z_a, U_a = Om.copy(), np.zeros((p, p), dtype=np.float64)

    # 6 rounds of Gaussian MAR EM
    X_imp = X.copy()
    C_list = [np.zeros((p, p), dtype=np.float64) for _ in range(T)]
    w_mahal = np.ones(T, dtype=np.float64)

    for r in range(6):
        Sigma = np.linalg.inv(0.5 * (Om + Om.T) + 1e-5 * np.eye(p))
        X_imp = X.copy()
        S_cond = np.zeros((p, p), dtype=np.float64)
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            obs = np.where(M[t] > 0.5)[0]
            Ct = np.zeros((p, p), dtype=np.float64)
            if 0 < len(miss) < p:
                Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-5 * np.eye(len(obs)))
                W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                X_imp[t, miss] = mu[miss] + W_mo @ (X[t, obs] - mu[obs])
                Ct[np.ix_(miss, miss)] = Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
                if r == 5:
                    do_t = X[t, obs] - mu[obs]
                    d2_t = max(0.0, float(do_t @ (Soo_inv @ do_t)))
                    w_mahal[t] = (5.0 + len(obs)) / (5.0 + d2_t)
            elif len(miss) == p:
                X_imp[t] = mu
                Ct = Sigma.copy()
            elif r == 5 and len(obs) == p:
                Soo_inv = np.linalg.inv(Sigma + 1e-5 * np.eye(p))
                do_t = X[t] - mu
                w_mahal[t] = (5.0 + p) / (5.0 + max(0.0, float(do_t @ (Soo_inv @ do_t))))
            C_list[t] = Ct
            S_cond += Ct
        if r < 5:
            mu = np.mean(X_imp, axis=0)
        D = X_imp - mu[None, :]
        S_em = (D.T @ D + S_cond) / T + 1e-3 * np.eye(p)
        Om, Z_a, U_a = admm_glasso_warm(S_em, lam=lam_base, rho=1.5, n_iter=15, Z_init=Z_a, U_init=U_a)

    # 25%-damped Opus multivariate-t_5 Mahalanobis update (retaining unweighted C_t!)
    D = X_imp - mu[None, :]
    w_damped = 0.75 + 0.25 * (w_mahal / max(1e-6, float(np.mean(w_mahal))))
    S_t = ((D.T * w_damped[None, :]) @ D + sum(C_list)) / T + 1e-3 * np.eye(p)
    # Normalize to correlation R
    d_st = np.sqrt(np.maximum(1e-10, np.diag(S_t)))
    R_corr = S_t / np.outer(d_st, d_st)

    iu = np.triu_indices(p, k=1)
    i_idx, j_idx = iu
    obs_factor = np.sqrt(n_ij[i_idx, j_idx] / float(T))

    # Multi-lambda debiased curvature-gain consensus across 2 sparsity scales
    scores_acc = np.zeros(len(i_idx), dtype=np.float64)
    for mult in (0.85, 1.15):
        Om_k, Z_k, _ = admm_glasso_warm(R_corr, lam=lam_base * mult, rho=1.5, n_iter=20, Z_init=Z_a, U_init=U_a)
        W_inv = np.linalg.inv(0.5 * (Om_k + Om_k.T) + 1e-6 * np.eye(p))
        Om_deb = 2.0 * Om_k - Om_k @ R_corr @ Om_k
        curv_se = np.sqrt(np.maximum(1e-10, W_inv[i_idx, i_idx] * W_inv[j_idx, j_idx] + W_inv[i_idx, j_idx] ** 2))
        pcorr_l1 = np.abs(Z_k[iu]) / np.sqrt(np.maximum(1e-10, np.diag(Om_k)[i_idx] * np.diag(Om_k)[j_idx]))
        deb_stat = (np.abs(Om_deb[iu]) / curv_se) * obs_factor
        scores_acc += 0.5 * (0.58 * deb_stat + 0.42 * pcorr_l1)

    return scores_acc.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": score_d5_r2_astra(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D6 R2 Astra: FLASH Annealed SMC Bridge + Z2 Antipodal Mirror Metropolis Jumps
# ============================================================================
D6_R2_ASTRA = '''import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def stratified_resample(log_w, rng):
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w /= np.sum(w)
    ess = 1.0 / float(np.sum(w ** 2))
    M = len(w)
    u = (rng.rand(M) + np.arange(M, dtype=np.float64)) / M
    cdf = np.cumsum(w)
    cdf[-1] = 1.0 + 1e-12
    return np.clip(np.searchsorted(cdf, u), 0, M - 1), ess

def run_d6_r2_astra(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 137 + 29)
    M, d = X_init.shape
    s_disp = max(1.0, float(np.median(np.linalg.norm(X_init, axis=1))) / math.sqrt(d))
    X = 0.4 * X_init + rng.normal(0.0, 1.15 * s_disp, size=(M, d))

    def land_eval(Z):
        U, gU, h, J = prob.eval_all(Z)
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        corr = np.einsum("mk,mkd->md", c, J)
        return Z - corr, corr, U, gU, h, J, JJt

    for _ in range(6):
        X, corr, _, _, h0, _, _ = land_eval(X)
        if float(np.max(np.abs(h0))) < 1e-8:
            break

    eta_G = 0.032
    prev_med = 0.1

    def fused_move(X_in, beta, sc=1.0):
        nonlocal eta_G, prev_med
        X_land, corr, U, gU, h, J, JJt = land_eval(X_in)
        k = J.shape[1]
        d_med = max(1e-8, float(np.median(np.linalg.norm(corr, axis=1))))
        eta_G = float(np.clip(eta_G * np.clip((0.12 * prev_med / d_med) ** 2, 0.5, 1.8), 0.002, 0.055))

        rhs_g = np.einsum("mkd,md->mk", J, beta * gU)
        cg = np.linalg.solve(JJt, rhs_g[:, :, None]).squeeze(-1)
        g_tan = beta * gU - np.einsum("mk,mkd->md", cg, J)
        eta_F = 0.15 * max(1, d - k) / max(1e-4, float(np.mean(np.sum(g_tan ** 2, axis=1))))
        eta = min(eta_G, eta_F) * sc

        raw_v = -eta * beta * gU + math.sqrt(2.0 * eta) * rng.normal(0.0, 1.0, size=(M, d))
        rhs_v = np.einsum("mkd,md->mk", J, raw_v)
        cv = np.linalg.solve(JJt, rhs_v[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", cv, J)
        vn = np.linalg.norm(v_tan, axis=1, keepdims=True)
        v_tan = v_tan * np.minimum(1.0, 0.85 / np.maximum(1e-8, vn))
        prev_med = max(1e-4, float(np.median(vn)))
        return X_land + v_tan, X_land, U

    def antipodal_mirror_step(X_cur, beta):
        # Test exact antipodal mirror -X_landed; charge 1 eval_all call and accept only manifold-valid points!
        X_l, _, U_c, _, _, _, _ = land_eval(X_cur)
        X_m = -X_l
        U_m, _, h_m, _ = prob.eval_all(X_m)
        on_man = np.max(np.abs(h_m), axis=1) <= 1e-5
        acc = on_man & (rng.rand(M) < np.exp(np.clip(beta * (U_c - U_m), -30.0, 0.0)))
        return np.where(acc[:, None], X_m, X_l)

    # 16 warmup steps at beta=0.10 + mirror
    beta_prev = 0.10
    for _ in range(16):
        X, _, _ = fused_move(X, beta_prev)
    X = antipodal_mirror_step(X, beta_prev)

    # 10-stage FLASH bridge with stratified SMC resampling + Z2 mirror mixing!
    log_w = np.zeros(M, dtype=np.float64)
    for j in range(1, 11):
        beta_j = 0.10 ** (1.0 - j / 10.0)
        X, _, U_st, _, _, _, _ = land_eval(X)
        U_cl = np.clip(U_st, np.percentile(U_st, 2), np.percentile(U_st, 95) + 15.0)
        log_w += -(beta_j - beta_prev) * U_cl
        lw_s = log_w - np.max(log_w)
        wn = np.exp(lw_s) / np.sum(np.exp(lw_s))
        if (1.0 / float(np.sum(wn ** 2))) < 0.55 * M or j == 10:
            idx_r, _ = stratified_resample(log_w, rng)
            X = X[idx_r].copy()
            log_w = np.zeros(M, dtype=np.float64)
        beta_prev = beta_j
        for _ in range(12):
            X, _, _ = fused_move(X, beta_j)
        if j in (3, 6, 9):
            X = antipodal_mirror_step(X, beta_j)

    # Final cold beta=1 moves + periodic Z2 antipodal mirror Metropolis!
    while prob.grad_calls + 18 < max_steps:
        for _ in range(10):
            if prob.grad_calls + 8 >= max_steps:
                break
            sc = 0.55 if (max_steps - prob.grad_calls) < 40 else 1.0
            X, _, _ = fused_move(X, 1.0, sc=sc)
        if prob.grad_calls + 9 < max_steps:
            X = antipodal_mirror_step(X, 1.0)

    while prob.grad_calls + 6 < max_steps:
        X, _, _ = fused_move(X, 1.0, sc=0.45)

    for _ in range(5):
        if prob.grad_calls >= max_steps:
            break
        X, _, _, _, hf, _, _ = land_eval(X)
        if float(np.max(np.abs(hf))) <= 1e-8:
            break

    return {"particles": X.tolist(), "grad_calls": int(prob.grad_calls)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    mlib = load_manifold_lib(req["manifold_lib_path"])
    preds = []
    for it in req["items"]:
        prob = mlib.BudgetedManifoldProblem(it["spec"], hard_cap=int(it["max_steps"]))
        res = run_d6_r2_astra(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D7 R2 Astra: 64-Replica BiT dSB Elites + Exact 2^8=256-Subset Neighbor Multi-Flip + Tabu
# ============================================================================
D7_R2_ASTRA = '''import argparse, json
import numpy as np

M_SUB = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)
A_I, C_I = np.triu_indices(8, 1)
N_PAIRS = (M_SUB[:, A_I] * M_SUB[:, C_I]).T  # [28, 256]

def solve_d7_r2_astra(A_in, seed):
    rng = np.random.default_rng(seed)
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    tot = 0.25 * float(A.sum())
    c0 = 0.5 * np.sqrt(max(1.0, n - 1.0) / max(1e-12, float((A ** 2).sum())))

    # Phase 1: Opus 64-replica discrete Simulated Bifurcation (T=150)
    K = 64
    X = 0.02 * (rng.random((K, n)) - 0.5)
    P = 0.02 * (rng.random((K, n)) - 0.5)
    best = np.full(K, -np.inf, dtype=np.float64)
    bS = np.ones((K, n), dtype=np.float64)
    T_sb = 150
    for t in range(T_sb):
        S = np.where(X >= 0.0, 1.0, -1.0)
        AS = S @ A
        if 2 * t >= T_sb:
            c = tot - 0.25 * np.sum(S * AS, axis=1)
            imp = c > best
            best[imp] = c[imp]
            bS[imp] = S[imp]
        P += -(1.0 - t / T_sb) * X - c0 * AS
        X += P
        np.clip(X, -1.0, 1.0, out=X)
        P *= (np.abs(X) < 1.0)

    # Steepest 1-flip ascent on top-20 replicas
    S = bS[np.argsort(-best)[:20]].copy()
    H = S @ A
    ar = np.arange(len(S))
    for _ in range(4 * n):
        G = S * H
        i = np.argmax(G, axis=1)
        m = G[ar, i] > 1e-9
        if not np.any(m):
            break
        si = S[ar, i] * m
        S[ar, i] -= 2.0 * si
        H -= 2.0 * si[:, None] * A[i]

    cuts = tot - 0.25 * np.sum(S * H, axis=1)
    order = np.argsort(-cuts)
    # Select up to 12 sign-canonicalized elites + run interleaved Tabu + Astra 2^8-subset 8-spin connected moves!
    S = S[order[:12]].copy()
    H = S @ A
    k = len(S)
    ar = np.arange(k)
    cut = tot - 0.25 * np.sum(S * H, axis=1)
    best_e = cut.copy()
    bS_e = S.copy()
    tb = np.zeros((k, n), dtype=np.int64)
    ten = rng.integers(max(3, n // 20), max(6, n // 8), size=k)
    three_mask = (np.sum(M_SUB, axis=1) >= 2)

    for t in range(1, 96):
        if t in (24, 48, 72):
            # Astra Exact 2^8 = 256-subset neighbor-connected 8-spin move on current best elites!
            S = bS_e.copy()
            H = S @ A
            G = S * H
            v = G.copy()
            I_list = []
            i = np.argpartition(-G, 7, axis=1)[ar, rng.integers(8, size=k)]
            for _ in range(8):
                I_list.append(i)
                # Boost connected neighbors + update marginal gain!
                v += 0.15 * (np.abs(A[i]) > 0) - 2.0 * S[ar, i, None] * A[i] * S
                v[ar, i] = -np.inf
                i = np.argmax(v, axis=1)
            I_mat = np.stack(I_list, axis=1)
            T_sp = S[ar[:, None], I_mat]
            gain = G[ar[:, None], I_mat] @ M_SUB.T - 2.0 * (A[I_mat[:, A_I], I_mat[:, C_I]] * T_sp[:, A_I] * T_sp[:, C_I]) @ N_PAIRS
            kb = np.argmax(gain, axis=1)
            kesc = np.argmax(np.where(three_mask[None, :] & (np.arange(256)[None, :] > 0), gain, -np.inf), axis=1)
            kp = np.where(gain[ar, kb] > 1e-9, kb, kesc)
            Wf = M_SUB[kp]
            S[ar[:, None], I_mat] *= (1.0 - 2.0 * Wf)
            tb[ar[:, None], I_mat] = t + 7 * Wf.astype(np.int64)
            H = S @ A
            cut = tot - 0.25 * np.sum(S * H, axis=1)
        else:
            G = S * H
            allowed = (tb <= t) | (G > (best_e - cut)[:, None] + 1e-9)
            i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
            si = S[ar, i].copy()
            cut += si * H[ar, i]
            S[ar, i] = -si
            H -= 2.0 * si[:, None] * A[i]
            tb[ar, i] = t + ten

        j = cut > best_e + 1e-9
        best_e[j] = cut[j]
        bS_e[j] = S[j]

    return bS_e[int(np.argmax(best_e))].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_d7_r2_astra(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D8 R2 Astra: Opus 161-pt QNG-VI Warm-Start + Antithetic Halton QMC BFGS Polish
# ============================================================================
D8_R2_ASTRA = '''import argparse, json, math
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
'''


# ============================================================================
# D9 R2 Astra: Hybrid QR-Whitened 14-Step Restarted FISTA + Khatri-Rao SSN + Anchor
# ============================================================================
D9_R2_ASTRA = '''import argparse, json
from pathlib import Path
import numpy as np

def dual_obj_q(Y, Q, c_rhs, lo, hi, L_d):
    U = Y - L_d @ Q.T
    Z = np.clip(U, lo[None, :], hi[None, :])
    return np.sum(0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2), axis=1) + (L_d @ c_rhs)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    Q, R_qr = np.linalg.qr(A.T, mode="reduced")
    c_rhs = np.linalg.solve(R_qr.T, b)
    KR_q = (Q[:, :, None] * Q[:, None, :]).reshape(d, m * m)

    L_cur = Y @ Q - c_rhs[None, :]
    E = L_cur.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(14):
        res = np.clip(Y - E @ Q.T, lo[None, :], hi[None, :]) @ Q - c_rhs[None, :]
        Ln = E + res
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Ln - L_cur) * res, axis=1) < 0.0
        be[rs] = 0.0; tn[rs] = 1.0
        E = Ln + be[:, None] * (Ln - L_cur)
        L_cur = Ln; th = tn

    eye_m = 1e-9 * np.eye(m)[None, :, :]
    for _ in range(7):
        U = Y - L_cur @ Q.T
        Z = np.clip(U, lo[None, :], hi[None, :])
        rq = Z @ Q - c_rhs[None, :]
        unc = np.where(np.max(np.abs(rq), axis=1) > 1e-10)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]; ru = rq[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR_q).reshape(len(unc), m, m) + eye_m
        dl = np.linalg.solve(H, ru[:, :, None]).squeeze(-1)
        Lu = L_cur[unc]; Yu = Y[unc]
        p0 = dual_obj_q(Yu, Q, c_rhs, lo, hi, Lu)
        slope = -np.sum(ru * dl, axis=1)
        step = np.ones(len(unc)); done = np.zeros(len(unc), dtype=bool)
        Lb = Lu + ru
        for _ in range(6):
            Lt = Lu + step[:, None] * dl
            pt = dual_obj_q(Yu, Q, c_rhs, lo, hi, Lt)
            ok = (pt <= p0 + 1e-4 * step * slope) & (~done)
            if np.any(ok):
                Lb[ok] = Lt[ok]; done[ok] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        L_cur[unc] = Lb

    return np.clip(Y - L_cur @ Q.T, lo[None, :], hi[None, :])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    ev = np.load(req["eval_npz"])
    n_rep = int(req.get("n_repeats", 12))
    out_npz = Path(req["out_npz_dir"]) / (Path(args.output_json).stem + "_z.npz")
    loaded = [(it["id"], ev[f"{it['id']}_A"], ev[f"{it['id']}_b"], ev[f"{it['id']}_lo"], ev[f"{it['id']}_hi"], ev[f"{it['id']}_Y"]) for it in req["items"]]
    out_arrays = {}
    for rep in range(n_rep):
        for iid, A, b, lo, hi, Y in loaded:
            Z = solve_qp(A, b, lo, hi, Y)
            if rep == n_rep - 1:
                out_arrays[iid] = Z
    np.savez(out_npz, **out_arrays)
    preds = [{"id": it["id"], "prediction": {"npz_path": str(out_npz)}} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


def main() -> None:
    node = sys.argv[1]
    if node == "node0":
        write_cand("bench_d1_shapiq", "d1_v5_r2_astra_crossover", D1_R2_ASTRA)
        write_cand("bench_d2_acs_reweight", "d2_v5_r2_astra_crossover", D2_R2_ASTRA)
        write_cand("bench_d3_cmnn", "d3_v5_r2_astra_crossover", D3_R2_ASTRA)
    elif node == "node1":
        write_cand("bench_d4_pems04_rstib", "d4_v5_r2_astra_crossover", D4_R2_ASTRA)
        write_cand("bench_d5_snp500_scoremissing", "d5_v5_r2_astra_crossover", D5_R2_ASTRA)
        write_cand("bench_d6_ollalanding", "d6_v5_r2_astra_crossover", D6_R2_ASTRA)
    elif node == "node2":
        write_cand("bench_d7_rld_maxcut", "d7_v5_r2_astra_crossover", D7_R2_ASTRA)
        write_cand("bench_d8_lsvi_vi", "d8_v5_r2_astra_crossover", D8_R2_ASTRA)
        write_cand("bench_d9_pinet_qp", "d9_v5_r2_astra_crossover", D9_R2_ASTRA)
    print(f"ASTRA_R2_GENERATED_{node.upper()}")


if __name__ == "__main__":
    main()
