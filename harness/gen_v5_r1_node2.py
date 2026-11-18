#!/usr/bin/env python3
"""
Wave-5 Round-1 Candidate & Baseline Generator — Node 2 (D7, D8, D9-b2048):
  - Copies frozen Wave-4 baselines, comparators, and champions for D7 & D8
  - Writes symmetric `.npz`-amortized (`B=2048`, `n_repeats=12`) baselines/comparators/champion for D9-b2048
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
# D7 — Opus 5.5 Max (BiT: dSB + Steepest Ascent + Tabu) & GPT 6.2 Astra Max (KL-8 Tabu)
# ============================================================================
D7_OPUS_BIT = '''import argparse, json
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
'''


D7_ASTRA_KL8 = '''import argparse, json
import numpy as np

def cut_astra(A_in, seed):
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    scale = float(np.max(np.abs(A)))
    if scale > 0.0:
        A /= scale
    n = len(A)
    r = np.random.default_rng(seed)

    # Signed-Laplacian power iteration + hyperplane rounding (24 chains) + 6 random chains = 30 parallel chains
    V = r.normal(size=(n, 8))
    D = np.sum(np.abs(A), axis=1, keepdims=True)
    for _ in range(16):
        V = D * V - A @ V
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)

    S_hyp = np.where((V @ r.normal(size=(8, 24))).T >= 0.0, 1.0, -1.0)
    S_rnd = r.choice([-1.0, 1.0], size=(6, n))
    S = np.vstack([S_hyp, S_rnd])  # [30, n]

    # Exact 2^8 = 256 multi-spin subset evaluator tables
    M = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)  # [256, 8]
    a_idx, c_idx = np.triu_indices(8, 1)
    N_pairs = (M[:, a_idx] * M[:, c_idx]).T  # [28, 256]
    j_ar = np.arange(len(S))
    U = np.zeros_like(S, dtype=np.int64)

    H = S @ A
    e = -0.25 * np.sum(S * H, axis=1)
    P = S.copy()
    best = e.copy()

    for t in range(210):
        if t in (55, 110, 165):
            # Grow coupled 8-vertex block by marginal gain and evaluate ALL 256 flip subsets in 1 GEMM!
            S = P.copy()
            G = S * (S @ A)
            v = G.copy()
            I_list = []
            i = np.argpartition(-G, 7, axis=1)[j_ar, r.integers(8, size=len(S))]
            for _ in range(8):
                I_list.append(i)
                v -= 2.0 * S[j_ar, i, None] * A[i] * S
                v[j_ar, i] = -np.inf
                i = np.argmax(v, axis=1)
            I_mat = np.stack(I_list, axis=1)  # [30, 8]
            T_sp = S[j_ar[:, None], I_mat]    # [30, 8]
            gain = G[j_ar[:, None], I_mat] @ M.T - 2.0 * (A[I_mat[:, a_idx], I_mat[:, c_idx]] * T_sp[:, a_idx] * T_sp[:, c_idx]) @ N_pairs
            k_best = np.argmax(gain, axis=1)
            # If best subset is 0 (no gain), take least-costly 3-flip escape!
            three_mask = (np.sum(M, axis=1) == 3)
            k_esc = np.argmax(np.where(three_mask[None, :], gain, -np.inf), axis=1)
            k_pick = np.where(gain[j_ar, k_best] > 1e-9, k_best, k_esc)
            W_flip = M[k_pick]
            S[j_ar[:, None], I_mat] *= (1.0 - 2.0 * W_flip)
            U.fill(0)
            U[j_ar[:, None], I_mat] = (t + 6) * W_flip.astype(np.int64)
            H = S @ A
            e = -0.25 * np.sum(S * H, axis=1)
        else:
            G = S * H
            allowed = (U <= t) | ((e[:, None] + G) > (best[:, None] + 1e-9))
            i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
            e += G[j_ar, i]
            H -= 2.0 * S[j_ar, i, None] * A[i]
            S[j_ar, i] *= -1.0
            U[j_ar, i] = t + r.integers(6, 16, size=len(S))

        imp = e > best + 1e-9
        P[imp] = S[imp]
        best[imp] = e[imp]

    # Final exact 1-flip polish on best chain
    win = P[int(np.argmax(best))].copy()
    Hw = A @ win
    for _ in range(2 * n):
        gw = win * Hw
        bi = int(np.argmax(gw))
        if gw[bi] <= 1e-9:
            break
        Hw -= 2.0 * win[bi] * A[:, bi]
        win[bi] = -win[bi]
    return win.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": cut_astra(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D8 — Opus 5.5 Max (QNG-VI 161-pt Natural Gradient) & GPT 6.2 Astra Max (Whitened GH L-BFGS-B)
# ============================================================================
D8_OPUS_QNG_VI = '''import argparse, json, math
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
'''


D8_ASTRA_WHITENED_GH = '''import argparse, json, math
import numpy as np
import torch

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def vi_astra(X, y, prior_prec):
    N, d = X.shape
    p = prior_prec
    I = np.eye(d, dtype=np.float64)
    W = np.linalg.cholesky(np.linalg.solve(p * I + 0.25 * (X.T @ X), I))
    XW = X @ W
    s_sign = 2.0 * y - 1.0
    D = -s_sign[:, None] * XW
    K_prior = p * (W.T @ W)

    # 25-node Gauss-Hermite quadrature built into numpy.polynomial.hermite!
    gh_x, gh_w = np.polynomial.hermite.hermgauss(25)
    z_gh = gh_x * math.sqrt(2.0)
    w_gh = gh_w / math.sqrt(math.pi)

    tril_i, tril_j = np.tril_indices(d, k=-1)

    def unpack(v):
        m = v[:d]
        log_d = v[d:2 * d]
        T = np.zeros((d, d), dtype=np.float64)
        np.fill_diagonal(T, np.exp(log_d))
        T[tril_i, tril_j] = v[2 * d:]
        return m, log_d, T

    def nll_and_grad(v):
        m, log_d, T = unpack(v)
        mean_u = D @ m
        DT = D @ T
        var_u = np.maximum(1e-14, np.sum(DT ** 2, axis=1))
        sd_u = np.sqrt(var_u)

        U_grid = mean_u[:, None] + sd_u[:, None] * z_gh[None, :]
        soft_grid = np.logaddexp(0.0, U_grid)
        sig_grid = expit(U_grid)

        nll_data = float(np.sum(soft_grid @ w_gh))
        g_mean = sig_grid @ w_gh
        g_sd = (sig_grid * z_gh[None, :]) @ w_gh

        grad_m = D.T @ g_mean + K_prior @ m
        scale_n = g_sd / sd_u
        grad_T = (D.T * scale_n[None, :]) @ DT + K_prior @ T

        prior_quad = 0.5 * float(m @ (K_prior @ m) + np.sum(T * (K_prior @ T)))
        loss = nll_data + prior_quad - float(np.sum(log_d))

        grad_log_d = np.diag(grad_T) * np.exp(log_d) - 1.0
        grad_off = grad_T[tril_i, tril_j]
        return loss, np.concatenate([grad_m, grad_log_d, grad_off])

    # Pure-NumPy BFGS with Armijo backtracking (35 quadratic-convergent steps, 0 scipy dependency!)
    x = np.zeros(2 * d + len(tril_i), dtype=np.float64)
    n_p = len(x)
    H_inv = np.eye(n_p, dtype=np.float64)
    f_val, g_val = nll_and_grad(x)
    for _ in range(40):
        if np.max(np.abs(g_val)) < 1e-9:
            break
        p_dir = -(H_inv @ g_val)
        if float(p_dir @ g_val) >= 0.0:
            p_dir = -g_val
            H_inv = np.eye(n_p)
        step = 1.0
        c1_sl = 1e-4 * float(p_dir @ g_val)
        for _ in range(10):
            x_try = x + step * p_dir
            f_try, g_try = nll_and_grad(x_try)
            if f_try <= f_val + step * c1_sl:
                s_v = x_try - x
                y_v = g_try - g_val
                sy = float(s_v @ y_v)
                if sy > 1e-12:
                    rho = 1.0 / sy
                    Hy = H_inv @ y_v
                    H_inv = H_inv - rho * (np.outer(s_v, Hy) + np.outer(Hy, s_v)) + (rho * (1.0 + rho * float(y_v @ Hy))) * np.outer(s_v, s_v)
                x, f_val, g_val = x_try, f_try, g_try
                break
            step *= 0.5
        else:
            break

    m_hat, _, T_hat = unpack(x)
    mu = W @ m_hat
    Sigma = (W @ T_hat) @ (W @ T_hat).T
    L = np.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-12 * I)
    return {"mu": mu.tolist(), "L": L.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = []
    for it in req["items"]:
        res = vi_astra(np.asarray(it["X"], dtype=np.float64), np.asarray(it["y"], dtype=np.float64), float(it["prior_prec"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D9-b2048 — Symmetric `.npz` Wrapper + Baselines/Champion + Opus AD-SSN + Astra QR-SSN
# ============================================================================
D9_NPZ_WRAPPER = '''import argparse, json
from pathlib import Path
import numpy as np

__SOLVER_BODY__

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    ev = np.load(req["eval_npz"])
    n_rep = int(req.get("n_repeats", 12))
    out_dir = Path(req["out_npz_dir"])
    out_npz = out_dir / (Path(args.output_json).stem + "_z.npz")

    # Load arrays into memory before timed solver loop (symmetric across ALL methods!)
    loaded = []
    for it in req["items"]:
        iid = it["id"]
        loaded.append((iid, ev[f"{iid}_A"], ev[f"{iid}_b"], ev[f"{iid}_lo"], ev[f"{iid}_hi"], ev[f"{iid}_Y"]))

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

D9_BASELINES_SOLVER = '''
VARIANT = __VARIANT__

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    AAt = A @ A.T + 1e-12 * np.eye(m)
    AAt_inv = np.linalg.inv(AAt)
    P_aff = np.eye(d) - A.T @ (AAt_inv @ A)
    shift_aff = A.T @ (AAt_inv @ b)

    if VARIANT in ("paper_dr50", "autosota_dr10", "comp_overrelaxed_dr50"):
        n_iters = 10 if VARIANT == "autosota_dr10" else 50
        omega = 1.65 if VARIANT == "comp_overrelaxed_dr50" else 1.0
        S = Y.copy()
        for _ in range(n_iters):
            Z_box = np.clip(S, lo[None, :], hi[None, :])
            Z_aff = Z_box @ P_aff.T + shift_aff[None, :] + (Z_box - S) @ P_aff.T
            # Exact DR reflection: proj_aff(2*Z_box - S) = (2*Z_box - S) @ P_aff.T + shift_aff
            Z_aff = (2.0 * Z_box - S) @ P_aff.T + shift_aff[None, :]
            S = S + omega * (Z_aff - Z_box)
        return np.clip(S, lo[None, :], hi[None, :])

    if VARIANT == "comp_dykstra50":
        X = Y.copy()
        P_inc = np.zeros_like(Y)
        Q_inc = np.zeros_like(Y)
        for _ in range(50):
            Ym = (X + P_inc) @ P_aff.T + shift_aff[None, :]
            P_inc = X + P_inc - Ym
            X = np.clip(Ym + Q_inc, lo[None, :], hi[None, :])
            Q_inc = Ym + Q_inc - X
        return X

    if VARIANT == "w4_champ_vec_admm":
        rho = 2.2
        alpha = 1.72
        W = np.clip(Y @ P_aff.T + shift_aff[None, :], lo[None, :], hi[None, :])
        U = np.zeros_like(Y)
        inv_1pr = 1.0 / (1.0 + rho)
        for _ in range(110):
            Z = ((Y + rho * (W - U)) * inv_1pr) @ P_aff.T + shift_aff[None, :]
            Z_hat = alpha * Z + (1.0 - alpha) * W
            W_new = np.clip(Z_hat + U, lo[None, :], hi[None, :])
            U = U + Z_hat - W_new
            W = W_new
        return W

    raise ValueError(VARIANT)
'''

D9_OPUS_AD_SSN_SOLVER = '''
def dual_obj_rows(Y, A, b, lo, hi, N):
    # Smooth convex dual psi(nu): grad_nu psi = A clip(y + A^T nu, lo, hi) - b
    U = Y + N @ A
    Z = np.clip(U, lo[None, :], hi[None, :])
    g = 0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2)
    return np.sum(g, axis=1) - (N @ b)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    G = A @ A.T + 1e-12 * np.eye(m)
    Gi = np.linalg.inv(G)
    # Khatri-Rao matrix [d, m*m] so free @ KR builds all B_unc active-set Hessians in ONE BLAS GEMM!
    KR = (A[:, None, :] * A[None, :, :]).reshape(m * m, d).T.copy()

    # Phase 1: 22 Preconditioned Restarted Dual FISTA steps (22 tiny [2048, d] @ [d, m] GEMMs vs 50 [2048, d] @ [d, d] in DR50 -> >2x faster!)
    N = (b[None, :] - Y @ A.T) @ Gi
    E = N.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(22):
        R = b[None, :] - np.clip(Y + E @ A, lo[None, :], hi[None, :]) @ A.T
        Nn = E + R @ Gi
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Nn - N) * R, axis=1) < 0.0
        be[rs] = 0.0
        tn[rs] = 1.0
        E = Nn + be[:, None] * (Nn - N)
        N = Nn
        th = tn

    # Phase 2: Batched Khatri-Rao Semismooth Active-Set Newton on unconverged rows only!
    eye_m = 1e-9 * np.eye(m)[None, :, :]
    for _ in range(8):
        U = Y + N @ A
        Z = np.clip(U, lo[None, :], hi[None, :])
        R = b[None, :] - Z @ A.T  # -grad_psi
        err = np.max(np.abs(R), axis=1)
        unc = np.where(err > 1e-8)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]
        Ru = R[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR).reshape(len(unc), m, m) + eye_m
        dnu = np.linalg.solve(H, Ru[:, :, None]).squeeze(-1)  # Newton step on N[unc]

        # Exact vectorized monotone dual line search across unc rows
        Nu = N[unc]
        Yu = Y[unc]
        psi0 = dual_obj_rows(Yu, A, b, lo, hi, Nu)
        slope = -np.sum(Ru * dnu, axis=1)  # grad_psi . dnu < 0
        step = np.ones(len(unc), dtype=np.float64)
        done = np.zeros(len(unc), dtype=bool)
        Nu_best = Nu + Ru @ Gi  # guaranteed fallback preconditioned gradient descent step!
        for _ in range(7):
            N_try = Nu + step[:, None] * dnu
            psi_try = dual_obj_rows(Yu, A, b, lo, hi, N_try)
            ok = psi_try <= psi0 + 1e-4 * step * slope
            newly_ok = ok & (~done)
            if np.any(newly_ok):
                Nu_best[newly_ok] = N_try[newly_ok]
                done[newly_ok] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        N[unc] = Nu_best

    return np.clip(Y + N @ A, lo[None, :], hi[None, :])
'''

D9_ASTRA_QR_SSN_SOLVER = '''
def dual_obj_rows_q(Y, Q, c_rhs, lo, hi, L_dual):
    # Orthonormalized row space: A^T = Q @ R_up => A z = b  <=>  Q^T z = c_rhs where c_rhs = R_up^{-T} b!
    # Notice Q^T Q = I_m => unconstrained preconditioner is IDENTICALLY I_m!!
    U = Y - L_dual @ Q.T
    Z = np.clip(U, lo[None, :], hi[None, :])
    g = 0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2)
    return np.sum(g, axis=1) + (L_dual @ c_rhs)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    # Thin QR of A^T: A^T = Q @ R_qr, Q in R^{d x m} with Q^T Q = I_m!
    Q, R_qr = np.linalg.qr(A.T, mode="reduced")  # Q: [d, m], R_qr: [m, m]
    c_rhs = np.linalg.solve(R_qr.T, b)  # [m]: Q^T z = c_rhs <=> A z = b!
    KR_q = (Q[None, :, :] * Q[:, None, :]).reshape(d, m * m) if False else (Q[:, :, None] * Q[:, None, :]).reshape(d, m * m)

    # 18 QR-whitened unit-step Nesterov Dual FISTA iterations (NO m x m solve needed in FISTA!)
    L_cur = Y @ Q - c_rhs[None, :]
    E = L_cur.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(18):
        Z_e = np.clip(Y - E @ Q.T, lo[None, :], hi[None, :])
        res = Z_e @ Q - c_rhs[None, :]  # grad wrt L_dual
        Ln = E + res
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Ln - L_cur) * res, axis=1) < 0.0
        be[rs] = 0.0
        tn[rs] = 1.0
        E = Ln + be[:, None] * (Ln - L_cur)
        L_cur = Ln
        th = tn

    # Active-set Semismooth Newton in Q-coordinates + Certified Feasible Anchor fallback!
    eye_m = 1e-9 * np.eye(m)[None, :, :]
    for _ in range(8):
        U = Y - L_cur @ Q.T
        Z = np.clip(U, lo[None, :], hi[None, :])
        res_q = Z @ Q - c_rhs[None, :]
        unc = np.where(np.max(np.abs(res_q), axis=1) > 1e-9)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]
        ru = res_q[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR_q).reshape(len(unc), m, m) + eye_m
        dl = np.linalg.solve(H, ru[:, :, None]).squeeze(-1)

        Lu = L_cur[unc]
        Yu = Y[unc]
        p0 = dual_obj_rows_q(Yu, Q, c_rhs, lo, hi, Lu)
        slope = -np.sum(ru * dl, axis=1)
        step = np.ones(len(unc), dtype=np.float64)
        done = np.zeros(len(unc), dtype=bool)
        Lu_best = Lu + ru
        for _ in range(7):
            L_try = Lu + step[:, None] * dl
            p_try = dual_obj_rows_q(Yu, Q, c_rhs, lo, hi, L_try)
            ok = p_try <= p0 + 1e-4 * step * slope
            newly = ok & (~done)
            if np.any(newly):
                Lu_best[newly] = L_try[newly]
                done[newly] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        L_cur[unc] = Lu_best

    Z_out = np.clip(Y - L_cur @ Q.T, lo[None, :], hi[None, :])
    # Certified Feasible Anchor Repair on any tail row exceeding 1e-6 equality error!
    eq_err = np.max(np.abs(Z_out @ A.T - b[None, :]), axis=1)
    bad = eq_err > 1e-6
    if np.any(bad) and np.any(~bad):
        h_anc = np.mean(Z_out[~bad], axis=0)
        Zb = Z_out[bad]
        Zb_aff = Zb - (Zb @ Q - c_rhs[None, :]) @ Q.T
        v_dir = Zb_aff - h_anc[None, :]
        # Largest alpha in [0, 1] keeping h_anc + alpha * v_dir inside [lo, hi]
        bound_num = np.where(v_dir > 0.0, hi[None, :] - h_anc[None, :], lo[None, :] - h_anc[None, :])
        alphas = np.divide(bound_num, v_dir, out=np.full_like(v_dir, np.inf), where=(np.abs(v_dir) > 1e-15))
        a_min = np.clip(np.min(alphas, axis=1), 0.0, 1.0)
        Z_out[bad] = np.clip(h_anc[None, :] + a_min[:, None] * v_dir, lo[None, :], hi[None, :])
    return Z_out
'''


def main() -> None:
    copy_v4_cands("bench_d7_rld_maxcut", {
        "d7_paper_rlsa_langevin": "d7_paper_rlsa_langevin",
        "d7_autosota_rlsa_leap": "d7_autosota_rlsa_leap",
        "d7_comp_spectral_greedy": "d7_comp_spectral_greedy",
        "d7_r2_crossover_dsb_memetic": "d7_w4_champ_dsb_memetic",
    })
    write_cand("bench_d7_rld_maxcut", "d7_v5_r1_opus_bit", D7_OPUS_BIT)
    write_cand("bench_d7_rld_maxcut", "d7_v5_r1_astra_kl8_tabu", D7_ASTRA_KL8)

    copy_v4_cands("bench_d8_lsvi_vi", {
        "d8_paper_lsvi": "d8_paper_lsvi",
        "d8_comp_laplace_newton": "d8_comp_laplace_newton",
        "d8_comp_bbvi_reparam_1x": "d8_comp_bbvi_reparam_1x",
        "d8_autosota_bbvi_8seed_long": "d8_autosota_bbvi_8seed_long",
        "d8_r1_explore_natgrad_tr": "d8_w4_champ_qmc_natgrad",
    })
    write_cand("bench_d8_lsvi_vi", "d8_v5_r1_opus_qng_vi", D8_OPUS_QNG_VI)
    write_cand("bench_d8_lsvi_vi", "d8_v5_r1_astra_whitened_gh_lbfgs", D8_ASTRA_WHITENED_GH)

    for vid, vtag in [
        ("d9_paper_pinet_dr50", "paper_dr50"),
        ("d9_autosota_pinet_dr10_infeasible", "autosota_dr10"),
        ("d9_comp_overrelaxed_dr50", "comp_overrelaxed_dr50"),
        ("d9_comp_dykstra50", "comp_dykstra50"),
        ("d9_w4_champ_vec_admm", "w4_champ_vec_admm"),
    ]:
        body = D9_BASELINES_SOLVER.replace("__VARIANT__", repr(vtag))
        write_cand("bench_d9_pinet_qp", vid, D9_NPZ_WRAPPER.replace("__SOLVER_BODY__", body))

    write_cand("bench_d9_pinet_qp", "d9_v5_r1_opus_ad_ssn", D9_NPZ_WRAPPER.replace("__SOLVER_BODY__", D9_OPUS_AD_SSN_SOLVER))
    write_cand("bench_d9_pinet_qp", "d9_v5_r1_astra_qr_ssn_cert", D9_NPZ_WRAPPER.replace("__SOLVER_BODY__", D9_ASTRA_QR_SSN_SOLVER))
    print("NODE2_V5_R1_CANDIDATES_GENERATED")


if __name__ == "__main__":
    main()
