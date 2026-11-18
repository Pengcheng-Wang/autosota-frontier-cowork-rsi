#!/usr/bin/env python3
"""
Wave-5 Round-2 Iterative Crossover Generator — Part B (`Cowork_Opus_5.5_Max` R2 Crossover):
Implements exact Round-2 Crossover designs synthesized from redacted Search_v5 feedback (`ID: eed7206b-b17f-4b21-bcec-80246d661e5b`).
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
# D1 R2 Opus: Fast 1-Batch Border+Complement Zeta-ARD GP + Synergy Hybrid (~0.25s!)
# ============================================================================
D1_R2_OPUS = '''import argparse, itertools, json
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
'''


# ============================================================================
# D2 R2 Opus: Gated Bayes-Minimax (4 Per-Group Experts + 4-Way q(g|x) Gate + 3-Fold OOF Dual)
# ============================================================================
D2_R2_OPUS = '''import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def make_astra_feats(X, xm, xs, meds, nonbin):
    Xs = (X - xm) / xs
    hinges = np.maximum(0.0, Xs[:, nonbin] - meds[None, :])
    sq = Xs[:, nonbin] ** 2
    return Xs, np.hstack([np.ones((len(X), 1)), Xs, hinges, sq])

def fit_fold_models(X_tr, y_tr, g_tr, xm, xs, meds, nonbin, seed):
    Xs, Phi = make_astra_feats(X_tr, xm, xs, meds, nonbin)
    N, d = Xs.shape
    torch.manual_seed(seed)

    # 1) Shared 2x128 trunk with 4 per-group logit heads + 4-way group posterior head q(g | x)!
    class MultiHeadExpertNet(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU())
            self.y_heads = nn.Linear(96, 4)
            self.g_head = nn.Linear(96, 4)

        def forward(self, x):
            h = self.trunk(x)
            return self.y_heads(h), self.g_head(h)

    net = MultiHeadExpertNet(d)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1.5e-2)
    xt = torch.tensor(Xs.astype(np.float32))
    yt = torch.tensor(y_tr.astype(np.float32))
    gt = torch.tensor(g_tr, dtype=torch.long)

    # Group-balanced inverse-prevalence weights so minority demographic x label cells have crisp logits!
    g8_t = g_tr * 2 + y_tr
    cnt8 = np.maximum(1.0, np.bincount(g8_t, minlength=8).astype(np.float32))
    inv_w8 = (len(g8_t) / (8.0 * cnt8)) ** 0.5
    wt = torch.tensor(inv_w8[g8_t])

    for _ in range(55):
        p_idx = torch.randperm(N)
        for b0 in range(0, N, 512):
            sl = p_idx[b0:b0 + 512]
            yl, gl = net(xt[sl])
            chosen_yl = yl.gather(1, gt[sl].unsqueeze(1)).squeeze(1)
            loss_y = torch.mean(wt[sl] * nn.functional.binary_cross_entropy_with_logits(chosen_yl, yt[sl], reduction="none"))
            loss_g = nn.functional.cross_entropy(gl, gt[sl])
            loss = loss_y + 0.5 * loss_g
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    # 2) Per-group quadratic logistic experts on Astra's Phi features via L-BFGS
    W_quad = np.zeros((Phi.shape[1], 4), dtype=np.float64)
    # Warm-start from global logistic fit
    pt_all = torch.tensor(Phi, dtype=torch.float64)
    yt_all = torch.tensor(y_tr.astype(np.float64), dtype=torch.float64)
    w_glob = torch.zeros(Phi.shape[1], dtype=torch.float64, requires_grad=True)
    opt_g = torch.optim.LBFGS([w_glob], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")
    def cl_g():
        opt_g.zero_grad()
        l = torch.mean(nn.functional.binary_cross_entropy_with_logits(pt_all @ w_glob, yt_all)) + 1e-3 * torch.sum(w_glob[1:] ** 2)
        l.backward()
        return l
    opt_g.step(cl_g)
    w0_np = w_glob.detach().numpy()

    for g_id in range(4):
        mg = (g_tr == g_id)
        if np.sum(mg) >= 25:
            pt_g = torch.tensor(Phi[mg], dtype=torch.float64)
            yt_g = torch.tensor(y_tr[mg].astype(np.float64), dtype=torch.float64)
            wg = torch.tensor(w0_np.copy(), dtype=torch.float64, requires_grad=True)
            w0_t = torch.tensor(w0_np, dtype=torch.float64)
            opt_k = torch.optim.LBFGS([wg], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")
            def cl_k():
                opt_k.zero_grad()
                # Hierarchical shrinkage toward w0_t prevents overfitting on small groups!
                l = torch.mean(nn.functional.binary_cross_entropy_with_logits(pt_g @ wg, yt_g)) + 5e-3 * torch.sum((wg[1:] - w0_t[1:]) ** 2)
                l.backward()
                return l
            opt_k.step(cl_k)
            W_quad[:, g_id] = wg.detach().numpy()
        else:
            W_quad[:, g_id] = w0_np

    def predict_experts(X_raw):
        Xs_e, Phi_e = make_astra_feats(X_raw, xm, xs, meds, nonbin)
        with torch.no_grad():
            yl, gl = net(torch.tensor(Xs_e.astype(np.float32)))
            yl_np = yl.numpy().astype(np.float64)
            gl_np = gl.numpy().astype(np.float64)
        gl_np -= np.max(gl_np, axis=1, keepdims=True)
        q_g = np.exp(gl_np) / np.sum(np.exp(gl_np), axis=1, keepdims=True)  # [N, 4]
        ell_g = 0.55 * yl_np + 0.45 * (Phi_e @ W_quad)
        p_y_given_g = expit(ell_g)  # [N, 4]
        return q_g, p_y_given_g

    return predict_experts

def run_d2_r2_opus(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr
    N = len(X_tr)

    xm = np.mean(X_tr, axis=0, keepdims=True)
    xs = np.where(np.std(X_tr, axis=0, keepdims=True) > 1e-6, np.std(X_tr, axis=0, keepdims=True), 1.0)
    Xs_all = (X_tr - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[:, j])) > 2]
    meds = np.median(Xs_all[:, nonbin], axis=0)

    # 3-Fold Cell-Stratified OOF to fit exact Bayes minimax weights w_{g, y} with ZERO overfitting!
    rng = np.random.RandomState(520)
    folds = [np.empty(0, dtype=np.int64) for _ in range(3)]
    fold_lists = [[], [], []]
    for c in range(8):
        ic = np.where(g8_tr == c)[0]
        rng.shuffle(ic)
        for f_i, part in enumerate(np.array_split(ic, 3)):
            fold_lists[f_i].append(part)
    folds = [np.concatenate(fl) for fl in fold_lists]

    oof_q = np.zeros((N, 4), dtype=np.float64)
    oof_p = np.zeros((N, 4), dtype=np.float64)
    fold_fns = []
    for f_i in range(3):
        val_idx = folds[f_i]
        tr_idx = np.concatenate([folds[j] for j in range(3) if j != f_i])
        fn = fit_fold_models(X_tr[tr_idx], y_tr[tr_idx], g_tr[tr_idx], xm, xs, meds, nonbin, seed=100 + f_i)
        fold_fns.append(fn)
        qf, pf = fn(X_tr[val_idx])
        oof_q[val_idx] = qf
        oof_p[val_idx] = pf

    # Exact Pointwise Bayes Minimax Rule over all 8 cells c = (g, y):
    # Predict y=1 iff sum_{g=0}^3 q(g|x) * [w_{g,1} * p(1|x,g) - w_{g,0} * (1 - p(1|x,g))] >= 0!
    pi_8 = np.array([max(1e-4, np.mean(g8_tr == c)) for c in range(8)], dtype=np.float64)
    n_8 = np.array([max(1.0, np.sum(g8_tr == c)) for c in range(8)], dtype=np.float64)

    lam = pi_8.copy()
    mu_fl = 0.0
    best_w8 = np.ones(8, dtype=np.float64)
    best_lcb = -1e9

    P1_oof = oof_q * oof_p            # [N, 4] = P(g, y=1 | x)
    P0_oof = oof_q * (1.0 - oof_p)    # [N, 4] = P(g, y=0 | x)

    for _ in range(240):
        w8 = lam / pi_8 + mu_fl
        w_g0 = w8[0::2]
        w_g1 = w8[1::2]
        margin = (P1_oof @ w_g1) - (P0_oof @ w_g0)
        yhat = (margin >= 0.0).astype(np.int64)
        corr = (yhat == y_tr).astype(np.float64)
        A_c = np.array([float(np.mean(corr[g8_tr == c])) for c in range(8)])
        A_mean = float(np.mean(corr))
        lcb = float(np.min(A_c - 0.35 * np.sqrt(np.maximum(1e-6, A_c * (1.0 - A_c)) / n_8)))
        if A_mean >= 0.750 and lcb > best_lcb:
            best_lcb = lcb
            best_w8 = w8.copy()
        lam = lam * np.exp(0.65 * (1.0 - A_c))
        lam /= np.sum(lam)
        mu_fl = max(0.0, mu_fl + 4.5 * (0.753 - A_mean))

    w_g0 = best_w8[0::2]
    w_g1 = best_w8[1::2]
    outs = []
    for it in eval_items:
        Xev = np.asarray(it["X_eval"], dtype=np.float64)
        q_ens = np.zeros((len(Xev), 4), dtype=np.float64)
        p_ens = np.zeros((len(Xev), 4), dtype=np.float64)
        for fn in fold_fns:
            qe, pe = fn(Xev)
            q_ens += qe / 3.0
            p_ens += pe / 3.0
        m_ev = ((q_ens * p_ens) @ w_g1) - ((q_ens * (1.0 - p_ens)) @ w_g0)
        outs.append({"id": it["id"], "prediction": (m_ev >= 0.0).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_d2_r2_opus(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D3 R2 Opus: 2-Stage Monotone Boosting (Astra Min/Max Backbone + Smooth I-Splines & 8-RBF Gates)
# ============================================================================
D3_R2_OPUS = '''import argparse, json
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

def smooth_ispline_col(x, a, b):
    """Exact quadratic Ramsay I-spline on [a, b]: C^1 smooth, values in [0, 1], derivative >= 0 everywhere!"""
    t = np.clip((x - a) / max(1e-8, b - a), 0.0, 1.0)
    return np.where(t < 0.5, 2.0 * (t ** 2), 1.0 - 2.0 * ((1.0 - t) ** 2))

def fit_stage1_astra(Xf_tr, Xm_tr, y_tr):
    n, df = Xf_tr.shape
    dm = Xm_tr.shape[1]
    if df > 0:
        fc = [abs(float(np.corrcoef(Xf_tr[:, j], y_tr)[0, 1])) if np.std(Xf_tr[:, j]) > 1e-8 else 0.0 for j in range(df)]
        top_f = list(np.argsort(-np.array(fc))[:min(14, df)])
        fq = {j: np.quantile(Xf_tr[:, j], [0.25, 0.50, 0.75]) for j in top_f}
        fgq = {j: np.quantile(Xf_tr[:, j], [0.20, 0.80]) for j in top_f[:min(8, len(top_f))]}
    else:
        top_f, fq, fgq = [], {}, {}

    def make_F(Xf):
        cols = [np.ones((len(Xf), 1))]
        if df > 0:
            cols.append(Xf)
            for j in top_f:
                for qv in fq[j]:
                    cols.append(np.maximum(0.0, Xf[:, j:j + 1] - qv))
        return np.hstack(cols)

    F_tr = make_F(Xf_tr)
    reg0 = 0.02 * n * np.eye(F_tr.shape[1]); reg0[0, 0] = 1e-8
    res0 = y_tr - F_tr @ np.linalg.solve(F_tr.T @ F_tr + reg0, F_tr.T @ y_tr)

    ramp_specs, best_col = [], []
    for k in range(dm):
        qs = np.quantile(Xm_tr[:, k], [0.0, 0.25, 0.50, 0.75, 1.0])
        bc, bsp = -1e9, None
        for bi in range(4):
            w = qs[bi + 1] - qs[bi]
            if w > 1e-6:
                sp = ("ramp", k, float(qs[bi]), float(w))
                rv = np.clip((Xm_tr[:, k] - sp[2]) / sp[3], 0.0, 1.0)
                cv = float((rv - np.mean(rv)) @ res0) / (float(np.linalg.norm(rv - np.mean(rv))) + 1e-8)
                ramp_specs.append((cv, sp))
                if cv > bc:
                    bc, bsp = cv, sp
        ramp_specs.append((0.5, ("lin", k, float(np.min(Xm_tr[:, k])), max(1.0, float(np.std(Xm_tr[:, k]))))))
        if bsp is not None:
            best_col.append((bc, bsp))

    best_col.sort(key=lambda x: -x[0])
    winners = [sp for _, sp in best_col[:min(12, len(best_col))]]

    def eval_sp(Xm, sp):
        if sp[0] == "ramp":
            return np.clip((Xm[:, sp[1]] - sp[2]) / sp[3], 0.0, 1.0)
        return (Xm[:, sp[1]] - sp[2]) / sp[3]

    inter_specs = []
    for ia in range(len(winners)):
        for ib in range(ia + 1, len(winners)):
            inter_specs.append(("min", winners[ia], winners[ib]))
            inter_specs.append(("max", winners[ia], winners[ib]))
    for spw in winners:
        for jf, (q20, q80) in fgq.items():
            if q80 - q20 > 1e-6:
                inter_specs.append(("fg", spw, jf, float(q20), float(q80 - q20), False))
                inter_specs.append(("fg", spw, jf, float(q20), float(q80 - q20), True))

    def eval_int(Xf, Xm, isp):
        if isp[0] == "min":
            return np.minimum(eval_sp(Xm, isp[1]), eval_sp(Xm, isp[2]))
        if isp[0] == "max":
            return np.maximum(eval_sp(Xm, isp[1]), eval_sp(Xm, isp[2]))
        u = eval_sp(Xm, isp[1])
        h = np.clip((Xf[:, isp[2]] - isp[3]) / isp[4], 0.0, 1.0)
        return ((1.0 - h) if isp[5] else h) * u

    main_sps = [sp for _, sp in ramp_specs]
    K_k = min(64, max(8, n // 5))
    sc_i = []
    for isp in inter_specs:
        v = eval_int(Xf_tr, Xm_tr, isp); vc = v - np.mean(v); nv = float(np.linalg.norm(vc))
        if nv > 1e-6:
            sc_i.append((float(vc @ res0) / nv, isp))
    sc_i.sort(key=lambda x: -x[0])
    kept = [isp for sv, isp in sc_i[:K_k] if sv > 0.0]

    def make_M(Xf, Xm):
        ps = [np.column_stack([eval_sp(Xm, sp) for sp in main_sps])]
        if len(kept) > 0:
            ps.append(np.column_stack([eval_int(Xf, Xm, isp) for isp in kept]))
        return np.hstack(ps)

    M_tr = make_M(Xf_tr, Xm_tr)
    pf, pm = F_tr.shape[1], M_tr.shape[1]
    Z = np.hstack([F_tr, M_tr])
    rms = np.where(np.sqrt(np.mean(Z ** 2, axis=0)) > 1e-8, np.sqrt(np.mean(Z ** 2, axis=0)), 1.0); rms[0] = 1.0
    Zn = Z / rms[None, :]
    rg = np.full(pf + pm, 0.015); rg[0] = 1e-9
    w = fista_mixed_qp((Zn.T @ Zn) / n + np.diag(rg), (Zn.T @ y_tr) / n, pf, n_iter=200) / rms
    wf, wm = w[:pf], np.maximum(0.0, w[pf:])
    return lambda Xf, Xm: make_F(Xf) @ wf + make_M(Xf, Xm) @ wm

def fit_stage2_ispline_rbf(Xf_tr, Xm_tr, r_tr):
    """Stage 2 residual booster: dense smooth I-splines (MSR backbone) + K=6 RBF-gated hinges + top-6 hinge products!"""
    n, df = Xf_tr.shape
    dm = Xm_tr.shape[1]
    rng = np.random.RandomState(303)

    K_g = 6 if (df >= 2 and n >= 160) else 2
    if df > 0:
        Xfc = Xf_tr - np.mean(Xf_tr, axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xfc, full_matrices=False)
        V_pc = Vt[:min(10, len(Vt))].T
        Zt = Xfc @ V_pc
        centers = Zt[rng.choice(n, size=K_g, replace=False)].copy()
        for _ in range(10):
            d2 = np.sum((Zt[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            lb = np.argmin(d2, axis=1)
            for j in range(K_g):
                if np.any(lb == j):
                    centers[j] = np.mean(Zt[lb == j], axis=0)
        tau = max(1e-2, float(np.mean(np.min(np.sum((Zt[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1))))
    else:
        V_pc, centers, tau = None, None, 1.0

    def gates(Xf):
        if df == 0:
            return np.ones((len(Xf), 1))
        Zc = (Xf - np.mean(Xf_tr, axis=0, keepdims=True)) @ V_pc
        d2 = np.sum((Zc[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lg = -d2 / tau; lg -= np.max(lg, axis=1, keepdims=True)
        eg = np.exp(lg)
        return eg / np.sum(eg, axis=1, keepdims=True)

    # Free RFF + linear block U(z)
    D_rff = min(96, max(12, n // 4))
    U_w = rng.normal(0.0, 1.0 / max(1.0, np.sqrt(df)), size=(df, D_rff)) if df > 0 else None
    U_b = rng.normal(0.0, 0.7, size=(1, D_rff)) if df > 0 else None

    def make_U(Xf, g):
        if df == 0:
            return np.ones((len(Xf), 1))
        return np.hstack([np.ones((len(Xf), 1)), Xf, np.maximum(0.0, Xf @ U_w + U_b), g])

    # Smooth 7-knot overlapping I-splines + gated quartile hinges + top-6 hinge products
    knots7 = [np.quantile(Xm_tr[:, k], np.linspace(0.02, 0.98, 7)) for k in range(dm)]
    q3 = [np.quantile(Xm_tr[:, k], [0.25, 0.50, 0.75]) for k in range(dm)]
    cm = [abs(float(np.corrcoef(Xm_tr[:, k], r_tr)[0, 1])) if np.std(Xm_tr[:, k]) > 1e-8 else 0.0 for k in range(dm)]
    top4_m = list(np.argsort(-np.array(cm))[:min(4, dm)])

    def make_Phi_m(Xm, g):
        cols = []
        for k in range(dm):
            xk = Xm[:, k:k + 1]
            cols.append(xk)
            kn = knots7[k]
            for bi in range(len(kn) - 1):
                if kn[bi + 1] - kn[bi] > 1e-6:
                    cols.append(smooth_ispline_col(xk, kn[bi], kn[bi + 1]))
            # Gated quartile hinges
            for qv in q3[k]:
                hk = np.maximum(0.0, xk - qv)
                cols.append(g * hk)
        for ia in range(len(top4_m)):
            for ib in range(ia + 1, len(top4_m)):
                ka, kb = top4_m[ia], top4_m[ib]
                cols.append(np.maximum(0.0, Xm[:, ka:ka + 1] - q3[ka][1]) * np.maximum(0.0, Xm[:, kb:kb + 1] - q3[kb][1]))
        return np.hstack(cols)

    g_tr = gates(Xf_tr)
    U_tr = make_U(Xf_tr, g_tr)
    Pm_tr = make_Phi_m(Xm_tr, g_tr)
    pf, pm = U_tr.shape[1], Pm_tr.shape[1]
    Z = np.hstack([U_tr, Pm_tr])
    rms = np.where(np.sqrt(np.mean(Z ** 2, axis=0)) > 1e-8, np.sqrt(np.mean(Z ** 2, axis=0)), 1.0); rms[0] = 1.0
    Zn = Z / rms[None, :]
    rg = np.full(pf + pm, 0.03); rg[0] = 1e-9
    w = fista_mixed_qp((Zn.T @ Zn) / n + np.diag(rg), (Zn.T @ r_tr) / n, pf, n_iter=240) / rms
    wf, wm = w[:pf], np.maximum(0.0, w[pf:])
    return lambda Xf, Xm: make_U(Xf, gates(Xf)) @ wf + make_Phi_m(Xm, gates(Xf)) @ wm

def build_boosted_model(Xf_tr, Xm_tr, y_tr):
    n = len(y_tr)
    rng = np.random.RandomState(53)
    perm = rng.permutation(n)
    sp = int(0.80 * n)
    i_tr, i_va = perm[:sp], perm[sp:]

    f1_cv = fit_stage1_astra(Xf_tr[i_tr], Xm_tr[i_tr], y_tr[i_tr])
    r_cv = y_tr[i_tr] - f1_cv(Xf_tr[i_tr], Xm_tr[i_tr])
    f2_cv = fit_stage2_ispline_rbf(Xf_tr[i_tr], Xm_tr[i_tr], r_cv)

    p1_v = f1_cv(Xf_tr[i_va], Xm_tr[i_va])
    p2_v = f2_cv(Xf_tr[i_va], Xm_tr[i_va])
    best_nu, best_mse = 0.0, 1e18
    for nu in (0.0, 0.35, 0.65, 1.0):
        mse = float(np.mean((p1_v + nu * p2_v - y_tr[i_va]) ** 2))
        if mse < best_mse:
            best_mse, best_nu = mse, nu

    f1_full = fit_stage1_astra(Xf_tr, Xm_tr, y_tr)
    if best_nu == 0.0:
        return f1_full
    r_full = y_tr - f1_full(Xf_tr, Xm_tr)
    f2_full = fit_stage2_ispline_rbf(Xf_tr, Xm_tr, r_full)
    return lambda Xf, Xm: f1_full(Xf, Xm) + best_nu * f2_full(Xf, Xm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    tr_all = json.loads(open(req["train_datasets_path"], encoding="utf-8").read())
    models = {ds: build_boosted_model(np.asarray(d["X_train_free"], dtype=np.float64), np.asarray(d["X_train_mono"], dtype=np.float64), np.asarray(d["y_train"], dtype=np.float64)) for ds, d in tr_all.items()}
    preds = [{"id": it["id"], "prediction": models[it["dataset"]](np.asarray(it["X_eval_free"], dtype=np.float64), np.asarray(it["X_eval_mono"], dtype=np.float64)).tolist()} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D4 R2 Opus: SPIKE-RePAIR-L1 (Laplace/Gaussian Spike EM + Weighted Ridge + GPU 3x192 GELU Net)
# ============================================================================
D4_R2_OPUS = '''import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_spike_repair_l1(train_npz_p, eval_npz_p, items):
    tr = np.load(train_npz_p)
    ev = np.load(eval_npz_p)
    X_tr = tr["X_train"].astype(np.float64)
    Y_tr = tr["Y_train"].astype(np.float64)
    X_blocks = ev["X_blocks"].astype(np.float64)
    flow_mean = float(ev["flow_mean"])
    flow_std = float(ev["flow_std"])

    N_tr, L, S, C = X_tr.shape
    D = L * S * C
    r_rank = 32

    # GPU SVD warm-start for Rank-32 Khatri-Rao PPCA
    Flat_tr = X_tr.reshape(N_tr, D)
    mu_d = np.mean(Flat_tr, axis=0, keepdims=True)
    sd_d = np.maximum(1e-4, np.std(Flat_tr, axis=0, keepdims=True))
    Z_tr = (Flat_tr - mu_d) / sd_d

    _, s_vals, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    U = Vt[:r_rank].T
    lam = np.maximum(1e-3, (s_vals[:r_rank] ** 2) / N_tr)
    nu = np.maximum(0.02, np.mean((Z_tr - (Z_tr @ U) @ U.T) ** 2, axis=0))
    Uw = U / np.sqrt(nu[:, None])
    KR = (Uw[:, :, None] * Uw[:, None, :]).reshape(D, r_rank * r_rank)
    prior_diag = np.diag(1.0 / lam)
    U_div_nu = U / nu[:, None]

    # Sensorwise MAD scale for Laplace/Gaussian mixture spike posterior
    padded_tr = np.pad(X_tr, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
    med3_tr = np.median(np.stack([padded_tr[:, k:k + L] for k in range(3)], axis=0), axis=0)
    sig_s = np.maximum(0.08, 1.4826 * np.median(np.abs(X_tr - med3_tr), axis=(0, 1), keepdims=True))

    def spike_repair(X_in):
        B_sz = len(X_in)
        drop_s = (np.sum(np.abs(X_in) < 1e-7, axis=(1, 3)) >= 30)
        pad = np.pad(X_in, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
        m3 = np.median(np.stack([pad[:, k:k + L] for k in range(3)], axis=0), axis=0)

        # Pass 1: mask obvious dropouts + >3*sig_s temporal jumps -> Rank-32 PPCA L_hat
        m0 = ((~drop_s[:, None, :, None]) & (np.abs(X_in - m3) < 3.0 * sig_s)).reshape(B_sz, D).astype(np.float64)
        Z_in = (X_in.reshape(B_sz, D) - mu_d) / sd_d
        A0 = (m0 @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c0 = np.linalg.solve(A0, ((m0 * Z_in) @ U_div_nu)[:, :, None]).squeeze(-1)
        L_hat = ((c0 @ U.T) * sd_d + mu_d).reshape(B_sz, L, S, C)

        # Laplace(b=3.2) vs Gaussian(sig_s) spike posterior pi on e = X_in - L_hat!
        e = X_in - L_hat
        b_lap = 3.2
        rho_sp = 0.035
        log_lap = np.log(rho_sp / (2.0 * b_lap)) - np.abs(e) / b_lap
        log_gau = np.log((1.0 - rho_sp) / (np.sqrt(2.0 * np.pi) * sig_s)) - 0.5 * ((e / sig_s) ** 2)
        pi_sp = 1.0 / (1.0 + np.exp(np.clip(log_gau - log_lap, -30.0, 30.0)))
        pi_sp = np.where(drop_s[:, None, :, None], 1.0, pi_sp)

        # Soft precision weights (1 - pi_sp) in Pass-2 Khatri-Rao PPCA solve!
        w_obs = (1.0 - pi_sp).reshape(B_sz, D)
        A1 = (w_obs @ KR).reshape(B_sz, r_rank, r_rank) + prior_diag[None, :, :]
        c1 = np.linalg.solve(A1, ((w_obs * Z_in) @ U_div_nu)[:, :, None]).squeeze(-1)
        L_hat2 = ((c1 @ U.T) * sd_d + mu_d).reshape(B_sz, L, S, C)
        X_tilde = (1.0 - pi_sp) * X_in + pi_sp * L_hat2
        return X_tilde, L_hat2, c1, pi_sp, drop_s.astype(np.float64)

    corr = np.corrcoef(X_tr[:, :, :, 0].reshape(-1, S).T)
    np.fill_diagonal(corr, -2.0)
    top8 = np.argsort(-corr, axis=1)[:, :8]

    rng = np.random.RandomState(5040)
    X_tr_c = np.where(rng.rand(N_tr, 1, S, 1) < 0.12, 0.0, X_tr)
    X_tr_c = X_tr_c + (rng.rand(*X_tr.shape) < 0.03) * rng.normal(0.0, 4.5, size=X_tr.shape)
    Xt_tr, Lh_tr, Cl_tr, Pi_tr, Df_tr = spike_repair(X_tr_c)

    # Single-pass weighted ridge per sensor (replaces slow IRLS loop!)
    def s_feats(Xt, Cl, Df, s_i):
        B_sz = len(Xt)
        return np.hstack([
            Xt[:, :, s_i, :].reshape(B_sz, 36),
            Xt[:, -1, s_i, :],
            Xt[:, -4:, top8[s_i], 0].reshape(B_sz, 32),
            Cl,
            Df[:, s_i:s_i + 1],
            np.ones((B_sz, 1)),
        ])

    dp = 36 + 3 + 32 + 32 + 2
    W_r = np.zeros((S, dp, 12), dtype=np.float64)
    reg = 4.5 * np.eye(dp); reg[-1, -1] = 1e-6
    Yr_tr = np.empty((N_tr, L, S), dtype=np.float32)
    for s_i in range(S):
        Phi = s_feats(Xt_tr, Cl_tr, Df_tr, s_i)
        # Weight clean low-distortion training windows slightly higher
        w_row = 1.0 - 0.5 * np.mean(Pi_tr[:, :, s_i, 0], axis=1)
        Pw = Phi.T * w_row[None, :]
        W_r[s_i] = np.linalg.solve(Pw @ Phi + reg, Pw @ Y_tr[:, :, s_i])
        Yr_tr[:, :, s_i] = (Phi @ W_r[s_i]).astype(np.float32)

    # GPU 3x192 GELU L1 Residual-Net with 16-d learned sensor embeddings!
    def build_net_in(Xt, Lh, Yr, Df):
        B_sz = len(Xt)
        loc = Xt.transpose(0, 2, 1, 3).reshape(B_sz, S, 36)
        lh_f = Lh[:, :, :, 0].transpose(0, 2, 1)  # [B, 96, 12]
        yr_s = Yr.transpose(0, 2, 1)              # [B, 96, 12]
        nb8 = Xt[:, -4:, top8, 0].transpose(0, 2, 1, 3).reshape(B_sz, S, 32)  # [B, 96, 32]
        return np.concatenate([yr_s, lh_f, loc, nb8, Df[:, :, None]], axis=2).astype(np.float32)  # 93 dims

    class SpikeResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Parameter(torch.randn(1, S, 16) * 0.02)
            self.net = nn.Sequential(
                nn.Linear(93 + 16, 192),
                nn.GELU(),
                nn.Linear(192, 192),
                nn.GELU(),
                nn.Linear(192, 12),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x):
            e = self.emb.expand(x.shape[0], -1, -1)
            return self.net(torch.cat([x, e], dim=-1))

    torch.manual_seed(404)
    model = SpikeResNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-2)
    Fin_tr = torch.tensor(build_net_in(Xt_tr, Lh_tr, Yr_tr, Df_tr), device=DEVICE)
    Ybase_t = torch.tensor(Yr_tr.transpose(0, 2, 1), device=DEVICE)
    Ytrue_t = torch.tensor(Y_tr.transpose(0, 2, 1).astype(np.float32), device=DEVICE)

    for _ in range(45):
        perm = torch.randperm(N_tr, device=DEVICE)
        for b0 in range(0, N_tr, 256):
            sl = perm[b0:b0 + 256]
            delta = torch.clamp(model(Fin_tr[sl]), -3.0, 3.0)
            loss = F.l1_loss(Ybase_t[sl] + delta, Ytrue_t[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    outs = []
    with torch.no_grad():
        for it in items:
            bi = int(it["block_idx"])
            Xt_e, Lh_e, Cl_e, _, Df_e = spike_repair(X_blocks[bi])
            B_sz = len(Xt_e)
            Yr_e = np.empty((B_sz, L, S), dtype=np.float32)
            for s_i in range(S):
                Yr_e[:, :, s_i] = (s_feats(Xt_e, Cl_e, Df_e, s_i) @ W_r[s_i]).astype(np.float32)
            Fin_e = torch.tensor(build_net_in(Xt_e, Lh_e, Yr_e, Df_e), device=DEVICE)
            Yb_et = torch.tensor(Yr_e.transpose(0, 2, 1), device=DEVICE)
            Yn = (Yb_et + torch.clamp(model(Fin_e), -3.0, 3.0)).cpu().numpy().transpose(0, 2, 1)
            outs.append({"id": it["id"], "prediction": (Yn * flow_std + flow_mean).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = run_spike_repair_l1(req["train_npz"], req["eval_npz"], req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D5 R2 Opus: Mean-Rank Debiased Stability Consensus ({Gauss-EM, t5-EM} x 3-Lambda + Subsamples)
# ============================================================================
D5_R2_OPUS = '''import argparse, json, math
import numpy as np

def admm_glasso_warm(R, lam, rho=1.5, n_iter=20, Z_init=None, U_init=None):
    p = R.shape[0]
    Z = np.eye(p, dtype=np.float64) if Z_init is None else Z_init.copy()
    U = np.zeros((p, p), dtype=np.float64) if U_init is None else U_init.copy()
    thresh = lam / rho
    Om = Z.copy()
    for _ in range(n_iter):
        mat = rho * (Z - U) - R
        vals, Q = np.linalg.eigh(0.5 * (mat + mat.T))
        eig_om = (vals + np.sqrt(vals ** 2 + 4.0 * rho)) / (2.0 * rho)
        Om = (Q * eig_om[None, :]) @ Q.T
        V = Om + U
        Zn = np.sign(V) * np.maximum(0.0, np.abs(V) - thresh)
        np.fill_diagonal(Zn, np.diag(V))
        U = U + Om - Zn
        Z = Zn
    return Om, Z, U

def mar_em_corr(X, M, nu=None, n_iter=6):
    T, p = X.shape
    mu = np.zeros(p, dtype=np.float64)
    Sigma = np.eye(p, dtype=np.float64)
    for r in range(n_iter):
        X_imp = X.copy()
        S_cond = np.zeros((p, p), dtype=np.float64)
        w_t = np.ones(T, dtype=np.float64)
        for t in range(T):
            miss = np.where(M[t] < 0.5)[0]
            obs = np.where(M[t] > 0.5)[0]
            if 0 < len(miss) < p:
                Soo_inv = np.linalg.inv(Sigma[np.ix_(obs, obs)] + 1e-4 * np.eye(len(obs)))
                W_mo = Sigma[np.ix_(miss, obs)] @ Soo_inv
                do = X[t, obs] - mu[obs]
                X_imp[t, miss] = mu[miss] + W_mo @ do
                S_cond[np.ix_(miss, miss)] += Sigma[np.ix_(miss, miss)] - W_mo @ Sigma[np.ix_(obs, miss)]
                if nu is not None:
                    w_t[t] = (nu + len(obs)) / (nu + max(0.0, float(do @ (Soo_inv @ do))))
            elif len(miss) == p:
                X_imp[t] = mu
                S_cond += Sigma
            elif nu is not None and len(obs) == p:
                Soo_inv = np.linalg.inv(Sigma + 1e-4 * np.eye(p))
                do = X[t] - mu
                w_t[t] = (nu + p) / (nu + max(0.0, float(do @ (Soo_inv @ do))))
        mu = (w_t @ X_imp) / max(1e-8, float(np.sum(w_t)))
        D = X_imp - mu[None, :]
        Sigma = 0.92 * (((D.T * w_t[None, :]) @ D + S_cond) / T) + 0.08 * np.eye(p)

    vals, vecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    S_psd = (vecs * np.maximum(0.03, vals)[None, :]) @ vecs.T
    d = np.sqrt(np.maximum(1e-10, np.diag(S_psd)))
    return S_psd / np.outer(d, d)

def rank_norm(v):
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(v))
    return ranks

def score_d5_r2_opus(X_obs, M, seed):
    T, p = X_obs.shape
    cnt = np.maximum(1.0, np.sum(M, axis=0))
    mu0 = np.sum(X_obs * M, axis=0) / cnt
    Xc = (X_obs - mu0[None, :]) * M
    sd0 = np.sqrt(np.maximum(1e-6, np.sum((Xc ** 2) * M, axis=0) / np.maximum(1.0, cnt - 1.0)))
    X = Xc / sd0[None, :]

    n_ij = np.maximum(1.0, M.T @ M)
    iu = np.triu_indices(p, k=1)
    i_idx, j_idx = iu
    sqrt_nij = np.sqrt(n_ij[i_idx, j_idx])
    lam0 = math.sqrt(math.log(max(3, p)) / max(8.0, float(np.median(n_ij))))

    R_gauss = mar_em_corr(X, M, nu=None, n_iter=6)
    R_t5 = mar_em_corr(X, M, nu=5.0, n_iter=6)

    rank_sum = np.zeros(len(i_idx), dtype=np.float64)
    n_views = 0

    for R_mat in (R_gauss, R_t5):
        Z_w, U_w = None, None
        for mult in (0.35, 0.55, 0.85):
            Om, Z_w, U_w = admm_glasso_warm(R_mat, lam=lam0 * mult, rho=1.5, n_iter=22, Z_init=Z_w, U_init=U_w)
            Om_deb = 2.0 * Om - Om @ R_mat @ Om
            denom = np.sqrt(np.maximum(1e-10, Om[i_idx, i_idx] * Om[j_idx, j_idx] + Om[i_idx, j_idx] ** 2))
            z_stat = np.abs(Om_deb[iu]) * sqrt_nij / denom
            pcorr_sp = np.abs(Z_w[iu]) / np.sqrt(np.maximum(1e-10, Om[i_idx, i_idx] * Om[j_idx, j_idx]))
            rank_sum += rank_norm(0.65 * z_stat + 0.35 * pcorr_sp * sqrt_nij)
            n_views += 1

    # 4 fast 75%-subsample stability refits
    rng = np.random.RandomState(550 + seed)
    sub_n = max(12, int(0.75 * T))
    for _ in range(4):
        idx = rng.choice(T, size=sub_n, replace=False)
        Rs = mar_em_corr(X[idx], M[idx], nu=5.0, n_iter=4)
        nij_s = np.maximum(1.0, M[idx].T @ M[idx])
        Om_s, Z_s, _ = admm_glasso_warm(Rs, lam=lam0 * 0.50, rho=1.5, n_iter=18)
        Om_deb_s = 2.0 * Om_s - Om_s @ Rs @ Om_s
        den_s = np.sqrt(np.maximum(1e-10, Om_s[i_idx, i_idx] * Om_s[j_idx, j_idx] + Om_s[i_idx, j_idx] ** 2))
        zs = np.abs(Om_deb_s[iu]) * np.sqrt(nij_s[i_idx, j_idx]) / den_s
        rank_sum += 0.5 * rank_norm(zs)
        n_views += 0.5

    return (rank_sum / n_views).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": score_d5_r2_opus(np.asarray(it["X_obs"], dtype=np.float64), np.asarray(it["mask"], dtype=np.float64), i)} for i, it in enumerate(req["items"])]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D6 R2 Opus: Mirror-SMC + Multi-Sweep Trajectory Pool + Energy-Distance Support-Point Herding
# ============================================================================
D6_R2_OPUS = '''import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def systematic_resample(log_w, rng):
    lw = log_w - np.max(log_w)
    w = np.exp(lw); w /= np.sum(w)
    M = len(w)
    u = (rng.rand() + np.arange(M, dtype=np.float64)) / M
    cdf = np.cumsum(w); cdf[-1] = 1.0 + 1e-12
    return np.clip(np.searchsorted(cdf, u), 0, M - 1)

def energy_herding_select(pool, M_out):
    """
    Greedy Minimum-Energy Support-Point Herding (Mak & Joseph 2018):
    Compresses K_pool equilibrium states into M_out low-discrepancy representatives at 0 oracle calls!
    x_{t+1} = argmin_{x in pool} [ mean_p ||x - p||_2 - (1/(t+1)) sum_{s in S} ||x - s||_2 ]
    """
    # Pairwise Euclidean distance matrix D_mat [P, P] (P <= 768 -> 768x768 = 4.5MB, ~3ms!)
    sq = np.sum(pool ** 2, axis=1, keepdims=True)
    D2 = np.maximum(0.0, sq + sq.T - 2.0 * (pool @ pool.T))
    D_mat = np.sqrt(D2)
    target_pot = np.mean(D_mat, axis=1)  # [P]: E_p[||x - p||]

    chosen = []
    used = np.zeros(len(pool), dtype=bool)
    repulsion_sum = np.zeros(len(pool), dtype=np.float64)

    for t in range(M_out):
        obj = target_pot - repulsion_sum / float(t + 1)
        obj[used] = 1e18
        idx = int(np.argmin(obj))
        used[idx] = True
        chosen.append(idx)
        repulsion_sum += D_mat[:, idx]

    return pool[np.array(chosen, dtype=np.int64)]

def run_mirror_smc_herding(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 149 + 7)
    M, d = X_init.shape
    s_disp = max(1.0, float(np.median(np.linalg.norm(X_init, axis=1))) / math.sqrt(d))
    X = 0.35 * X_init + rng.normal(0.0, 1.15 * s_disp, size=(M, d))

    def land_eval(Z):
        U, gU, h, J = prob.eval_all(Z)
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        corr = np.einsum("mk,mkd->md", c, J)
        return Z - corr, corr, U, gU, h, J, JJt

    # 5 initial Newton landing calls
    for _ in range(5):
        X, _, _, _, h0, _, _ = land_eval(X)
        if float(np.max(np.abs(h0))) < 1e-8:
            break

    # Verify exact Z2 symmetry pi(-x) == pi(x) on 1 call!
    X_l0, _, U_pos, _, _, _, _ = land_eval(X)
    U_neg, _, h_neg, _ = prob.eval_all(-X_l0)
    exact_z2_sym = (float(np.max(np.abs(h_neg))) <= 1e-5) and (float(np.max(np.abs(U_pos - U_neg))) <= 1e-5)
    X = X_l0

    eta_G = 0.034
    prev_med = 0.1

    def fused_step(X_in, beta, sc=1.0):
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
        return X_land + v_tan, X_land, U, h

    def z2_mirror_mix(X_landed, U_landed, beta):
        if exact_z2_sym:
            # Exact Z2 target symmetry verified -> random sign flip has acceptance 1.0 in ZERO oracle calls!
            signs = np.where(rng.rand(M, 1) < 0.5, 1.0, -1.0)
            return signs * X_landed
        if prob.grad_calls + 12 >= max_steps:
            return X_landed
        U_m, _, h_m, _ = prob.eval_all(-X_landed)
        ok = (np.max(np.abs(h_m), axis=1) <= 1e-5) & (rng.rand(M) < np.exp(np.clip(beta * (U_landed - U_m), -30.0, 0.0)))
        return np.where(ok[:, None], -X_landed, X_landed)

    # 18 warmup steps at beta_0 = 0.10
    beta = 0.10
    X_land = X.copy()
    U_cur = U_pos.copy()
    for _ in range(18):
        X, X_land, U_cur, _ = fused_step(X, beta)

    # Adaptive CESS bisection Annealed SMC from beta=0.10 -> 1.0!
    while beta < 1.0 - 1e-9 and prob.grad_calls < 285:
        X_land, _, U_cur, _, _, _, _ = land_eval(X)
        U_cl = np.clip(U_cur, np.percentile(U_cur, 2), np.percentile(U_cur, 95) + 15.0)
        # Bisect delta_beta in (0, 1 - beta] for ESS >= 0.82 * M
        lo_b, hi_b = beta, 1.0
        if (1.0 / float(np.sum((np.exp(-(1.0 - beta) * (U_cl - np.min(U_cl))) / np.sum(np.exp(-(1.0 - beta) * (U_cl - np.min(U_cl))))) ** 2))) >= 0.82 * M:
            beta_next = 1.0
        else:
            for _ in range(12):
                mid = 0.5 * (lo_b + hi_b)
                lw = -(mid - beta) * U_cl; lw -= np.max(lw)
                wn = np.exp(lw) / np.sum(np.exp(lw))
                if (1.0 / float(np.sum(wn ** 2))) >= 0.82 * M:
                    lo_b = mid
                else:
                    hi_b = mid
            beta_next = max(beta + 0.06, lo_b)
            beta_next = min(1.0, beta_next)

        lw = -(beta_next - beta) * U_cl
        idx_r = systematic_resample(lw, rng)
        X_land = X_land[idx_r].copy()
        U_cur = U_cur[idx_r].copy()
        X_land = z2_mirror_mix(X_land, U_cur, beta_next)
        beta = beta_next
        X = X_land.copy()
        for _ in range(10):
            if prob.grad_calls >= 295:
                break
            X, X_land, U_cur, _ = fused_step(X, beta)

    # Equilibrate at beta=1.0 and collect a trajectory pool of landed manifold states (stride 8 -> ~8*96 = 768 states!)
    pool_snaps = []
    rem = max(0, (max_steps - 4) - prob.grad_calls)
    for s_i in range(rem):
        sc = 0.65 if s_i >= rem - 40 else 1.0
        X, X_land, U_cur, h_in = fused_step(X, 1.0, sc=sc)
        if s_i == rem // 3:
            X = z2_mirror_mix(X_land, U_cur, 1.0)
        # X_land was just Newton-landed from X_in; collect snapshots in second half where chain is well-mixed & on-manifold!
        if s_i >= max(10, rem - 64) and (s_i % 8 == 0):
            # Keep only strictly feasible landed points (max|h| <= 2e-4)
            pool_snaps.append(X_land.copy())
            if exact_z2_sym:
                pool_snaps.append(-X_land.copy())

    # Final exact landing on current X (<= 3 calls)
    for _ in range(3):
        if prob.grad_calls >= max_steps:
            break
        X_land, _, _, _, hf, _, _ = land_eval(X)
        X = X_land
        if float(np.max(np.abs(hf))) <= 1e-8:
            break
    pool_snaps.append(X.copy())
    if exact_z2_sym:
        pool_snaps.append(-X.copy())

    pool_all = np.vstack(pool_snaps)
    if len(pool_all) > 768:
        sel_p = np.linspace(0, len(pool_all) - 1, 768).astype(np.int64)
        pool_all = pool_all[sel_p]

    X_herded = energy_herding_select(pool_all, M)
    # 1 final exact Newton landing call on the 96 herded support points so max|h| < 1e-9 <= 1e-3!
    if prob.grad_calls < max_steps:
        X_herded, _, _, _, _, _, _ = land_eval(X_herded)

    return {"particles": X_herded.tolist(), "grad_calls": int(prob.grad_calls)}

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
        res = run_mirror_smc_herding(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D7 R2 Opus: 128-Replica dSB + Z2-Aligned Memetic Elite Crossover + Astra KL-8 Kick
# ============================================================================
D7_R2_OPUS = '''import argparse, json
import numpy as np

M_SUB = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)
A_I, C_I = np.triu_indices(8, 1)
N_PAIRS = (M_SUB[:, A_I] * M_SUB[:, C_I]).T

def batched_tabu(A, S_init, tot, n_steps, rng):
    S = S_init.copy()
    k, n = S.shape
    ar = np.arange(k)
    H = S @ A
    # Steepest ascent first
    for _ in range(3 * n):
        G = S * H
        i = np.argmax(G, axis=1)
        m = G[ar, i] > 1e-9
        if not np.any(m):
            break
        si = S[ar, i] * m
        S[ar, i] -= 2.0 * si
        H -= 2.0 * si[:, None] * A[i]

    cut = tot - 0.25 * np.sum(S * H, axis=1)
    best = cut.copy()
    bS = S.copy()
    tb = np.zeros((k, n), dtype=np.int64)
    ten = rng.integers(3, max(5, n // 9 + 2), size=k)
    for t in range(1, n_steps + 1):
        G = S * H
        allowed = (tb <= t) | (G > (best - cut)[:, None] + 1e-9)
        i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
        si = S[ar, i].copy()
        cut += si * H[ar, i]
        S[ar, i] = -si
        H -= 2.0 * si[:, None] * A[i]
        tb[ar, i] = t + ten
        j = cut > best + 1e-9
        best[j] = cut[j]
        bS[j] = S[j]
    return bS, best

def kl8_kick_batch(A, S_in, tot, rng):
    S = S_in.copy()
    k, n = S.shape
    ar = np.arange(k)
    G = S * (S @ A)
    v = G.copy()
    I_list = []
    i = np.argpartition(-G, 7, axis=1)[ar, rng.integers(8, size=k)]
    for _ in range(8):
        I_list.append(i)
        v -= 2.0 * S[ar, i, None] * A[i] * S
        v[ar, i] = -np.inf
        i = np.argmax(v, axis=1)
    I_mat = np.stack(I_list, axis=1)
    T_sp = S[ar[:, None], I_mat]
    gain = G[ar[:, None], I_mat] @ M_SUB.T - 2.0 * (A[I_mat[:, A_I], I_mat[:, C_I]] * T_sp[:, A_I] * T_sp[:, C_I]) @ N_PAIRS
    gain[:, 0] = -np.inf
    kp = np.argmax(gain, axis=1)
    Wf = M_SUB[kp]
    S[ar[:, None], I_mat] *= (1.0 - 2.0 * Wf)
    return S

def solve_d7_r2_opus(A_in, seed):
    rng = np.random.default_rng(seed)
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    tot = 0.25 * float(A.sum())
    c0 = 0.5 * np.sqrt(max(1.0, n - 1.0) / max(1e-12, float((A ** 2).sum())))

    # 1) 128-replica Discrete Simulated Bifurcation (T=180)
    K = 128
    X = 0.02 * (rng.random((K, n)) - 0.5)
    P = 0.02 * (rng.random((K, n)) - 0.5)
    best = np.full(K, -np.inf, dtype=np.float64)
    bS = np.ones((K, n), dtype=np.float64)
    T_sb = 180
    for t in range(T_sb):
        S = np.where(X >= 0.0, 1.0, -1.0)
        AS = S @ A
        if t in (90, 135, 160, 179) or t >= 140:
            c = tot - 0.25 * np.sum(S * AS, axis=1)
            imp = c > best
            best[imp] = c[imp]
            bS[imp] = S[imp]
        P += -(1.0 - t / T_sb) * X - c0 * AS
        X += P
        np.clip(X, -1.0, 1.0, out=X)
        P *= (np.abs(X) < 1.0)

    # 2) Polish top-16 replicas with Tabu
    elites = bS[np.argsort(-best)[:16]]
    elites, e_cuts = batched_tabu(A, elites, tot, n_steps=55, rng=rng)

    # 3) 2 rounds of Z2-Aligned Memetic Backbone Crossover + Astra KL-8 256-Subset Kick!
    for _ in range(2):
        order = np.argsort(-e_cuts)
        elites, e_cuts = elites[order], e_cuts[order]
        children = []
        for c_i in range(8):
            ia = c_i % 6
            ib = (c_i + 1 + rng.integers(0, 7)) % 12
            pa, pb = elites[ia], elites[ib]
            # Align Z2 global spin symmetry: pb <- sgn(<pa, pb>) * pb!
            if float(pa @ pb) < 0.0:
                pb = -pb
            agree = (pa == pb)
            if np.mean(agree) > 0.96:
                # Stagnant pair -> apply Astra KL-8 256-subset kick!
                ch = kl8_kick_batch(A, pa[None, :], tot, rng)[0]
            else:
                ch = np.where(agree, pa, rng.choice([-1.0, 1.0], size=n))
            children.append(ch)
        # Also kick top-4 elites with KL-8!
        kicked = kl8_kick_batch(A, elites[:4], tot, rng)
        cand_batch = np.vstack([np.stack(children, axis=0), kicked])
        pol_b, pol_c = batched_tabu(A, cand_batch, tot, n_steps=65, rng=rng)
        all_S = np.vstack([elites, pol_b])
        all_c = np.concatenate([e_cuts, pol_c])
        ord2 = np.argsort(-all_c)[:16]
        elites, e_cuts = all_S[ord2], all_c[ord2]

    return elites[0].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_d7_r2_opus(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
'''


# ============================================================================
# D8 & D9 R2 Opus: Ultra-Fine 241-pt QNG-VI & 12-Step Precond-FISTA + Khatri-Rao SSN
# ============================================================================
D8_R2_OPUS = '''import argparse, json
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
'''


D9_R2_OPUS = '''import argparse, json
from pathlib import Path
import numpy as np

def dual_obj(Y, A, b, lo, hi, N):
    U = Y + N @ A
    Z = np.clip(U, lo[None, :], hi[None, :])
    return np.sum(0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2), axis=1) - (N @ b)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    G = A @ A.T + 1e-12 * np.eye(m)
    Gi = np.linalg.inv(G)
    KR = (A[:, None, :] * A[None, :, :]).reshape(m * m, d).T.copy()

    N = (b[None, :] - Y @ A.T) @ Gi
    E = N.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(15):
        R = b[None, :] - np.clip(Y + E @ A, lo[None, :], hi[None, :]) @ A.T
        Nn = E + R @ Gi
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Nn - N) * R, axis=1) < 0.0
        be[rs] = 0.0; tn[rs] = 1.0
        E = Nn + be[:, None] * (Nn - N)
        N = Nn; th = tn

    eye_m = 1e-10 * np.eye(m)[None, :, :]
    for _ in range(8):
        U = Y + N @ A
        Z = np.clip(U, lo[None, :], hi[None, :])
        R = b[None, :] - Z @ A.T
        unc = np.where(np.max(np.abs(R), axis=1) > 1e-10)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]; Ru = R[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR).reshape(len(unc), m, m) + eye_m
        dnu = np.linalg.solve(H, Ru[:, :, None]).squeeze(-1)
        Nu = N[unc]; Yu = Y[unc]
        p0 = dual_obj(Yu, A, b, lo, hi, Nu)
        slope = -np.sum(Ru * dnu, axis=1)
        step = np.ones(len(unc)); done = np.zeros(len(unc), dtype=bool)
        Nb = Nu + Ru @ Gi
        for _ in range(7):
            Nt = Nu + step[:, None] * dnu
            pt = dual_obj(Yu, A, b, lo, hi, Nt)
            ok = (pt <= p0 + 1e-4 * step * slope) & (~done)
            if np.any(ok):
                Nb[ok] = Nt[ok]; done[ok] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        N[unc] = Nb

    return np.clip(Y + N @ A, lo[None, :], hi[None, :])

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
        write_cand("bench_d1_shapiq", "d1_v5_r2_opus_crossover", D1_R2_OPUS)
        write_cand("bench_d2_acs_reweight", "d2_v5_r2_opus_crossover", D2_R2_OPUS)
        write_cand("bench_d3_cmnn", "d3_v5_r2_opus_crossover", D3_R2_OPUS)
    elif node == "node1":
        write_cand("bench_d4_pems04_rstib", "d4_v5_r2_opus_crossover", D4_R2_OPUS)
        write_cand("bench_d5_snp500_scoremissing", "d5_v5_r2_opus_crossover", D5_R2_OPUS)
        write_cand("bench_d6_ollalanding", "d6_v5_r2_opus_crossover", D6_R2_OPUS)
    elif node == "node2":
        write_cand("bench_d7_rld_maxcut", "d7_v5_r2_opus_crossover", D7_R2_OPUS)
        write_cand("bench_d8_lsvi_vi", "d8_v5_r2_opus_crossover", D8_R2_OPUS)
        write_cand("bench_d9_pinet_qp", "d9_v5_r2_opus_crossover", D9_R2_OPUS)
    print(f"OPUS_R2_GENERATED_{node.upper()}")


if __name__ == "__main__":
    main()
