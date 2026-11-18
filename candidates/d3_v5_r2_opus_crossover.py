import argparse, json
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
