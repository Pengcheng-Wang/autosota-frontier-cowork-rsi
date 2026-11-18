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
