import argparse, json
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
