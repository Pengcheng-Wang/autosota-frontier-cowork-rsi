import argparse, json
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
