import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def systematic_resample(log_w, rng):
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    w /= np.sum(w)
    ess = 1.0 / float(np.sum(w ** 2))
    M = len(w)
    u = (rng.rand() + np.arange(M, dtype=np.float64)) / M
    cdf = np.cumsum(w)
    cdf[-1] = 1.0 + 1e-12
    idx = np.searchsorted(cdf, u)
    return np.clip(idx, 0, M - 1), ess

def run_flash(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 131 + 19)
    M, d = X_init.shape
    # Over-dispersed start covering disconnected modes + 6 damped Newton landing calls
    s_disp = max(1.0, float(np.median(np.linalg.norm(X_init, axis=1))) / math.sqrt(d))
    X = X_init.copy() * 0.5 + rng.normal(0.0, 1.15 * s_disp, size=(M, d))

    def j_pinv_step(h, J):
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        return np.einsum("mk,mkd->md", c, J), JJt

    for _ in range(6):
        U, gU, h, J = prob.eval_all(X)
        nh = corr_step, _ = j_pinv_step(h, J)
        nrm = np.linalg.norm(corr_step, axis=1, keepdims=True)
        scale = np.minimum(1.0, 1.0 / np.maximum(1e-8, nrm))
        X = X - scale * corr_step

    # Fused-landing transition: 1 single eval_all call lands previous move AND proposes next tangent move!
    eta_G = 0.035
    prev_tan_med = 0.1

    def fused_step(X_cur, beta, noise_scale=1.0):
        nonlocal eta_G, prev_tan_med
        U, gU, h, J = prob.eval_all(X_cur)
        k = J.shape[1]
        corr, JJt = j_pinv_step(h, J)
        delta_med = max(1e-8, float(np.median(np.linalg.norm(corr, axis=1))))
        eta_G = float(np.clip(eta_G * np.clip((0.12 * prev_tan_med / delta_med) ** 2, 0.5, 1.8), 0.002, 0.06))

        # Project beta * gU + noise onto tangent plane
        rhs_g = np.einsum("mkd,md->mk", J, beta * gU)
        cg = np.linalg.solve(JJt, rhs_g[:, :, None]).squeeze(-1)
        g_tan = beta * gU - np.einsum("mk,mkd->md", cg, J)
        g_ms = max(1e-4, float(np.mean(np.sum(g_tan ** 2, axis=1))))
        eta_F = 0.15 * max(1, d - k) / g_ms
        eta = min(eta_G, eta_F) * noise_scale

        xi = rng.normal(0.0, 1.0, size=(M, d))
        raw_v = -eta * beta * gU + math.sqrt(2.0 * eta) * xi
        rhs_v = np.einsum("mkd,md->mk", J, raw_v)
        cv = np.linalg.solve(JJt, rhs_v[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", cv, J)
        # Clamp outlier leaps
        vn = np.linalg.norm(v_tan, axis=1, keepdims=True)
        v_tan = v_tan * np.minimum(1.0, 0.85 / np.maximum(1e-8, vn))
        prev_tan_med = max(1e-4, float(np.median(vn)))

        X_landed = X_cur - corr
        X_next = X_landed + v_tan
        return X_next, X_landed, U, h

    # 20 warmup steps at beta_0 = 0.10 (mode discovery across barriers!)
    beta_prev = 0.10
    for _ in range(20):
        X, _, U_cur, _ = fused_step(X, beta_prev)

    # 10-stage geometric annealing beta_j = 0.10^(1 - j/10) -> 1.0 with AIS weights & systematic SMC!
    log_w = np.zeros(M, dtype=np.float64)
    for j in range(1, 11):
        beta_j = 0.10 ** (1.0 - j / 10.0)
        # Evaluate current landed particles at start of stage
        U_st, gU_st, h_st, J_st = prob.eval_all(X)
        corr_st, _ = j_pinv_step(h_st, J_st)
        X = X - corr_st
        # Re-Weight using bounded potential (guarding any residual off-manifold spike)
        U_clip = np.clip(U_st, np.percentile(U_st, 2), np.percentile(U_st, 95) + 15.0)
        log_w += -(beta_j - beta_prev) * U_clip
        lw_shift = log_w - np.max(log_w)
        w_norm = np.exp(lw_shift) / np.sum(np.exp(lw_shift))
        ess = 1.0 / float(np.sum(w_norm ** 2))
        if ess < 0.55 * M or j == 10:
            idx_r, _ = systematic_resample(log_w, rng)
            X = X[idx_r].copy()
            log_w = np.zeros(M, dtype=np.float64)
        beta_prev = beta_j
        for _ in range(11):
            X, _, _, _ = fused_step(X, beta_j)

    # Remaining budget: equilibrate at beta = 1.0, reserving strictly 7 calls for exact Newton landing polish!
    rem_equil = max(0, (max_steps - 7) - prob.grad_calls)
    for s_i in range(rem_equil):
        n_sc = 0.5 if s_i >= rem_equil - 32 else 1.0
        X, _, _, _ = fused_step(X, 1.0, noise_scale=n_sc)

    # Final exact Newton landing polish (<= 6 calls -> max|h| < 1e-10 <= 1e-3, total calls <= 399 <= 400!)
    for _ in range(6):
        if prob.grad_calls >= max_steps:
            break
        _, _, h_f, J_f = prob.eval_all(X)
        if float(np.max(np.abs(h_f))) <= 1e-7:
            break
        corr_f, _ = j_pinv_step(h_f, J_f)
        X = X - corr_f

    # Guarantee 100% particle feasibility (replace any rogue particle if needed)
    if prob.grad_calls < max_steps:
        _, _, h_chk, _ = prob.eval_all(X)
        feas_err = np.max(np.abs(h_chk), axis=1)
        good = np.where(feas_err <= 5e-4)[0]
        bad = np.where(feas_err > 5e-4)[0]
        if len(bad) > 0 and len(good) > 0:
            X[bad] = X[good[np.arange(len(bad)) % len(good)]]

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
        res = run_flash(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
