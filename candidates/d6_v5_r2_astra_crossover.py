import argparse, importlib.util, json, math
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
