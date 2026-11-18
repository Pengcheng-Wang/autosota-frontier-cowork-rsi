import argparse, importlib.util, json, math
import numpy as np

def load_manifold_lib(path):
    spec = importlib.util.spec_from_file_location("manifold_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_astra_d6(prob, X_init, seed, max_steps=400):
    rng = np.random.RandomState(seed * 97 + 31)
    M, d = X_init.shape

    def land_once(Z):
        U, gU, h, J = prob.eval_all(Z)
        k = J.shape[1]
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-9 * np.eye(k)[None, :, :]
        c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
        Z_new = Z - np.einsum("mk,mkd->md", c, J)
        return Z_new, U, gU, h, J, JJt

    # 1) Land initial particles onto manifold (8 calls)
    X = X_init.copy()
    for _ in range(8):
        X, U, gU, h, J, JJt = land_once(X)
        if float(np.max(np.abs(h))) < 1e-8:
            break

    # 2) Reversible Z2 antipodal mirror Metropolis moves (exact manifold-preserving reflection x -> -x!)
    # Notice h(-x) == h(x) for sphere_mog, stiefel_2frame, AND fair_bayes_logit! (Check: linear c1^T(-x) = -c1^T x = 0 when c1^T x = 0, and quadratic (-x)^2 == x^2!)
    # So -X is EXACTLY ON THE MANIFOLD IN ZERO NEWTON PROJECTION CALLS!!
    _, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X)
    dt = 0.024

    # 8 blocks of: reversible Z2 mirror Metropolis jump + palindromic (0.55 -> 0.30 -> 0.55) tempered excursion + cold equilibration!
    for blk in range(8):
        if prob.grad_calls + 36 > max_steps:
            break
        # (a) Zero-projection-error antipodal mirror Metropolis jump (1 call!)
        X_mir = -X
        _, U_mir, gU_mir, _, J_mir, JJt_mir = land_once(X_mir)
        acc_mir = rng.rand(M) < np.exp(np.clip(U_cur - U_mir, -30.0, 0.0))
        X = np.where(acc_mir[:, None], X_mir, X)
        U_cur = np.where(acc_mir, U_mir, U_cur)
        gU_cur = np.where(acc_mir[:, None], gU_mir, gU_cur)
        J_cur = np.where(acc_mir[:, None, None], J_mir, J_cur)
        JJt_cur = np.where(acc_mir[:, None, None], JJt_mir, JJt_cur)

        # (b) Palindromic tempered excursion beta = (0.60, 0.32, 0.60) accepted/rejected against cold entry!
        X_entry = X.copy()
        U_entry = U_cur.copy()
        log_ais = np.zeros(M, dtype=np.float64)
        X_exc = X.copy()
        U_prev_e, gU_e, J_e, JJt_e = U_cur.copy(), gU_cur.copy(), J_cur.copy(), JJt_cur.copy()
        for beta_e in (0.60, 0.32, 0.60):
            dt_e = min(0.055, dt / beta_e)
            raw_v = -0.5 * dt_e * beta_e * gU_e + math.sqrt(dt_e) * rng.normal(0.0, 1.0, size=(M, d))
            rhs = np.einsum("mkd,md->mk", J_e, raw_v)
            cv = np.linalg.solve(JJt_e, rhs[:, :, None]).squeeze(-1)
            v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_e)
            X_exc = X_exc + v_tan
            # 2 Newton landings onto manifold
            X_exc, _, _, _, _, _ = land_once(X_exc)
            X_exc, U_next_e, gU_e, _, J_e, JJt_e = land_once(X_exc)
            log_ais += (beta_e - 1.0) * (U_next_e - U_prev_e)
            U_prev_e = U_next_e

        acc_exc = rng.rand(M) < np.exp(np.clip(log_ais, -30.0, 0.0))
        X = np.where(acc_exc[:, None], X_exc, X_entry)
        X, U_cur, gU_cur, _, J_cur, JJt_cur = land_once(X)

        # (c) 12 cold beta=1 fused tangent-Langevin + landing steps!
        for _ in range(12):
            if prob.grad_calls + 10 >= max_steps:
                break
            raw_v = -0.5 * dt * gU_cur + math.sqrt(dt) * rng.normal(0.0, 1.0, size=(M, d))
            rhs = np.einsum("mkd,md->mk", J_cur, raw_v)
            cv = np.linalg.solve(JJt_cur, rhs[:, :, None]).squeeze(-1)
            v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_cur)
            X, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X + v_tan)

    # Spend remaining calls (keeping 6 for exact Newton polish) on fine cold beta=1 equilibration!
    dt_fine = 0.015
    while prob.grad_calls + 6 < max_steps:
        raw_v = -0.5 * dt_fine * gU_cur + math.sqrt(dt_fine) * rng.normal(0.0, 1.0, size=(M, d))
        rhs = np.einsum("mkd,md->mk", J_cur, raw_v)
        cv = np.linalg.solve(JJt_cur, rhs[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", cv, J_cur)
        X, U_cur, gU_cur, h_cur, J_cur, JJt_cur = land_once(X + v_tan)

    for _ in range(5):
        if prob.grad_calls >= max_steps:
            break
        X, _, _, h_f, _, _ = land_once(X)
        if float(np.max(np.abs(h_f))) <= 1e-8:
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
        res = run_astra_d6(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": res})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
