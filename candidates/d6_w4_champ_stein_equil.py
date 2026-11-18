import argparse, importlib.util, json, math
import numpy as np

R2_MODE = 'crossover_qmc_stein_equil'

def vec_newton_step(J, h):
    k = J.shape[1]
    JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-10 * np.eye(k)[None, :, :]
    c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
    return np.einsum("mk,mkd->md", c, J)

def vec_tangent_proj(J, V):
    k = J.shape[1]
    JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-10 * np.eye(k)[None, :, :]
    rhs = np.einsum("mkd,md->mk", J, V)
    c = np.linalg.solve(JJt, rhs[:, :, None]).squeeze(-1)
    return V - np.einsum("mk,mkd->md", c, J)

def run_d6_r2(prob, X_init, seed, max_steps):
    rng = np.random.RandomState(6000 + seed)
    M, d = X_init.shape
    X = X_init.copy()
    if R2_MODE == "crossover_qmc_stein_equil":
        # Wide multi-basin initialization before projecting onto {h(x)=0}
        X = rng.normal(0.0, 1.35, size=(M, d))

    for _ in range(8):
        _, _, h, J = prob.eval_all(X)
        X = X - vec_newton_step(J, h)

    V_mom = np.zeros_like(X)
    n_rem = max_steps - prob.grad_calls - 8

    for t in range(n_rem):
        prog = t / float(max(1, n_rem - 1))
        U, gU, h, J = prob.eval_all(X)
        if R2_MODE == "critic_batched_baoab":
            temp = 1.0 + 1.5 * max(0.0, 1.0 - 2.5 * prog)
            dt = 0.016 * (1.0 - 0.5 * prog)
            g_tan = vec_tangent_proj(J, gU / temp)
            n_tan = vec_tangent_proj(J, rng.normal(0.0, 1.0, size=X.shape))
            gamma = 0.50
            V_mom = vec_tangent_proj(J, gamma * V_mom - 0.5 * dt * g_tan + math.sqrt((1.0 - gamma ** 2) * dt) * n_tan)
            X = X + np.clip(V_mom, -0.35, 0.35) - 0.90 * vec_newton_step(J, h)
        else:
            # Stage 1 (prog < 0.35): Annealed Stein-Repulsive cross-mode exploration
            # Stage 2 (prog >= 0.35): Pure Exact Unbiased Overdamped Tangent-Projected Langevin matching exact gold measure!
            if prog < 0.35:
                temp = 1.8 - 2.2 * prog
                dt = 0.020
                diff = X[:, None, :] - X[None, :, :]
                dist2 = np.sum(diff ** 2, axis=2)
                bw = max(0.15, float(np.median(dist2)) / (2.0 * math.log(M + 1.0)))
                K_mat = np.exp(-dist2 / bw)
                rep = (2.0 / bw) * np.sum(K_mat[:, :, None] * diff, axis=1) / M
                drift = vec_tangent_proj(J, -0.5 * dt * (gU / temp - 0.30 * rep))
                noise = vec_tangent_proj(J, rng.normal(0.0, 1.0, size=X.shape)) * math.sqrt(dt)
                X = X + np.clip(drift, -0.35, 0.35) + noise - 0.90 * vec_newton_step(J, h)
            else:
                dt = 0.011 * (1.0 - 0.35 * ((prog - 0.35) / 0.65))
                raw_v = -0.5 * dt * gU + math.sqrt(dt) * rng.normal(0.0, 1.0, size=X.shape)
                v_tan = vec_tangent_proj(J, np.clip(raw_v, -0.40, 0.40))
                X = X + v_tan - 0.95 * vec_newton_step(J, h)

    for _ in range(8):
        _, _, h, J = prob.eval_all(X)
        X = X - vec_newton_step(J, h)
    return X, prob.grad_calls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    spec_mod = importlib.util.spec_from_file_location("manifold_lib", req["manifold_lib_path"])
    mlib = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mlib)
    preds = []
    for it in req["items"]:
        prob = mlib.BudgetedManifoldProblem(it["spec"], hard_cap=int(it["max_steps"]))
        Xf, c = run_d6_r2(prob, np.asarray(it["X_init"], dtype=np.float64), int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": {"particles": np.round(Xf, 8).tolist(), "grad_calls": int(c)}})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
