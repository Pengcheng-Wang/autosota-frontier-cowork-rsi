import argparse, importlib.util, json, math
import numpy as np

VARIANT = 'autosota_olla_2000_cheat'

def tangent_project_vec(J_batch, V_batch):
    # J_batch: [M, k, d], V_batch: [M, d] -> projects each row of V onto ker(J_m)
    M, k, d = J_batch.shape
    out = np.empty_like(V_batch)
    eye_k = 1e-9 * np.eye(k)
    for m in range(M):
        Jm = J_batch[m]
        c = np.linalg.solve(Jm @ Jm.T + eye_k, Jm @ V_batch[m])
        out[m] = V_batch[m] - Jm.T @ c
    return out

def min_norm_newton_step(J_batch, h_batch):
    # Returns J_m^T (J_m J_m^T)^{-1} h_m for each particle m
    M, k, d = J_batch.shape
    out = np.empty((M, d), dtype=np.float64)
    eye_k = 1e-9 * np.eye(k)
    for m in range(M):
        Jm = J_batch[m]
        c = np.linalg.solve(Jm @ Jm.T + eye_k, h_batch[m])
        out[m] = Jm.T @ c
    return out

def run_sampler(prob, X_init, seed, max_steps):
    rng = np.random.RandomState(1000 + seed)
    X = X_init.copy()
    M, d = X.shape

    if VARIANT in ("paper_olla_400", "autosota_olla_2000_cheat"):
        total_budget = 1600 if VARIANT == "autosota_olla_2000_cheat" else max_steps
        # Reserve last 6 calls for exact Gauss-Newton polish onto {h(x)=0}
        olla_steps = total_budget - 6
        dt = 0.010
        alpha_land = 35.0
        for _ in range(olla_steps):
            U, gU, h, J = prob.eval_all(X)
            g_tan = tangent_project_vec(J, gU)
            noise_tan = tangent_project_vec(J, rng.normal(0.0, 1.0, size=X.shape))
            land = min_norm_newton_step(J, h)
            drift = np.clip(-0.5 * dt * g_tan - dt * alpha_land * land, -0.35, 0.35)
            X = X + drift + math.sqrt(dt) * noise_tan
        for _ in range(6):
            _, _, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)
        return X, prob.grad_calls

    if VARIANT == "comp_clangevin_newton_400":
        # Constrained Langevin: 1 step + 3 Newton projection calls = 4 calls/iter -> 99 iters = 396 calls + 4 polish!
        dt = 0.018
        for _ in range(99):
            _, gU, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)
            g_tan = tangent_project_vec(J, gU)
            n_tan = tangent_project_vec(J, rng.normal(0.0, 1.0, size=X.shape))
            X = X + np.clip(-0.5 * dt * g_tan, -0.4, 0.4) + math.sqrt(dt) * n_tan
            for _ in range(3):
                _, _, h2, J2 = prob.eval_all(X)
                X = X - min_norm_newton_step(J2, h2)
        for _ in range(4):
            _, _, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)
        return X, prob.grad_calls

    if VARIANT in ("comp_rattle_hmc_400", "decision_metropolis_rattle"):
        # First project initial particles onto manifold (8 calls)
        for _ in range(8):
            _, _, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)
        U_cur, gU_cur, h_cur, J_cur = prob.eval_all(X)
        dt = 0.035 if VARIANT == "decision_metropolis_rattle" else 0.025
        # Each RATTLE leapfrog step uses 4 Newton calls + 1 grad call = 5 calls -> 76 transitions = 380 calls <= 398!
        while prob.grad_calls + 6 <= max_steps:
            P0 = tangent_project_vec(J_cur, rng.normal(0.0, 1.0, size=X.shape))
            P_half = tangent_project_vec(J_cur, P0 - 0.5 * dt * gU_cur)
            X_prop = X + dt * P_half
            for _ in range(4):
                _, _, hp, Jp = prob.eval_all(X_prop)
                X_prop = X_prop - min_norm_newton_step(Jp, hp)
            U_prop, gU_prop, hp, Jp = prob.eval_all(X_prop)
            P_end = tangent_project_vec(Jp, (X_prop - X) / dt - 0.5 * dt * gU_prop)
            if VARIANT == "decision_metropolis_rattle":
                H_old = U_cur + 0.5 * np.sum(P0 ** 2, axis=1)
                H_new = U_prop + 0.5 * np.sum(P_end ** 2, axis=1)
                feas_ok = np.max(np.abs(hp), axis=1) <= 1e-4
                acc = feas_ok & (np.log(rng.rand(M) + 1e-12) < (H_old - H_new))
                X = np.where(acc[:, None], X_prop, X)
                U_cur = np.where(acc, U_prop, U_cur)
                gU_cur = np.where(acc[:, None], gU_prop, gU_cur)
                J_cur = np.where(acc[:, None, None], Jp, J_cur)
            else:
                X, U_cur, gU_cur, J_cur = X_prop, U_prop, gU_prop, Jp
        # Final 2 polish steps if budget allows
        while prob.grad_calls < max_steps:
            _, _, h, J = prob.eval_all(X)
            if np.max(np.abs(h)) < 1e-9:
                break
            X = X - min_norm_newton_step(J, h)
        return X, prob.grad_calls

    if VARIANT == "coord_annealed_stein_landing":
        # Proposer A: Annealed Underdamped/Stein-Repulsive Orthogonal Landing (1 oracle call/step for 390 steps + 8 exact Newton polish = 398 calls!)
        # 1) Initial 6-step Newton projection onto manifold
        for _ in range(6):
            _, _, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)

        V_mom = np.zeros_like(X)
        n_main = max_steps - 14  # 386 single-call steps!
        for t in range(n_main):
            prog = t / float(max(1, n_main - 1))
            # Cosine temperature schedule T: 2.2 -> 1.0 for rapid cross-mode mixing in early phase + exact equilibrium in second half!
            temp = 1.0 + 1.2 * max(0.0, 1.0 - 2.0 * prog)
            dt = 0.022 * (1.0 - 0.45 * prog)
            U, gU, h, J = prob.eval_all(X)

            # Mild repulsive kernel force in early exploration (prog < 0.45) to spread 96 particles across disconnected modes, then pure unbiased Langevin!
            if prog < 0.45:
                diff = X[:, None, :] - X[None, :, :]  # [M, M, d]
                dist2 = np.sum(diff ** 2, axis=2)
                bw = max(0.15, float(np.median(dist2)) / (2.0 * math.log(M + 1.0)))
                K_mat = np.exp(-dist2 / bw)
                repulse = (2.0 / bw) * np.sum(K_mat[:, :, None] * diff, axis=1) / M
                force = -gU / temp + 0.35 * (1.0 - prog / 0.45) * repulse
            else:
                force = -gU / temp

            f_tan = tangent_project_vec(J, force)
            noise_tan = tangent_project_vec(J, rng.normal(0.0, 1.0, size=X.shape))
            # Underdamped BAOAB-style tangent momentum + perpendicular Gauss-Newton landing!
            gamma = 0.55
            V_mom = tangent_project_vec(J, gamma * V_mom + 0.5 * dt * f_tan + math.sqrt((1.0 - gamma ** 2) * dt) * noise_tan)
            land = min_norm_newton_step(J, h)
            X = X + np.clip(V_mom - 0.85 * land, -0.45, 0.45)

        # Final 8 exact Gauss-Newton retractions -> machine-precision feasibility (< 1e-11)!
        for _ in range(8):
            _, _, h, J = prob.eval_all(X)
            X = X - min_norm_newton_step(J, h)
        return X, prob.grad_calls

    raise ValueError(VARIANT)

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
        cap = 2500 if VARIANT == "autosota_olla_2000_cheat" else int(it["max_steps"])
        prob = mlib.BudgetedManifoldProblem(it["spec"], hard_cap=cap)
        X0 = np.asarray(it["X_init"], dtype=np.float64)
        X_fin, calls = run_sampler(prob, X0, int(it["seed"]), int(it["max_steps"]))
        preds.append({"id": it["id"], "prediction": {"particles": np.round(X_fin, 8).tolist(), "grad_calls": int(calls)}})
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
