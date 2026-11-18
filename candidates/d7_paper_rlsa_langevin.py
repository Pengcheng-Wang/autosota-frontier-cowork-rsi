import argparse, json, math
import numpy as np

VARIANT = 'paper_rlsa'

def cut_vals_batch(A, S):
    # S: [M, n] in {-1, +1} -> cut values [M]
    return 0.25 * (np.sum(A) - np.sum((S @ A) * S, axis=1))

def greedy_1flip_polish(A, s_vec, max_sweeps=15):
    s = s_vec.copy()
    field = A @ s  # gain of flipping i is +s_i * field_i!
    for _ in range(max_sweeps):
        gains = s * field
        i_best = int(np.argmax(gains))
        if gains[i_best] <= 1e-9:
            break
        old_s = s[i_best]
        s[i_best] = -old_s
        field -= 2.0 * old_s * A[:, i_best]
    return s

def solve_maxcut(A, seed):
    n = len(A)
    rng = np.random.RandomState(2000 + seed)

    if VARIANT in ("paper_rlsa", "autosota_rlsa_leap"):
        # Regularized Langevin Simulated Annealing across M=24 parallel continuous spin chains in [-1, 1]^n
        M = 24
        X = rng.uniform(-0.5, 0.5, size=(M, n))
        best_s = np.ones(n)
        best_c = -1.0
        K_steps = 180
        for t in range(1, K_steps + 1):
            temp = 1.5 * ((0.02 / 1.5) ** (t / K_steps))
            # Soft spin relaxation s = tanh(X / tau), gradient of -0.25 s^T A s reduces to +A s (want s^T A s minimized!)
            S_soft = np.tanh(X / 0.5)
            grad = -(S_soft @ A)  # ascent direction for -s^T A s is -(A s)
            X = np.clip(X + 0.08 * grad + math.sqrt(0.04 * temp) * rng.normal(0, 1, size=X.shape), -2.5, 2.5)
            if VARIANT == "autosota_rlsa_leap" and t % 30 == 0:
                S_bin = np.where(X >= 0, 1.0, -1.0)
                c_now = cut_vals_batch(A, S_bin)
                top_idx = int(np.argmax(c_now))
                worst4 = np.argsort(c_now)[:4]
                X[worst4] = X[top_idx][None, :] + rng.normal(0, 0.25, size=(4, n))
        S_bin = np.where(X >= 0, 1.0, -1.0)
        c_all = cut_vals_batch(A, S_bin)
        return S_bin[int(np.argmax(c_all))].tolist()

    if VARIANT == "comp_spectral_greedy":
        # Top negative-eigenvalue eigenvector of A (= top eigenvector of D - A / -A) + hyperplane sweep + 1-flip polish
        vals, vecs = np.linalg.eigh(A)
        V_low = vecs[:, :4]  # 4 smallest eigenvectors of A minimize s^T A s!
        best_s, best_c = None, -1.0
        for _ in range(24):
            g = rng.normal(0, 1, size=4)
            s0 = np.where((V_low @ g) >= 0, 1.0, -1.0)
            sp = greedy_1flip_polish(A, s0)
            c = float(0.25 * (np.sum(A) - float(sp @ A @ sp)))
            if c > best_c:
                best_c, best_s = c, sp
        return best_s.tolist()

    if VARIANT == "decision_incremental_sa":
        # Proposer D: Fast O(1)-delta incremental neighbor-field Simulated Annealing across 6 restarts
        best_s, best_c = None, -1.0
        total_flips = 28 * n
        for _ in range(6):
            s = np.where(rng.rand(n) >= 0.5, 1.0, -1.0)
            field = A @ s
            cur_c = float(0.25 * (np.sum(A) - float(s @ field)))
            idxs = rng.randint(0, n, size=total_flips)
            u_rand = rng.rand(total_flips)
            for k in range(total_flips):
                T_k = 1.8 * (0.02 ** (k / float(total_flips)))
                i = int(idxs[k])
                gain = s[i] * field[i]  # exact cut change if i flips!
                if gain >= 0.0 or u_rand[k] < math.exp(max(-30.0, gain / T_k)):
                    old_s = s[i]
                    s[i] = -old_s
                    field -= 2.0 * old_s * A[:, i]
                    cur_c += gain
                    if cur_c > best_c:
                        best_c, best_s = cur_c, s.copy()
        best_s = greedy_1flip_polish(A, best_s)
        return best_s.tolist()

    if VARIANT == "coord_burer_monteiro_tabu":
        # Proposer A: Low-Rank Burer-Monteiro Riemannian Block-Coordinate SDP Relaxation (rank r=10)
        # + Goemans-Williamson Random Hyperplane Rounding + Perturbed Iterated Local Search (ILS)!
        r = 10
        V = rng.normal(0.0, 1.0, size=(n, r))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        # Exact zero-tuning cyclical Barzilai-Borwein / row-normalization minimization of Tr(V^T A V):
        # Notice minimizing sum_{i!=j} A_{ij} <v_i, v_j> s.t. ||v_i||=1 has exact row update v_i <- - (A_i V) / ||A_i V||!
        for it in range(55):
            G = A @ V  # [n, r]
            step = 0.65 if it < 35 else 0.35
            V_new = (1.0 - step) * V - step * (G / np.maximum(1e-8, np.linalg.norm(G, axis=1, keepdims=True)))
            V = V_new / np.maximum(1e-8, np.linalg.norm(V_new, axis=1, keepdims=True))

        # Round 32 random Goemans-Williamson hyperplanes in vectorized batch, pick top 4, polish with ILS!
        H = rng.normal(0.0, 1.0, size=(r, 32))
        S_cand = np.where((V @ H).T >= 0.0, 1.0, -1.0)  # [32, n]
        c_cand = cut_vals_batch(A, S_cand)
        top4 = np.argsort(-c_cand)[:4]

        best_s, best_c = None, -1.0
        for idx in top4:
            s = greedy_1flip_polish(A, S_cand[int(idx)], max_sweeps=30)
            c = float(0.25 * (np.sum(A) - float(s @ A @ s)))
            if c > best_c:
                best_c, best_s = c, s.copy()
            # 3 Perturbed Iterated Local Search kicks from SDP basin
            for _ in range(3):
                s_kick = s.copy()
                flip_k = rng.choice(n, size=max(3, n // 22), replace=False)
                s_kick[flip_k] *= -1.0
                s_pol = greedy_1flip_polish(A, s_kick, max_sweeps=30)
                cp = float(0.25 * (np.sum(A) - float(s_pol @ A @ s_pol)))
                if cp > best_c:
                    best_c, best_s = cp, s_pol.copy()
                    s = s_pol
        return best_s.tolist()

    raise ValueError(VARIANT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_maxcut(np.asarray(it["adj"], dtype=np.float64), int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
