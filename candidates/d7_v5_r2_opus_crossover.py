import argparse, json
import numpy as np

M_SUB = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)
A_I, C_I = np.triu_indices(8, 1)
N_PAIRS = (M_SUB[:, A_I] * M_SUB[:, C_I]).T

def batched_tabu(A, S_init, tot, n_steps, rng):
    S = S_init.copy()
    k, n = S.shape
    ar = np.arange(k)
    H = S @ A
    # Steepest ascent first
    for _ in range(3 * n):
        G = S * H
        i = np.argmax(G, axis=1)
        m = G[ar, i] > 1e-9
        if not np.any(m):
            break
        si = S[ar, i] * m
        S[ar, i] -= 2.0 * si
        H -= 2.0 * si[:, None] * A[i]

    cut = tot - 0.25 * np.sum(S * H, axis=1)
    best = cut.copy()
    bS = S.copy()
    tb = np.zeros((k, n), dtype=np.int64)
    ten = rng.integers(3, max(5, n // 9 + 2), size=k)
    for t in range(1, n_steps + 1):
        G = S * H
        allowed = (tb <= t) | (G > (best - cut)[:, None] + 1e-9)
        i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
        si = S[ar, i].copy()
        cut += si * H[ar, i]
        S[ar, i] = -si
        H -= 2.0 * si[:, None] * A[i]
        tb[ar, i] = t + ten
        j = cut > best + 1e-9
        best[j] = cut[j]
        bS[j] = S[j]
    return bS, best

def kl8_kick_batch(A, S_in, tot, rng):
    S = S_in.copy()
    k, n = S.shape
    ar = np.arange(k)
    G = S * (S @ A)
    v = G.copy()
    I_list = []
    i = np.argpartition(-G, 7, axis=1)[ar, rng.integers(8, size=k)]
    for _ in range(8):
        I_list.append(i)
        v -= 2.0 * S[ar, i, None] * A[i] * S
        v[ar, i] = -np.inf
        i = np.argmax(v, axis=1)
    I_mat = np.stack(I_list, axis=1)
    T_sp = S[ar[:, None], I_mat]
    gain = G[ar[:, None], I_mat] @ M_SUB.T - 2.0 * (A[I_mat[:, A_I], I_mat[:, C_I]] * T_sp[:, A_I] * T_sp[:, C_I]) @ N_PAIRS
    gain[:, 0] = -np.inf
    kp = np.argmax(gain, axis=1)
    Wf = M_SUB[kp]
    S[ar[:, None], I_mat] *= (1.0 - 2.0 * Wf)
    return S

def solve_d7_r2_opus(A_in, seed):
    rng = np.random.default_rng(seed)
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    tot = 0.25 * float(A.sum())
    c0 = 0.5 * np.sqrt(max(1.0, n - 1.0) / max(1e-12, float((A ** 2).sum())))

    # 1) 128-replica Discrete Simulated Bifurcation (T=180)
    K = 128
    X = 0.02 * (rng.random((K, n)) - 0.5)
    P = 0.02 * (rng.random((K, n)) - 0.5)
    best = np.full(K, -np.inf, dtype=np.float64)
    bS = np.ones((K, n), dtype=np.float64)
    T_sb = 180
    for t in range(T_sb):
        S = np.where(X >= 0.0, 1.0, -1.0)
        AS = S @ A
        if t in (90, 135, 160, 179) or t >= 140:
            c = tot - 0.25 * np.sum(S * AS, axis=1)
            imp = c > best
            best[imp] = c[imp]
            bS[imp] = S[imp]
        P += -(1.0 - t / T_sb) * X - c0 * AS
        X += P
        np.clip(X, -1.0, 1.0, out=X)
        P *= (np.abs(X) < 1.0)

    # 2) Polish top-16 replicas with Tabu
    elites = bS[np.argsort(-best)[:16]]
    elites, e_cuts = batched_tabu(A, elites, tot, n_steps=55, rng=rng)

    # 3) 2 rounds of Z2-Aligned Memetic Backbone Crossover + Astra KL-8 256-Subset Kick!
    for _ in range(2):
        order = np.argsort(-e_cuts)
        elites, e_cuts = elites[order], e_cuts[order]
        children = []
        for c_i in range(8):
            ia = c_i % 6
            ib = (c_i + 1 + rng.integers(0, 7)) % 12
            pa, pb = elites[ia], elites[ib]
            # Align Z2 global spin symmetry: pb <- sgn(<pa, pb>) * pb!
            if float(pa @ pb) < 0.0:
                pb = -pb
            agree = (pa == pb)
            if np.mean(agree) > 0.96:
                # Stagnant pair -> apply Astra KL-8 256-subset kick!
                ch = kl8_kick_batch(A, pa[None, :], tot, rng)[0]
            else:
                ch = np.where(agree, pa, rng.choice([-1.0, 1.0], size=n))
            children.append(ch)
        # Also kick top-4 elites with KL-8!
        kicked = kl8_kick_batch(A, elites[:4], tot, rng)
        cand_batch = np.vstack([np.stack(children, axis=0), kicked])
        pol_b, pol_c = batched_tabu(A, cand_batch, tot, n_steps=65, rng=rng)
        all_S = np.vstack([elites, pol_b])
        all_c = np.concatenate([e_cuts, pol_c])
        ord2 = np.argsort(-all_c)[:16]
        elites, e_cuts = all_S[ord2], all_c[ord2]

    return elites[0].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_d7_r2_opus(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
