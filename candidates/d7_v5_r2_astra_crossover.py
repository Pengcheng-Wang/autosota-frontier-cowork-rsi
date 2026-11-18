import argparse, json
import numpy as np

M_SUB = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)
A_I, C_I = np.triu_indices(8, 1)
N_PAIRS = (M_SUB[:, A_I] * M_SUB[:, C_I]).T  # [28, 256]

def solve_d7_r2_astra(A_in, seed):
    rng = np.random.default_rng(seed)
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    tot = 0.25 * float(A.sum())
    c0 = 0.5 * np.sqrt(max(1.0, n - 1.0) / max(1e-12, float((A ** 2).sum())))

    # Phase 1: Opus 64-replica discrete Simulated Bifurcation (T=150)
    K = 64
    X = 0.02 * (rng.random((K, n)) - 0.5)
    P = 0.02 * (rng.random((K, n)) - 0.5)
    best = np.full(K, -np.inf, dtype=np.float64)
    bS = np.ones((K, n), dtype=np.float64)
    T_sb = 150
    for t in range(T_sb):
        S = np.where(X >= 0.0, 1.0, -1.0)
        AS = S @ A
        if 2 * t >= T_sb:
            c = tot - 0.25 * np.sum(S * AS, axis=1)
            imp = c > best
            best[imp] = c[imp]
            bS[imp] = S[imp]
        P += -(1.0 - t / T_sb) * X - c0 * AS
        X += P
        np.clip(X, -1.0, 1.0, out=X)
        P *= (np.abs(X) < 1.0)

    # Steepest 1-flip ascent on top-20 replicas
    S = bS[np.argsort(-best)[:20]].copy()
    H = S @ A
    ar = np.arange(len(S))
    for _ in range(4 * n):
        G = S * H
        i = np.argmax(G, axis=1)
        m = G[ar, i] > 1e-9
        if not np.any(m):
            break
        si = S[ar, i] * m
        S[ar, i] -= 2.0 * si
        H -= 2.0 * si[:, None] * A[i]

    cuts = tot - 0.25 * np.sum(S * H, axis=1)
    order = np.argsort(-cuts)
    # Select up to 12 sign-canonicalized elites + run interleaved Tabu + Astra 2^8-subset 8-spin connected moves!
    S = S[order[:12]].copy()
    H = S @ A
    k = len(S)
    ar = np.arange(k)
    cut = tot - 0.25 * np.sum(S * H, axis=1)
    best_e = cut.copy()
    bS_e = S.copy()
    tb = np.zeros((k, n), dtype=np.int64)
    ten = rng.integers(max(3, n // 20), max(6, n // 8), size=k)
    three_mask = (np.sum(M_SUB, axis=1) >= 2)

    for t in range(1, 96):
        if t in (24, 48, 72):
            # Astra Exact 2^8 = 256-subset neighbor-connected 8-spin move on current best elites!
            S = bS_e.copy()
            H = S @ A
            G = S * H
            v = G.copy()
            I_list = []
            i = np.argpartition(-G, 7, axis=1)[ar, rng.integers(8, size=k)]
            for _ in range(8):
                I_list.append(i)
                # Boost connected neighbors + update marginal gain!
                v += 0.15 * (np.abs(A[i]) > 0) - 2.0 * S[ar, i, None] * A[i] * S
                v[ar, i] = -np.inf
                i = np.argmax(v, axis=1)
            I_mat = np.stack(I_list, axis=1)
            T_sp = S[ar[:, None], I_mat]
            gain = G[ar[:, None], I_mat] @ M_SUB.T - 2.0 * (A[I_mat[:, A_I], I_mat[:, C_I]] * T_sp[:, A_I] * T_sp[:, C_I]) @ N_PAIRS
            kb = np.argmax(gain, axis=1)
            kesc = np.argmax(np.where(three_mask[None, :] & (np.arange(256)[None, :] > 0), gain, -np.inf), axis=1)
            kp = np.where(gain[ar, kb] > 1e-9, kb, kesc)
            Wf = M_SUB[kp]
            S[ar[:, None], I_mat] *= (1.0 - 2.0 * Wf)
            tb[ar[:, None], I_mat] = t + 7 * Wf.astype(np.int64)
            H = S @ A
            cut = tot - 0.25 * np.sum(S * H, axis=1)
        else:
            G = S * H
            allowed = (tb <= t) | (G > (best_e - cut)[:, None] + 1e-9)
            i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
            si = S[ar, i].copy()
            cut += si * H[ar, i]
            S[ar, i] = -si
            H -= 2.0 * si[:, None] * A[i]
            tb[ar, i] = t + ten

        j = cut > best_e + 1e-9
        best_e[j] = cut[j]
        bS_e[j] = S[j]

    return bS_e[int(np.argmax(best_e))].tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": solve_d7_r2_astra(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
