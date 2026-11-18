import argparse, json
import numpy as np

def cut_astra(A_in, seed):
    A = np.array(A_in, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    scale = float(np.max(np.abs(A)))
    if scale > 0.0:
        A /= scale
    n = len(A)
    r = np.random.default_rng(seed)

    # Signed-Laplacian power iteration + hyperplane rounding (24 chains) + 6 random chains = 30 parallel chains
    V = r.normal(size=(n, 8))
    D = np.sum(np.abs(A), axis=1, keepdims=True)
    for _ in range(16):
        V = D * V - A @ V
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)

    S_hyp = np.where((V @ r.normal(size=(8, 24))).T >= 0.0, 1.0, -1.0)
    S_rnd = r.choice([-1.0, 1.0], size=(6, n))
    S = np.vstack([S_hyp, S_rnd])  # [30, n]

    # Exact 2^8 = 256 multi-spin subset evaluator tables
    M = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)  # [256, 8]
    a_idx, c_idx = np.triu_indices(8, 1)
    N_pairs = (M[:, a_idx] * M[:, c_idx]).T  # [28, 256]
    j_ar = np.arange(len(S))
    U = np.zeros_like(S, dtype=np.int64)

    H = S @ A
    e = -0.25 * np.sum(S * H, axis=1)
    P = S.copy()
    best = e.copy()

    for t in range(210):
        if t in (55, 110, 165):
            # Grow coupled 8-vertex block by marginal gain and evaluate ALL 256 flip subsets in 1 GEMM!
            S = P.copy()
            G = S * (S @ A)
            v = G.copy()
            I_list = []
            i = np.argpartition(-G, 7, axis=1)[j_ar, r.integers(8, size=len(S))]
            for _ in range(8):
                I_list.append(i)
                v -= 2.0 * S[j_ar, i, None] * A[i] * S
                v[j_ar, i] = -np.inf
                i = np.argmax(v, axis=1)
            I_mat = np.stack(I_list, axis=1)  # [30, 8]
            T_sp = S[j_ar[:, None], I_mat]    # [30, 8]
            gain = G[j_ar[:, None], I_mat] @ M.T - 2.0 * (A[I_mat[:, a_idx], I_mat[:, c_idx]] * T_sp[:, a_idx] * T_sp[:, c_idx]) @ N_pairs
            k_best = np.argmax(gain, axis=1)
            # If best subset is 0 (no gain), take least-costly 3-flip escape!
            three_mask = (np.sum(M, axis=1) == 3)
            k_esc = np.argmax(np.where(three_mask[None, :], gain, -np.inf), axis=1)
            k_pick = np.where(gain[j_ar, k_best] > 1e-9, k_best, k_esc)
            W_flip = M[k_pick]
            S[j_ar[:, None], I_mat] *= (1.0 - 2.0 * W_flip)
            U.fill(0)
            U[j_ar[:, None], I_mat] = (t + 6) * W_flip.astype(np.int64)
            H = S @ A
            e = -0.25 * np.sum(S * H, axis=1)
        else:
            G = S * H
            allowed = (U <= t) | ((e[:, None] + G) > (best[:, None] + 1e-9))
            i = np.argmax(np.where(allowed, G, -np.inf), axis=1)
            e += G[j_ar, i]
            H -= 2.0 * S[j_ar, i, None] * A[i]
            S[j_ar, i] *= -1.0
            U[j_ar, i] = t + r.integers(6, 16, size=len(S))

        imp = e > best + 1e-9
        P[imp] = S[imp]
        best[imp] = e[imp]

    # Final exact 1-flip polish on best chain
    win = P[int(np.argmax(best))].copy()
    Hw = A @ win
    for _ in range(2 * n):
        gw = win * Hw
        bi = int(np.argmax(gw))
        if gw[bi] <= 1e-9:
            break
        Hw -= 2.0 * win[bi] * A[:, bi]
        win[bi] = -win[bi]
    return win.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    preds = [{"id": it["id"], "prediction": cut_astra(it["adj"], int(it["seed"]))} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
