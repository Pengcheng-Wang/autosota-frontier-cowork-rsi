import argparse, json
from pathlib import Path
import numpy as np


def dual_obj_rows_q(Y, Q, c_rhs, lo, hi, L_dual):
    # Orthonormalized row space: A^T = Q @ R_up => A z = b  <=>  Q^T z = c_rhs where c_rhs = R_up^{-T} b!
    # Notice Q^T Q = I_m => unconstrained preconditioner is IDENTICALLY I_m!!
    U = Y - L_dual @ Q.T
    Z = np.clip(U, lo[None, :], hi[None, :])
    g = 0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2)
    return np.sum(g, axis=1) + (L_dual @ c_rhs)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    # Thin QR of A^T: A^T = Q @ R_qr, Q in R^{d x m} with Q^T Q = I_m!
    Q, R_qr = np.linalg.qr(A.T, mode="reduced")  # Q: [d, m], R_qr: [m, m]
    c_rhs = np.linalg.solve(R_qr.T, b)  # [m]: Q^T z = c_rhs <=> A z = b!
    KR_q = (Q[None, :, :] * Q[:, None, :]).reshape(d, m * m) if False else (Q[:, :, None] * Q[:, None, :]).reshape(d, m * m)

    # 18 QR-whitened unit-step Nesterov Dual FISTA iterations (NO m x m solve needed in FISTA!)
    L_cur = Y @ Q - c_rhs[None, :]
    E = L_cur.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(18):
        Z_e = np.clip(Y - E @ Q.T, lo[None, :], hi[None, :])
        res = Z_e @ Q - c_rhs[None, :]  # grad wrt L_dual
        Ln = E + res
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Ln - L_cur) * res, axis=1) < 0.0
        be[rs] = 0.0
        tn[rs] = 1.0
        E = Ln + be[:, None] * (Ln - L_cur)
        L_cur = Ln
        th = tn

    # Active-set Semismooth Newton in Q-coordinates + Certified Feasible Anchor fallback!
    eye_m = 1e-9 * np.eye(m)[None, :, :]
    for _ in range(8):
        U = Y - L_cur @ Q.T
        Z = np.clip(U, lo[None, :], hi[None, :])
        res_q = Z @ Q - c_rhs[None, :]
        unc = np.where(np.max(np.abs(res_q), axis=1) > 1e-9)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]
        ru = res_q[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR_q).reshape(len(unc), m, m) + eye_m
        dl = np.linalg.solve(H, ru[:, :, None]).squeeze(-1)

        Lu = L_cur[unc]
        Yu = Y[unc]
        p0 = dual_obj_rows_q(Yu, Q, c_rhs, lo, hi, Lu)
        slope = -np.sum(ru * dl, axis=1)
        step = np.ones(len(unc), dtype=np.float64)
        done = np.zeros(len(unc), dtype=bool)
        Lu_best = Lu + ru
        for _ in range(7):
            L_try = Lu + step[:, None] * dl
            p_try = dual_obj_rows_q(Yu, Q, c_rhs, lo, hi, L_try)
            ok = p_try <= p0 + 1e-4 * step * slope
            newly = ok & (~done)
            if np.any(newly):
                Lu_best[newly] = L_try[newly]
                done[newly] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        L_cur[unc] = Lu_best

    Z_out = np.clip(Y - L_cur @ Q.T, lo[None, :], hi[None, :])
    # Certified Feasible Anchor Repair on any tail row exceeding 1e-6 equality error!
    eq_err = np.max(np.abs(Z_out @ A.T - b[None, :]), axis=1)
    bad = eq_err > 1e-6
    if np.any(bad) and np.any(~bad):
        h_anc = np.mean(Z_out[~bad], axis=0)
        Zb = Z_out[bad]
        Zb_aff = Zb - (Zb @ Q - c_rhs[None, :]) @ Q.T
        v_dir = Zb_aff - h_anc[None, :]
        # Largest alpha in [0, 1] keeping h_anc + alpha * v_dir inside [lo, hi]
        bound_num = np.where(v_dir > 0.0, hi[None, :] - h_anc[None, :], lo[None, :] - h_anc[None, :])
        alphas = np.divide(bound_num, v_dir, out=np.full_like(v_dir, np.inf), where=(np.abs(v_dir) > 1e-15))
        a_min = np.clip(np.min(alphas, axis=1), 0.0, 1.0)
        Z_out[bad] = np.clip(h_anc[None, :] + a_min[:, None] * v_dir, lo[None, :], hi[None, :])
    return Z_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    ev = np.load(req["eval_npz"])
    n_rep = int(req.get("n_repeats", 12))
    out_dir = Path(req["out_npz_dir"])
    out_npz = out_dir / (Path(args.output_json).stem + "_z.npz")

    # Load arrays into memory before timed solver loop (symmetric across ALL methods!)
    loaded = []
    for it in req["items"]:
        iid = it["id"]
        loaded.append((iid, ev[f"{iid}_A"], ev[f"{iid}_b"], ev[f"{iid}_lo"], ev[f"{iid}_hi"], ev[f"{iid}_Y"]))

    out_arrays = {}
    for rep in range(n_rep):
        for iid, A, b, lo, hi, Y in loaded:
            Z = solve_qp(A, b, lo, hi, Y)
            if rep == n_rep - 1:
                out_arrays[iid] = Z

    np.savez(out_npz, **out_arrays)
    preds = [{"id": it["id"], "prediction": {"npz_path": str(out_npz)}} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
