import argparse, json
from pathlib import Path
import numpy as np

def dual_obj_q(Y, Q, c_rhs, lo, hi, L_d):
    U = Y - L_d @ Q.T
    Z = np.clip(U, lo[None, :], hi[None, :])
    return np.sum(0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2), axis=1) + (L_d @ c_rhs)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    Q, R_qr = np.linalg.qr(A.T, mode="reduced")
    c_rhs = np.linalg.solve(R_qr.T, b)
    KR_q = (Q[:, :, None] * Q[:, None, :]).reshape(d, m * m)

    L_cur = Y @ Q - c_rhs[None, :]
    E = L_cur.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(14):
        res = np.clip(Y - E @ Q.T, lo[None, :], hi[None, :]) @ Q - c_rhs[None, :]
        Ln = E + res
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Ln - L_cur) * res, axis=1) < 0.0
        be[rs] = 0.0; tn[rs] = 1.0
        E = Ln + be[:, None] * (Ln - L_cur)
        L_cur = Ln; th = tn

    eye_m = 1e-9 * np.eye(m)[None, :, :]
    for _ in range(7):
        U = Y - L_cur @ Q.T
        Z = np.clip(U, lo[None, :], hi[None, :])
        rq = Z @ Q - c_rhs[None, :]
        unc = np.where(np.max(np.abs(rq), axis=1) > 1e-10)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]; ru = rq[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR_q).reshape(len(unc), m, m) + eye_m
        dl = np.linalg.solve(H, ru[:, :, None]).squeeze(-1)
        Lu = L_cur[unc]; Yu = Y[unc]
        p0 = dual_obj_q(Yu, Q, c_rhs, lo, hi, Lu)
        slope = -np.sum(ru * dl, axis=1)
        step = np.ones(len(unc)); done = np.zeros(len(unc), dtype=bool)
        Lb = Lu + ru
        for _ in range(6):
            Lt = Lu + step[:, None] * dl
            pt = dual_obj_q(Yu, Q, c_rhs, lo, hi, Lt)
            ok = (pt <= p0 + 1e-4 * step * slope) & (~done)
            if np.any(ok):
                Lb[ok] = Lt[ok]; done[ok] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        L_cur[unc] = Lb

    return np.clip(Y - L_cur @ Q.T, lo[None, :], hi[None, :])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    ev = np.load(req["eval_npz"])
    n_rep = int(req.get("n_repeats", 12))
    out_npz = Path(req["out_npz_dir"]) / (Path(args.output_json).stem + "_z.npz")
    loaded = [(it["id"], ev[f"{it['id']}_A"], ev[f"{it['id']}_b"], ev[f"{it['id']}_lo"], ev[f"{it['id']}_hi"], ev[f"{it['id']}_Y"]) for it in req["items"]]
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
