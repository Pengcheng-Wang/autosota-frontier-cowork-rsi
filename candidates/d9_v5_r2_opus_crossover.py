import argparse, json
from pathlib import Path
import numpy as np

def dual_obj(Y, A, b, lo, hi, N):
    U = Y + N @ A
    Z = np.clip(U, lo[None, :], hi[None, :])
    return np.sum(0.5 * (U ** 2) - 0.5 * ((U - Z) ** 2), axis=1) - (N @ b)

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    G = A @ A.T + 1e-12 * np.eye(m)
    Gi = np.linalg.inv(G)
    KR = (A[:, None, :] * A[None, :, :]).reshape(m * m, d).T.copy()

    N = (b[None, :] - Y @ A.T) @ Gi
    E = N.copy()
    th = np.ones(len(Y), dtype=np.float64)
    for _ in range(15):
        R = b[None, :] - np.clip(Y + E @ A, lo[None, :], hi[None, :]) @ A.T
        Nn = E + R @ Gi
        tn = 0.5 + np.sqrt(0.25 + th ** 2)
        be = (th - 1.0) / tn
        rs = np.sum((Nn - N) * R, axis=1) < 0.0
        be[rs] = 0.0; tn[rs] = 1.0
        E = Nn + be[:, None] * (Nn - N)
        N = Nn; th = tn

    eye_m = 1e-10 * np.eye(m)[None, :, :]
    for _ in range(8):
        U = Y + N @ A
        Z = np.clip(U, lo[None, :], hi[None, :])
        R = b[None, :] - Z @ A.T
        unc = np.where(np.max(np.abs(R), axis=1) > 1e-10)[0]
        if len(unc) == 0:
            break
        Uu = U[unc]; Ru = R[unc]
        free = ((Uu > lo[None, :] + 1e-13) & (Uu < hi[None, :] - 1e-13)).astype(np.float64)
        H = (free @ KR).reshape(len(unc), m, m) + eye_m
        dnu = np.linalg.solve(H, Ru[:, :, None]).squeeze(-1)
        Nu = N[unc]; Yu = Y[unc]
        p0 = dual_obj(Yu, A, b, lo, hi, Nu)
        slope = -np.sum(Ru * dnu, axis=1)
        step = np.ones(len(unc)); done = np.zeros(len(unc), dtype=bool)
        Nb = Nu + Ru @ Gi
        for _ in range(7):
            Nt = Nu + step[:, None] * dnu
            pt = dual_obj(Yu, A, b, lo, hi, Nt)
            ok = (pt <= p0 + 1e-4 * step * slope) & (~done)
            if np.any(ok):
                Nb[ok] = Nt[ok]; done[ok] = True
            if np.all(done):
                break
            step[~done] *= 0.5
        N[unc] = Nb

    return np.clip(Y + N @ A, lo[None, :], hi[None, :])

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
