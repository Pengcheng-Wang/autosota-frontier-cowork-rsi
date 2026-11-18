import argparse, json
from pathlib import Path
import numpy as np


VARIANT = 'comp_dykstra50'

def solve_qp(A, b, lo, hi, Y):
    m, d = A.shape
    AAt = A @ A.T + 1e-12 * np.eye(m)
    AAt_inv = np.linalg.inv(AAt)
    P_aff = np.eye(d) - A.T @ (AAt_inv @ A)
    shift_aff = A.T @ (AAt_inv @ b)

    if VARIANT in ("paper_dr50", "autosota_dr10", "comp_overrelaxed_dr50"):
        n_iters = 10 if VARIANT == "autosota_dr10" else 50
        omega = 1.65 if VARIANT == "comp_overrelaxed_dr50" else 1.0
        S = Y.copy()
        for _ in range(n_iters):
            Z_box = np.clip(S, lo[None, :], hi[None, :])
            Z_aff = Z_box @ P_aff.T + shift_aff[None, :] + (Z_box - S) @ P_aff.T
            # Exact DR reflection: proj_aff(2*Z_box - S) = (2*Z_box - S) @ P_aff.T + shift_aff
            Z_aff = (2.0 * Z_box - S) @ P_aff.T + shift_aff[None, :]
            S = S + omega * (Z_aff - Z_box)
        return np.clip(S, lo[None, :], hi[None, :])

    if VARIANT == "comp_dykstra50":
        X = Y.copy()
        P_inc = np.zeros_like(Y)
        Q_inc = np.zeros_like(Y)
        for _ in range(50):
            Ym = (X + P_inc) @ P_aff.T + shift_aff[None, :]
            P_inc = X + P_inc - Ym
            X = np.clip(Ym + Q_inc, lo[None, :], hi[None, :])
            Q_inc = Ym + Q_inc - X
        return X

    if VARIANT == "w4_champ_vec_admm":
        rho = 2.2
        alpha = 1.72
        W = np.clip(Y @ P_aff.T + shift_aff[None, :], lo[None, :], hi[None, :])
        U = np.zeros_like(Y)
        inv_1pr = 1.0 / (1.0 + rho)
        for _ in range(110):
            Z = ((Y + rho * (W - U)) * inv_1pr) @ P_aff.T + shift_aff[None, :]
            Z_hat = alpha * Z + (1.0 - alpha) * W
            W_new = np.clip(Z_hat + U, lo[None, :], hi[None, :])
            U = U + Z_hat - W_new
            W = W_new
        return W

    raise ValueError(VARIANT)


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
