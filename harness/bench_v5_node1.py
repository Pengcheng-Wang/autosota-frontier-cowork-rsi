#!/usr/bin/env python3
"""
Wave-5 Frontier-Cowork Governed RSI Harness — Node 1 (8 x A100-80GB):
  - D4: Paper #90 RSTIB (PEMS04 on Wave-4-untouched highway sensors rank 96..191, 12 physical-day clusters/split)
  - D5: Paper #9 ScoreMissing (S&P 500 32%-MAR Precision Graph Recovery, 10 time+stock-disjoint clusters/split, split-isolated GT)
  - D6: Paper #99 OLLALanding (Manifold Sampling <= 400 Grad Calls on fresh hash-disjoint problem instances)
Implements all Opus 5.5 Max Milestone-1 Design Audit requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import average_precision_score, roc_auc_score

from harness_common_v5 import (
    W5_ROOT,
    execute_v5_tournament,
    read_json,
    run_isolated_predict,
    write_json,
)


def StandardScaler_np(X: np.ndarray) -> np.ndarray:
    m = np.mean(X, axis=0, keepdims=True)
    s = np.std(X, axis=0, keepdims=True)
    s = np.where(s > 1e-8, s, 1.0)
    return (X - m) / s


# ============================================================================
# D4: Paper #90 RSTIB — PEMS04 on Wave-4-Untouched Sensors [rank 96..191]
# ============================================================================
def prepare_and_score_d4(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W5_ROOT / "tracks" / "bench_d4_pems04_rstib"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"
    L_IN, H_OUT, N_SENS = 12, 12, 96

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        raw = np.load(W5_ROOT / "data" / "pems04" / "PEMS04.npz")["data"].astype(np.float32)

        # Select sensors ranked 96..191 from training prefix [0:8640] -> 0% overlap with Wave-4's rank 0..95!
        stds = np.std(raw[:8640, :, 0], axis=0)
        sens_idx = np.sort(np.argsort(-stds)[96:192])
        data = raw[:, sens_idx, :]  # [16992, 96, 3]

        # Standardize strictly using Train_v5 [0:8640] (first 30 physical days)
        tr_raw = data[:8640]
        ch_mean = np.mean(tr_raw, axis=(0, 1), keepdims=True)
        ch_std = np.std(tr_raw, axis=(0, 1), keepdims=True) + 1e-5
        data_norm = (data - ch_mean) / ch_std
        flow_mean = float(ch_mean[0, 0, 0])
        flow_std = float(ch_std[0, 0, 0])

        tr_starts = np.arange(0, 8640 - L_IN - H_OUT, 4)
        X_tr = np.stack([data_norm[s:s + L_IN] for s in tr_starts], axis=0)
        Y_tr = np.stack([data_norm[s + L_IN:s + L_IN + H_OUT, :, 0] for s in tr_starts], axis=0)
        np.savez_compressed(runs / "pems04_train.npz", X_train=X_tr, Y_train=Y_tr)

        # 3 non-overlapping multi-day horizons, each with 12 physical-day clusters (gap 32 > L+H=24!)
        split_ranges = {
            "search": 8700,
            "confirm": 11460,
            "holdout": 14220,
        }
        for st, t_start in split_ranges.items():
            rng = np.random.RandomState({"search": 5401, "confirm": 5402, "holdout": 5403}[st])
            X_blocks = []
            gt_blocks = []
            ids_list = []
            for b in range(12):
                b0 = t_start + b * 225
                b1 = b0 + 193  # 32-step buffer between day-clusters -> zero target/lookback overlap!
                starts = np.arange(b0, b1 - L_IN - H_OUT, 2)
                Xb_clean = np.stack([data_norm[s:s + L_IN] for s in starts], axis=0)
                Yb_true_raw = np.stack([data[s + L_IN:s + L_IN + H_OUT, :, 0] for s in starts], axis=0)

                Xb_noisy = Xb_clean.copy()
                drop_mask = rng.rand(len(Xb_noisy), 1, N_SENS, 1) < 0.12
                Xb_noisy = np.where(drop_mask, 0.0, Xb_noisy)
                spike_mask = rng.rand(*Xb_noisy.shape) < 0.03
                Xb_noisy = Xb_noisy + spike_mask * rng.normal(0.0, 4.5, size=Xb_noisy.shape).astype(np.float32)

                bid = f"v5_pems04_{st}_day{b:02d}"
                X_blocks.append(Xb_noisy)
                gt_blocks.append({"id": bid, "Y_true_raw": Yb_true_raw.tolist()})
                ids_list.append(bid)

            eval_npz_p = runs / f"pems04_eval_{st}.npz"
            np.savez_compressed(eval_npz_p, X_blocks=np.stack(X_blocks, axis=0), flow_mean=flow_mean, flow_std=flow_std)
            write_json(runs / f"input_{st}.json", {
                "train_npz": str(runs / "pems04_train.npz"),
                "eval_npz": str(eval_npz_p),
                "items": [{"id": bid, "block_idx": i} for i, bid in enumerate(ids_list)],
            })
            write_json(imm / f"gt_{st}.json", gt_blocks)
            write_json(imm / f"ids_{st}.json", ids_list)

    def score_d4(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=300)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.float32) for r in read_json(out_p)["predictions"]}
        gt_blocks = read_json(imm / f"gt_{stage}.json")

        items = []
        for blk in gt_blocks:
            bid = blk["id"]
            yt = np.asarray(blk["Y_true_raw"], dtype=np.float32)
            yp = preds_map[bid]
            mae = float(np.mean(np.abs(yp - yt)))
            items.append({
                "id": bid,
                "cluster_id": bid,
                "stratum_id": "pems04_untouched96",
                "value": mae,
            })
        return {
            "candidate_id": cid,
            "metrics": {
                "traffic_mae": float(np.mean([it["value"] for it in items])),
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "traffic_mae",
        "direction": "minimize",
        "decision_stage": "search",
        "min_improvement": 0.15,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.8},
    }
    rules_confirm = {
        "primary_metric": "traffic_mae",
        "direction": "minimize",
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.15,
        "min_confirm_clusters": 12,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 1.8},
        "required_comparators": [
            "d4_paper_rstib_ns12",
            "d4_base_huber_ridge",
            "d4_comp_spatial_mlp",
        ],
    }
    return execute_v5_tournament(
        bench_id="d4",
        bdir=bdir,
        score_fn=score_d4,
        baseline_id="d4_paper_rstib_ns12",
        comparators=["d4_paper_rstib_ns12", "d4_base_huber_ridge", "d4_comp_spatial_mlp"],
        autosota_audit_id="d4_autosota_rstib_ns50_zeroib",
        w4_champ_id="d4_w4_champ_kriging_svd",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D5: Paper #9 ScoreMissing — Split-Isolated GT + 10 Time+Stock-Disjoint Clusters
# ============================================================================
def prepare_and_score_d5(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W5_ROOT / "tracks" / "bench_d5_snp500_scoremissing"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(W5_ROOT / "data" / "snp500" / "snp_500_allstocks_5yr.csv")
        piv = df.pivot(index="date", columns="Name", values="close").sort_index().dropna(axis=1, how="any")
        rets = np.log(piv.values[1:] / piv.values[:-1]).astype(np.float64)  # [1258, 470]
        n_days, n_stocks = rets.shape

        rng = np.random.RandomState(50905)
        perm_stocks = rng.permutation(n_stocks)
        p_dim = 26  # 10 disjoint 26-stock universes = 260 stocks; 3 splits draw disjoint calendar eras!

        for st_idx, st in enumerate(["search", "confirm", "holdout"]):
            # Strictly non-overlapping 412-day eras (10-day buffer gap between eras)
            t_lo, t_hi = [(0, 412), (422, 834), (844, 1256)][st_idx]
            R_era = rets[t_lo:t_hi]  # [412, 470]
            # Per-split independent stock permutation -> 10 strictly disjoint 26-stock subsets!
            st_perm = rng.permutation(n_stocks)
            items_in = []
            items_gt = []

            for b in range(10):
                cols = st_perm[b * p_dim:(b + 1) * p_dim]
                # Per Opus 5.5 Max Audit: fit reference GraphicalLasso STRICTLY inside this split's own era (NEVER full 5-yr history!)
                X_era_clean = StandardScaler_np(R_era[:, cols])
                gl = GraphicalLasso(alpha=0.14, max_iter=250, tol=1e-4).fit(X_era_clean)
                prec_true = gl.precision_
                iu = np.triu_indices(p_dim, k=1)
                pcorr_true = np.abs(prec_true[iu]) / np.sqrt(prec_true[iu[0], iu[0]] * prec_true[iu[1], iu[1]] + 1e-12)
                thresh = max(0.035, float(np.quantile(pcorr_true, 0.84)))
                edges_gt = (pcorr_true >= thresh).astype(np.int64)

                # Strictly non-overlapping 40-day sub-window per cluster b (1-day gap between consecutive clusters!)
                # => clusters 0..9 share ZERO trading days AND ZERO stocks!
                w0 = b * 41
                w1 = w0 + 40
                X_win = StandardScaler_np(R_era[w0:w1, cols])
                rng_b = np.random.RandomState(15000 + st_idx * 100 + b)
                mkt_vol = np.abs(np.mean(X_win, axis=1, keepdims=True))
                miss_prob = np.clip(0.22 + 0.12 * mkt_vol, 0.10, 0.65)
                mask = (rng_b.rand(*X_win.shape) >= miss_prob).astype(np.float64)
                X_obs = np.where(mask > 0.5, X_win, 0.0)

                bid = f"v5_snp_{st}_reg{b:02d}"
                items_in.append({
                    "id": bid,
                    "p": p_dim,
                    "X_obs": np.round(X_obs, 6).tolist(),
                    "mask": mask.astype(int).tolist(),
                })
                items_gt.append({
                    "id": bid,
                    "edges_gt": edges_gt.tolist(),
                })

            write_json(runs / f"input_{st}.json", {"items": items_in})
            write_json(imm / f"gt_{st}.json", items_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in items_gt])

    def score_d5(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=240)
        preds_map = {r["id"]: np.asarray(r["prediction"], dtype=np.float64) for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        items = []
        aurocs = []
        for g in gt_list:
            bid = g["id"]
            y_true = np.asarray(g["edges_gt"], dtype=np.int64)
            scores = preds_map[bid]
            ap = float(average_precision_score(y_true, scores))
            auroc = float(roc_auc_score(y_true, scores))
            aurocs.append(auroc)
            items.append({
                "id": bid,
                "cluster_id": bid,
                "stratum_id": "snp500_isolated_ggm",
                "value": ap,
                "aux": {"auroc": auroc},
            })

        return {
            "candidate_id": cid,
            "metrics": {
                "edge_auprc": float(np.mean([it["value"] for it in items])),
                "edge_auroc": float(np.mean(aurocs)),
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "edge_auprc",
        "direction": "maximize",
        "decision_stage": "search",
        "min_improvement": 0.008,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
    }
    rules_confirm = {
        "primary_metric": "edge_auprc",
        "direction": "maximize",
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.008,
        "min_confirm_clusters": 10,
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.5},
        "required_comparators": [
            "d5_paper_msm_score_matching",
            "d5_base_mean_impute_ridge",
            "d5_comp_pairwise_cov_shrink",
        ],
    }
    return execute_v5_tournament(
        bench_id="d5",
        bdir=bdir,
        score_fn=score_d5,
        baseline_id="d5_paper_msm_score_matching",
        comparators=["d5_paper_msm_score_matching", "d5_base_mean_impute_ridge", "d5_comp_pairwise_cov_shrink"],
        autosota_audit_id="d5_autosota_msm_5x_ens",
        w4_champ_id="d5_w4_champ_multi_imp_stab",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


# ============================================================================
# D6: Paper #99 OLLALanding — Fresh Hash-Disjoint Manifold MCMC (<= 400 Steps)
# ============================================================================
MANIFOLD_LIB_CODE = '''import math
import numpy as np

class BudgetedManifoldProblem:
    """Deterministic equality-constrained manifold target {x in R^d : h(x) = 0} with call counter."""
    def __init__(self, spec: dict, hard_cap: int = 100000):
        self.kind = spec["kind"]
        self.dim = int(spec["dim"])
        self.params = spec["params"]
        self.hard_cap = int(hard_cap)
        self.grad_calls = 0

    def eval_all(self, X: np.ndarray):
        self.grad_calls += 1
        if self.grad_calls > self.hard_cap:
            raise RuntimeError(f"Manifold oracle call cap exceeded: {self.grad_calls} > {self.hard_cap}")
        M, d = X.shape
        if self.kind == "sphere_mog":
            centers = np.asarray(self.params["centers"], dtype=np.float64)
            inv_s2 = float(self.params["inv_s2"])
            rad2 = float(self.params["rad2"])
            diff = X[:, None, :] - centers[None, :, :]
            lw = -0.5 * inv_s2 * np.sum(diff ** 2, axis=2)
            lw_max = np.max(lw, axis=1, keepdims=True)
            w = np.exp(lw - lw_max)
            Z = np.sum(w, axis=1, keepdims=True)
            probs = w / Z
            U = -(lw_max[:, 0] + np.log(Z[:, 0]))
            grad_U = inv_s2 * np.sum(probs[:, :, None] * diff, axis=1)
            scales = np.asarray(self.params["scales"], dtype=np.float64)
            h = (np.sum(scales[None, :] * (X ** 2), axis=1) - rad2)[:, None]
            J = (2.0 * scales[None, :] * X)[:, None, :]
            return U, grad_U, h, J
        elif self.kind == "stiefel_2frame":
            p = d // 2
            u, v = X[:, :p], X[:, p:]
            A1 = np.asarray(self.params["A1"], dtype=np.float64)
            A2 = np.asarray(self.params["A2"], dtype=np.float64)
            b1 = np.asarray(self.params["b1"], dtype=np.float64)
            Au = u @ A1
            Av = v @ A2
            U = -0.5 * np.sum(u * Au, axis=1) - 0.5 * np.sum(v * Av, axis=1) - np.sum(u * b1[None, :], axis=1) + 0.25 * (np.sum(u * b1[None, :], axis=1) ** 2)
            proj_b = np.sum(u * b1[None, :], axis=1, keepdims=True)
            gu = -Au - b1[None, :] + 0.5 * proj_b * b1[None, :]
            gv = -Av
            grad_U = np.hstack([gu, gv])
            h1 = np.sum(u * u, axis=1) - 1.0
            h2 = np.sum(v * v, axis=1) - 1.0
            h3 = np.sum(u * v, axis=1)
            h = np.stack([h1, h2, h3], axis=1)
            J = np.zeros((M, 3, d), dtype=np.float64)
            J[:, 0, :p] = 2.0 * u
            J[:, 1, p:] = 2.0 * v
            J[:, 2, :p] = v
            J[:, 2, p:] = u
            return U, grad_U, h, J
        elif self.kind == "fair_bayes_logit":
            Xd = np.asarray(self.params["X"], dtype=np.float64)
            yd = np.asarray(self.params["y"], dtype=np.float64)
            c1 = np.asarray(self.params["c1"], dtype=np.float64)
            c2 = np.asarray(self.params["c2"], dtype=np.float64)
            logits = np.clip(X @ Xd.T, -25.0, 25.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            nll = -np.sum(yd[None, :] * logits - np.log1p(np.exp(logits)), axis=1)
            U = nll + 0.5 * 0.5 * np.sum(X ** 2, axis=1)
            grad_U = (probs - yd[None, :]) @ Xd + 0.5 * X
            rad2 = float(self.params["rad2"])
            h1 = X @ c1
            h2 = np.sum(c2[None, :] * (X ** 2), axis=1) - rad2
            h = np.stack([h1, h2], axis=1)
            J = np.zeros((M, 2, d), dtype=np.float64)
            J[:, 0, :] = c1[None, :]
            J[:, 1, :] = 2.0 * c2[None, :] * X
            return U, grad_U, h, J
        else:
            raise ValueError(self.kind)
'''


def energy_distance_1d_sliced(X: np.ndarray, Y: np.ndarray, dirs: np.ndarray) -> float:
    px = np.sort(X @ dirs.T, axis=0)
    py = np.sort(Y @ dirs.T, axis=0)
    if len(px) != len(py):
        q = np.linspace(0.0, 1.0, len(px))
        py = np.quantile(Y @ dirs.T, q, axis=0)
    return float(np.mean(np.abs(px - py)))


def generate_gold_manifold_samples(spec: dict, seed: int, n_particles: int = 256) -> np.ndarray:
    ns = {}
    exec(MANIFOLD_LIB_CODE, ns)
    prob = ns["BudgetedManifoldProblem"](spec, hard_cap=500000)
    rng = np.random.RandomState(seed)
    d = prob.dim
    X = rng.normal(0.0, 1.0, size=(n_particles, d))

    def newton_project(Z: np.ndarray, max_it: int = 4) -> np.ndarray:
        Zc = Z.copy()
        for _ in range(max_it):
            _, _, h, J = prob.eval_all(Zc)
            if np.max(np.abs(h)) < 1e-10:
                break
            k = J.shape[1]
            JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-10 * np.eye(k)[None, :, :]
            c = np.linalg.solve(JJt, h[:, :, None]).squeeze(-1)
            Zc = Zc - np.einsum("mk,mkd->md", c, J)
        return Zc

    X = newton_project(X, max_it=10)
    dt = 0.010
    for _ in range(1800):
        _, gU, _, J = prob.eval_all(X)
        k = J.shape[1]
        raw_v = -0.5 * dt * gU + math.sqrt(dt) * rng.normal(0.0, 1.0, size=X.shape)
        JJt = np.einsum("mkd,mld->mkl", J, J) + 1e-10 * np.eye(k)[None, :, :]
        rhs = np.einsum("mkd,md->mk", J, raw_v)
        c = np.linalg.solve(JJt, rhs[:, :, None]).squeeze(-1)
        v_tan = raw_v - np.einsum("mk,mkd->md", c, J)
        X = newton_project(X + v_tan, max_it=3)
    return X


def prepare_and_score_d6(phase: str, r1_map: dict[str, str], r2_map: dict[str, str]) -> dict:
    bdir = W5_ROOT / "tracks" / "bench_d6_ollalanding"
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"

    if phase == "round1":
        imm.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "manifold_lib.py").write_text(MANIFOLD_LIB_CODE, encoding="utf-8")

        seen_spec_hashes = set()
        for st_idx, st in enumerate(["search", "confirm", "holdout"]):
            specs_in = []
            specs_gt = []
            for fam_idx, fam in enumerate(["sphere_mog", "stiefel_2frame", "fair_bayes_logit"]):
                for k in range(4):
                    rng = np.random.RandomState(29900 + st_idx * 131 + fam_idx * 19 + k)
                    cid_base = f"v5_{fam}_{k:02d}"
                    if fam == "sphere_mog":
                        dim = 8 if k % 2 == 0 else 12
                        scales = np.exp(rng.uniform(-0.6, 0.6, size=dim))
                        raw_c = rng.normal(0.0, 1.0, size=(4, dim))
                        rad2 = float(dim * 0.5)
                        centers = raw_c * np.sqrt(rad2 / np.sum(scales[None, :] * raw_c ** 2, axis=1, keepdims=True))
                        spec = {
                            "kind": "sphere_mog",
                            "dim": dim,
                            "params": {"centers": centers.tolist(), "scales": scales.tolist(), "inv_s2": 3.8, "rad2": rad2},
                        }
                    elif fam == "stiefel_2frame":
                        dim = 10 if k % 2 == 0 else 14
                        p = dim // 2
                        M1, M2 = rng.normal(0, 1, (p, p)), rng.normal(0, 1, (p, p))
                        A1 = ((M1 + M1.T) * 0.95).tolist()
                        A2 = ((M2 + M2.T) * 0.95).tolist()
                        b1 = rng.normal(0, 1.6, size=p).tolist()
                        spec = {"kind": "stiefel_2frame", "dim": dim, "params": {"A1": A1, "A2": A2, "b1": b1}}
                    else:
                        dim = 8
                        N_pts = 120
                        Xd = rng.normal(0, 1.0, size=(N_pts, dim))
                        w_true = rng.normal(0, 1.25, size=dim)
                        yd = (rng.rand(N_pts) < (1.0 / (1.0 + np.exp(-(Xd @ w_true))))).astype(float)
                        c1 = rng.normal(0, 1.0, size=dim)
                        c1 /= np.linalg.norm(c1)
                        c2 = np.exp(rng.uniform(-0.4, 0.4, size=dim))
                        spec = {
                            "kind": "fair_bayes_logit",
                            "dim": dim,
                            "params": {"X": Xd.tolist(), "y": yd.tolist(), "c1": c1.tolist(), "c2": c2.tolist(), "rad2": 4.0},
                        }

                    sh = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                    assert sh not in seen_spec_hashes
                    seen_spec_hashes.add(sh)

                    gold = generate_gold_manifold_samples(spec, seed=27000 + st_idx * 73 + fam_idx * 17 + k, n_particles=256)
                    dirs = rng.normal(0, 1.0, size=(64, spec["dim"]))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

                    seeds = [0, 1] if st == "search" else [10, 11, 12]
                    for sd in seeds:
                        rng_s = np.random.RandomState(13000 + sd * 97 + fam_idx * 13 + k)
                        X_init = rng_s.normal(0.0, 1.0, size=(96, spec["dim"]))
                        iid = f"v5_{st}_{cid_base}_s{sd}"
                        specs_in.append({
                            "id": iid,
                            "seed": sd,
                            "max_steps": 400,
                            "spec": spec,
                            "X_init": np.round(X_init, 6).tolist(),
                        })
                        specs_gt.append({
                            "id": iid,
                            "cluster_id": f"{st}_{cid_base}",
                            "stratum_id": fam,
                            "seed_id": f"s{sd}",
                            "spec": spec,
                            "gold_samples": gold.tolist(),
                            "proj_dirs": dirs.tolist(),
                        })

            write_json(runs / f"input_{st}.json", {
                "manifold_lib_path": str(runs / "manifold_lib.py"),
                "items": specs_in,
            })
            write_json(imm / f"gt_{st}.json", specs_gt)
            write_json(imm / f"ids_{st}.json", [x["id"] for x in specs_gt])

    def score_d6(cid: str, stage: str) -> dict:
        c_py = bdir / "candidates" / f"{cid}.py"
        in_p = runs / f"input_{stage}.json"
        out_p = runs / f"preds_{stage}_{cid}.json"
        ids_p = imm / f"ids_{stage}.json"
        iso = run_isolated_predict(c_py, in_p, out_p, ids_p, imm, timeout_sec=360)
        preds_map = {r["id"]: r["prediction"] for r in read_json(out_p)["predictions"]}
        gt_list = read_json(imm / f"gt_{stage}.json")

        ns = {}
        exec(MANIFOLD_LIB_CODE, ns)
        items = []
        max_calls_all = 0.0
        max_err_all = 0.0
        for g in gt_list:
            iid = g["id"]
            pred_obj = preds_map[iid]
            X_fin = np.asarray(pred_obj["particles"], dtype=np.float64)
            calls_reported = float(pred_obj["grad_calls"])
            prob = ns["BudgetedManifoldProblem"](g["spec"])
            _, _, h_fin, _ = prob.eval_all(X_fin)
            m_err = float(np.max(np.abs(h_fin)))
            max_calls_all = max(max_calls_all, calls_reported)
            max_err_all = max(max_err_all, m_err)

            gold = np.asarray(g["gold_samples"], dtype=np.float64)
            dirs = np.asarray(g["proj_dirs"], dtype=np.float64)
            w1_dist = energy_distance_1d_sliced(X_fin, gold, dirs)
            items.append({
                "id": iid,
                "cluster_id": g["cluster_id"],
                "stratum_id": g["stratum_id"],
                "seed_id": g["seed_id"],
                "value": w1_dist,
                "aux": {"grad_calls": calls_reported, "manifold_err": m_err},
            })

        by_st = {}
        for it in items:
            by_st.setdefault(it["stratum_id"], []).append(it["value"])
        eq_w1 = float(np.mean([np.mean(v) for v in by_st.values()]))
        return {
            "candidate_id": cid,
            "metrics": {
                "sliced_w1_dist": eq_w1,
                "max_grad_calls": max_calls_all,
                "max_manifold_err": max_err_all,
                "eval_wall_clock_sec": float(iso["guard_wall_clock_sec"]),
            },
            "items": items,
        }

    rules_search = {
        "primary_metric": "sliced_w1_dist",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "decision_stage": "search",
        "min_improvement": 0.004,
        "hard_ceilings": {"max_grad_calls": 400.0, "max_manifold_err": 1e-3},
        "per_stratum_hard_ceilings": {"grad_calls": 400.0, "manifold_err": 1e-3},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
    }
    rules_confirm = {
        "primary_metric": "sliced_w1_dist",
        "direction": "minimize",
        "aggregation_weighting": "equal_stratum",
        "expected_strata": ["fair_bayes_logit", "sphere_mog", "stiefel_2frame"],
        "bootstrap_mode": "crossed_seed_cluster",
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "max_confirm_looks": 2,
        "min_improvement": 0.004,
        "min_confirm_clusters": 12,
        "hard_ceilings": {"max_grad_calls": 400.0, "max_manifold_err": 1e-3},
        "per_stratum_hard_ceilings": {"grad_calls": 400.0, "manifold_err": 1e-3},
        "max_ratio_vs_baseline": {"eval_wall_clock_sec": 2.2},
        "required_comparators": [
            "d6_paper_olla_400",
            "d6_comp_clangevin_newton_400",
            "d6_comp_rattle_hmc_400",
        ],
    }
    return execute_v5_tournament(
        bench_id="d6",
        bdir=bdir,
        score_fn=score_d6,
        baseline_id="d6_paper_olla_400",
        comparators=["d6_paper_olla_400", "d6_comp_clangevin_newton_400", "d6_comp_rattle_hmc_400"],
        autosota_audit_id="d6_autosota_olla_2000steps_cheat",
        w4_champ_id="d6_w4_champ_stein_equil",
        round1_candidates=r1_map,
        round2_candidates=r2_map,
        rules_search=rules_search,
        rules_confirm=rules_confirm,
        phase=phase,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["round1", "round2_final"], required=True)
    ap.add_argument("--spec-json", required=True)
    args = ap.parse_args()
    spec = read_json(Path(args.spec_json))
    for bid, fn in [("d4", prepare_and_score_d4), ("d5", prepare_and_score_d5), ("d6", prepare_and_score_d6)]:
        if bid in spec:
            print(f"=== [Node-1 V5 {args.phase}] Running {bid.upper()} ===", flush=True)
            res = fn(args.phase, spec[bid]["r1"], spec[bid].get("r2", {}))
            print(f"=== [Node-1 V5 {args.phase}] Done {bid.upper()}: {res.get('round1_incumbent_id') or res.get('selected_finalist')} ===", flush=True)


if __name__ == "__main__":
    main()
