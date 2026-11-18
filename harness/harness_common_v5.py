#!/usr/bin/env python3
"""Shared Governed Loop Guard v3.6 orchestration helpers for Wave-5 Frontier-Cowork RSI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

W5_ROOT = Path("/scratch/autosota_governed_v5")
W4_FORBID = Path("/scratch/autosota_governed_v4")
GUARD = W5_ROOT / "guard" / "governed_loop_guard.py"


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_guard(*args: str, allow_fail: bool = False) -> tuple[int, dict[str, Any]]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    p = subprocess.run(
        [sys.executable, str(GUARD), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    out = (p.stdout or "").strip()
    data: dict[str, Any] = {}
    if out:
        try:
            data = json.loads(out)
        except Exception:
            data = {"raw_stdout": out, "raw_stderr": (p.stderr or "")[:500]}
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Guard failed ({p.returncode}) on {args[:4]}: {out[:600]} {(p.stderr or '')[:400]}")
    return p.returncode, data


def run_isolated_predict(
    candidate_py: Path,
    input_json: Path,
    output_json: Path,
    expected_ids_json: Path,
    forbid_dir: Path,
    timeout_sec: int = 360,
) -> dict[str, Any]:
    run_guard("audit-leaks", "--paths", str(candidate_py))
    _, iso = run_guard(
        "run-isolated",
        "--mode", "predict",
        "--candidate-script", str(candidate_py),
        "--input-json", str(input_json),
        "--output-json", str(output_json),
        "--expected-ids-json", str(expected_ids_json),
        "--forbid-paths", str(forbid_dir), str(W4_FORBID),
        "--timeout-sec", str(timeout_sec),
    )
    return iso


def execute_v5_tournament(
    bench_id: str,
    bdir: Path,
    score_fn,
    baseline_id: str,
    comparators: list[str],
    autosota_audit_id: str | None,
    w4_champ_id: str,
    round1_candidates: dict[str, str],  # {cand_id: cowork_model_tag}
    round2_candidates: dict[str, str],  # {cand_id: cowork_model_tag}
    rules_search: dict[str, Any],
    rules_confirm: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    """
    Wave-5 Governed Tournament implementing Opus 5.5 Max Milestone-1 Design Audit:
      - Keeps `required_comparators` identical to Wave-4 for clean benchmark promotion gating.
      - Runs `w4_champ_id` (frozen Wave-4 Headless Champion refit only on Train_v5) as a parallel
        pre-registered arm on Search_v5, Confirm_v5, and (when unsealed) Holdout_v5.
      - Gives the primary Confirm_v5 promotion slot strictly to the best Wave-5 Cowork candidate.
    """
    imm = bdir / "immutable_eval"
    runs = bdir / "runs"
    cdir = bdir / "candidates"
    mf_path = bdir / "IMMUTABLE_MANIFEST.json"
    pin_path = W5_ROOT / "seals" / f"PIN_v5_{bench_id}.txt"
    ledger_path = bdir / "trial_ledger.jsonl"
    rs_path = imm / "gate_rules_search.json"
    rc_path = imm / "gate_rules_confirm.json"
    r1_state_path = runs / "round1_state.json"

    if phase == "round1":
        write_json(rs_path, rules_search)
        write_json(rc_path, rules_confirm)
        _, mf_info = run_guard("freeze-manifest", "--manifest", str(mf_path), "--root", str(bdir), "--paths", "immutable_eval")
        root_sha = mf_info["manifest_root_sha256"]
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        pin_path.write_text(root_sha, encoding="utf-8")
        if ledger_path.exists():
            ledger_path.unlink()

        base_res = score_fn(baseline_id, "search")
        base_p = write_json(runs / f"search_{baseline_id}.json", base_res)

        autosota_audit = None
        if autosota_audit_id:
            auto_res = score_fn(autosota_audit_id, "search")
            auto_p = write_json(runs / f"search_{autosota_audit_id}.json", auto_res)
            _, auto_dec = run_guard(
                "gate-decision",
                "--candidate", str(auto_p),
                "--incumbent", str(base_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--trial-id", f"{bench_id}_v5_audit_{autosota_audit_id}",
                allow_fail=True,
            )
            autosota_audit = {
                "autosota_id": autosota_audit_id,
                "baseline_metrics": base_res["metrics"],
                "autosota_metrics": auto_res["metrics"],
                "decision": auto_dec["decision"],
                "tripwires": auto_dec["tripwires"],
            }

        # Evaluate frozen Wave-4 Headless Champion in parallel on Search_v5 (redacted reference arm)
        w4_res = score_fn(w4_champ_id, "search")
        w4_p = write_json(runs / f"search_{w4_champ_id}.json", w4_res)
        _, w4_dec = run_guard(
            "gate-decision",
            "--candidate", str(w4_p),
            "--incumbent", str(base_p),
            "--baseline", str(base_p),
            "--rules", str(rs_path),
            "--trial-id", f"{bench_id}_v5_ref_{w4_champ_id}",
            allow_fail=True,
        )
        w4_search_ref = {
            "w4_champ_id": w4_champ_id,
            "metrics": w4_res["metrics"],
            "decision_vs_baseline": w4_dec["decision"],
            "tripwires": w4_dec["tripwires"],
        }

        inc_p = base_p
        inc_id = baseline_id
        r1_board = {}
        for idx, (cid, agent_tag) in enumerate(round1_candidates.items(), start=1):
            c_py = cdir / f"{cid}.py"
            res = score_fn(cid, "search")
            cp = write_json(runs / f"search_{cid}.json", res)
            redacted_p = runs / f"proposer_r1_{cid}.json"
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cp),
                "--incumbent", str(inc_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_v5_r1_{idx:02d}_{cid}",
                "--candidate-artifacts", str(c_py),
                "--proposer-out", str(redacted_p),
                "--hypothesis", f"Wave-5 Blind Round-1 proposal from {agent_tag}",
                allow_fail=True,
            )
            r1_board[cid] = {
                "proposer_agent": agent_tag,
                "metrics": res["metrics"],
                "decision_vs_running_incumbent": dec["decision"],
                "tripwires": dec["tripwires"],
                "redacted_feedback": read_json(redacted_p),
            }
            if dec["decision"] == "PROVISIONAL_KEEP":
                inc_p = cp
                inc_id = cid

        state = {
            "bench_id": bench_id,
            "manifest_root_sha256": root_sha,
            "baseline_search": base_res["metrics"],
            "autosota_audit": autosota_audit,
            "w4_headless_champ_search": w4_search_ref,
            "round1_leaderboard": r1_board,
            "round1_incumbent_id": inc_id,
        }
        write_json(r1_state_path, state)
        write_json(W5_ROOT / "results" / f"r1_v5_{bench_id}.json", state)
        return state

    elif phase == "round2_final":
        root_sha = pin_path.read_text(encoding="utf-8").strip()
        run_guard("verify-manifest", "--manifest", str(mf_path), "--expected-root-sha256", root_sha)
        r1_state = read_json(r1_state_path)
        base_p = runs / f"search_{baseline_id}.json"
        inc_id = r1_state["round1_incumbent_id"]
        inc_p = runs / f"search_{inc_id}.json"

        r2_board = {}
        for idx, (cid, agent_tag) in enumerate(round2_candidates.items(), start=1):
            c_py = cdir / f"{cid}.py"
            res = score_fn(cid, "search")
            cp = write_json(runs / f"search_{cid}.json", res)
            redacted_p = runs / f"proposer_r2_{cid}.json"
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cp),
                "--incumbent", str(inc_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_v5_r2_{idx:02d}_{cid}",
                "--candidate-artifacts", str(c_py),
                "--proposer-out", str(redacted_p),
                "--hypothesis", f"Wave-5 Redacted-Search Round-2 crossover proposal from {agent_tag}",
                allow_fail=True,
            )
            r2_board[cid] = {
                "proposer_agent": agent_tag,
                "metrics": res["metrics"],
                "decision_vs_running_incumbent": dec["decision"],
                "tripwires": dec["tripwires"],
            }
            if dec["decision"] == "PROVISIONAL_KEEP":
                inc_p = cp
                inc_id = cid

        # Order-invariant Search winner selection across all tripwire-clean Cowork R1+R2 candidates beating baseline by >= min_improvement
        pm = rules_search["primary_metric"]
        maximize = (rules_search["direction"] == "maximize")
        min_imp = float(rules_search.get("min_improvement", 0.0))
        base_val = float(r1_state["baseline_search"][pm])

        all_boards = {**r1_state["round1_leaderboard"], **r2_board}
        eligible = []
        for cid_k, b_info in all_boards.items():
            # Check candidate against baseline directly via gate-decision (without ledger append) to verify 0 hard/ratio tripwires!
            cp_k = runs / f"search_{cid_k}.json"
            _, b_dec = run_guard(
                "gate-decision",
                "--candidate", str(cp_k),
                "--incumbent", str(base_p),
                "--baseline", str(base_p),
                "--rules", str(rs_path),
                "--trial-id", f"{bench_id}_v5_elig_{cid_k}",
                allow_fail=True,
            )
            if b_dec.get("decision") == "PROVISIONAL_KEEP" and len(b_dec.get("tripwires", [])) == 0:
                val_k = float(b_info["metrics"][pm])
                gain_k = (val_k - base_val) if maximize else (base_val - val_k)
                if gain_k >= min_imp:
                    wall_k = float(b_info["metrics"].get("eval_wall_clock_sec", 1e9))
                    eligible.append((gain_k, -wall_k, cid_k))

        if not eligible:
            summary = {
                "benchmark": bench_id,
                "manifest_root_sha256": root_sha,
                "baseline_search": r1_state["baseline_search"],
                "autosota_audit": r1_state["autosota_audit"],
                "w4_headless_champ_search": r1_state["w4_headless_champ_search"],
                "round1_leaderboard": r1_state["round1_leaderboard"],
                "round2_leaderboard": r2_board,
                "selected_finalist": None,
                "status": "NO_SEARCH_GAIN_HALTED_BEFORE_CONFIRM",
                "confirm_verdicts": {},
                "head_to_head_confirm_vs_w4_champ": None,
                "holdout_reserve": {"status": "NOT_ATTEMPTED_NO_SEARCH_WIN"},
                "holdout_finalized": None,
                "head_to_head_holdout_vs_w4_champ": None,
                "ledger_verification": run_guard("verify-ledger", "--ledger", str(ledger_path))[1],
            }
            write_json(W5_ROOT / "results" / f"summary_v5_{bench_id}.json", summary)
            return summary

        eligible.sort(key=lambda t: (round(t[0], 7), t[1]), reverse=True)
        best_cid = eligible[0][2]
        best_py = cdir / f"{best_cid}.py"

        comp_conf_paths = {}
        for comp in comparators:
            comp_conf_paths[comp] = write_json(runs / f"confirm_{comp}.json", score_fn(comp, "confirm"))
        if baseline_id not in comp_conf_paths:
            comp_conf_paths[baseline_id] = write_json(runs / f"confirm_{baseline_id}.json", score_fn(baseline_id, "confirm"))
        w4_conf_p = write_json(runs / f"confirm_{w4_champ_id}.json", score_fn(w4_champ_id, "confirm"))
        cand_conf_p = write_json(runs / f"confirm_{best_cid}.json", score_fn(best_cid, "confirm"))

        confirm_verdicts = {}
        for comp in comparators:
            _, dec = run_guard(
                "gate-decision",
                "--candidate", str(cand_conf_p),
                "--incumbent", str(comp_conf_paths[comp]),
                "--baseline", str(comp_conf_paths[baseline_id]),
                "--rules", str(rc_path),
                "--ledger", str(ledger_path),
                "--trial-id", f"{bench_id}_v5_confirm_{best_cid}_vs_{comp}",
                "--comparator-id", comp,
                "--candidate-artifacts", str(best_py),
                allow_fail=True,
            )
            confirm_verdicts[comp] = {
                "cand": dec["candidate_primary"],
                "ref": dec["incumbent_primary"],
                "gain": dec["signed_improvement"],
                "lcb": dec["lcb_one_sided_vs_incumbent"],
                "signflip_p": dec.get("signflip_p_one_sided_vs_incumbent"),
                "signflip_method": dec.get("signflip_method"),
                "decision": dec["decision"],
                "tripwires": dec["tripwires"],
            }

        # Paired head-to-head Confirm comparison vs frozen Wave-4 Headless Champion (separate diagnostic rules without mutating main ledger)
        h2h_rules = dict(rules_confirm)
        h2h_rules["required_comparators"] = [w4_champ_id]
        h2h_rules_p = write_json(runs / "h2h_rules_confirm.json", h2h_rules)
        _, h2h_conf_dec = run_guard(
            "gate-decision",
            "--candidate", str(cand_conf_p),
            "--incumbent", str(w4_conf_p),
            "--baseline", str(comp_conf_paths[baseline_id]),
            "--rules", str(h2h_rules_p),
            "--trial-id", f"{bench_id}_v5_h2h_confirm_{best_cid}_vs_{w4_champ_id}",
            "--comparator-id", w4_champ_id,
            allow_fail=True,
        )
        h2h_confirm = {
            "cowork_cand_id": best_cid,
            "w4_champ_id": w4_champ_id,
            "cowork_metric": h2h_conf_dec["candidate_primary"],
            "w4_champ_metric": h2h_conf_dec["incumbent_primary"],
            "signed_gain_cowork_over_w4": h2h_conf_dec["signed_improvement"],
            "lcb_one_sided": h2h_conf_dec["lcb_one_sided_vs_incumbent"],
            "signflip_p": h2h_conf_dec.get("signflip_p_one_sided_vs_incumbent"),
            "decision": h2h_conf_dec["decision"],
            "tripwires": h2h_conf_dec["tripwires"],
        }

        seal_path = W5_ROOT / "seals" / f"HOLDOUT_SEAL_v5_{bench_id}.json"
        rc_h, holdout_res = run_guard(
            "gate-holdout",
            "--phase", "reserve",
            "--seal-file", str(seal_path),
            "--manifest", str(mf_path),
            "--expected-root-sha256", root_sha,
            "--ledger", str(ledger_path),
            "--rules", str(rc_path),
            "--candidate-id", best_cid,
            "--candidate-artifacts", str(best_py),
            allow_fail=True,
        )
        holdout_fin = None
        h2h_holdout = None
        if rc_h == 0 and holdout_res.get("status") == "RESERVED":
            cand_h = write_json(runs / f"holdout_{best_cid}.json", score_fn(best_cid, "holdout"))
            comp_h_args = []
            base_h_p = None
            for comp in comparators:
                cp_h = write_json(runs / f"holdout_{comp}.json", score_fn(comp, "holdout"))
                comp_h_args.extend(["--comparator-holdout", f"{comp}={cp_h}"])
                if comp == baseline_id:
                    base_h_p = cp_h
            if base_h_p is None:
                base_h_p = write_json(runs / f"holdout_{baseline_id}.json", score_fn(baseline_id, "holdout"))
            w4_h = write_json(runs / f"holdout_{w4_champ_id}.json", score_fn(w4_champ_id, "holdout"))

            _, holdout_fin = run_guard(
                "gate-holdout",
                "--phase", "finalize",
                "--seal-file", str(seal_path),
                "--manifest", str(mf_path),
                "--expected-root-sha256", root_sha,
                "--ledger", str(ledger_path),
                "--rules", str(rc_path),
                "--candidate-id", best_cid,
                "--candidate-artifacts", str(best_py),
                "--candidate-holdout", str(cand_h),
                *comp_h_args,
            )
            _, h2h_hold_dec = run_guard(
                "gate-decision",
                "--candidate", str(cand_h),
                "--incumbent", str(w4_h),
                "--baseline", str(base_h_p),
                "--rules", str(h2h_rules_p),
                "--trial-id", f"{bench_id}_v5_h2h_holdout_{best_cid}_vs_{w4_champ_id}",
                "--comparator-id", w4_champ_id,
                allow_fail=True,
            )
            h2h_holdout = {
                "cowork_cand_id": best_cid,
                "w4_champ_id": w4_champ_id,
                "cowork_metric": h2h_hold_dec["candidate_primary"],
                "w4_champ_metric": h2h_hold_dec["incumbent_primary"],
                "signed_gain_cowork_over_w4": h2h_hold_dec["signed_improvement"],
                "lcb_one_sided": h2h_hold_dec["lcb_one_sided_vs_incumbent"],
                "signflip_p": h2h_hold_dec.get("signflip_p_one_sided_vs_incumbent"),
                "decision": h2h_hold_dec["decision"],
            }

        _, ledger_ver = run_guard("verify-ledger", "--ledger", str(ledger_path))
        all_proposers = {**round1_candidates, **round2_candidates}
        summary = {
            "benchmark": bench_id,
            "manifest_root_sha256": root_sha,
            "baseline_search": r1_state["baseline_search"],
            "autosota_audit": r1_state["autosota_audit"],
            "w4_headless_champ_search": r1_state["w4_headless_champ_search"],
            "round1_leaderboard": r1_state["round1_leaderboard"],
            "round2_leaderboard": r2_board,
            "selected_finalist": best_cid,
            "selected_finalist_proposer": all_proposers.get(best_cid, "unknown"),
            "confirm_verdicts": confirm_verdicts,
            "head_to_head_confirm_vs_w4_champ": h2h_confirm,
            "holdout_reserve": holdout_res,
            "holdout_finalized": holdout_fin,
            "head_to_head_holdout_vs_w4_champ": h2h_holdout,
            "ledger_verification": ledger_ver,
        }
        write_json(W5_ROOT / "results" / f"summary_v5_{bench_id}.json", summary)
        return summary
    else:
        raise ValueError(phase)
