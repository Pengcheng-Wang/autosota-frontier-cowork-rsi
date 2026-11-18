# Milestone 1 — Frontier Cowork Pre-Flight Design Audit (`Opus 5.5, Max`)

- **Reviewer:** M365 Copilot `Cowork` — **`Opus 5.5, Max`** (`Claude Anthropic` Tier-1 Flagship, Maximum Effort)
- **Thread Title / ID:** `Wave-5 RSI Re-run Design Audit` (`b44b1562-9d8e-4131-9f37-d37d18c92c46`)
- **Timestamp:** `2026-11-18`
- **Verdict:** **`CONDITIONAL_GO`** (all must-fix items incorporated into `autosota_governed_v5` before Round-1 GPU spend)

## Verbatim Audit Response (`Opus 5.5, Max`)

> **Verdict: CONDITIONAL_GO.** The design is sound, but fix the must-fix items below first. D4 and D5 may only support report-only Holdout claims [unless temporal/sensor independence is enforced].
>
> **(A) Leakage**
> Exclude everything Wave-4 touched. Any unit Wave-4 used (any split, on any benchmark that shares the source) should go into Search_v5 only [or be excluded from Wave-5 eval]. Check this by content hash (rows, graphs, QP matrices, masks), not by index or seed. New seeds ≠ new data. New noise on D4, or new algorithm RNG on fixed D6/D7/D9 instances, only refreshes noise, not the task. Disjoint D8 row slices of one dataset give near-identical posteriors. Refresh the instance, or label that Holdout non-fresh.
> - **D1:** refit the trees with new seeds, on rows disjoint from the explained instances. Draw the imputation background from Search/Train only. If candidate code can reach the model object, exact tree algorithms skip the 200-call budget and reproduce GT. Expose only an out-of-process oracle that counts calls.
> - **D2/D3/D8:** use one global split per shared CSV. Group rows by entity and dedupe identical rows. Refit scalers, encoders and bins on Search_v5-train only.
> - **D4:** fit the scaler on Search_v5-train only, because a prefix over interleaved blocks sees Holdout time. Gap > 24 only works if the interface hard-caps lookback at L (no daily/weekly periodic inputs). If Wave-4 touched all 59 days, no fresh block exists [on those same sensors].
> - **D5:** compute GT per item from that item's own complete data, with λ frozen before the draw. Don't standardize on full history.
> - **Biggest risk:** `python -I -B` isolates the interpreter, not the data. Clean PEMS series, raw prices, GT, model pickles, Wave-4 directories and git history must all be unreadable (`--forbid-paths`). Also: scoring done by the harness.
>
> **(B) Champions: use a parallel arm, but give it a decision role**
> Keep `required_comparators` identical to Wave-4. That keeps this a clean replication with a comparable gate. Run the champions as a pre-registered paired arm in the same Confirm look and Holdout touch. Replacing the incumbent requires superiority under strict hill-climb, or non-inferiority under frontier mode. Requiring both gates is an intersection-union test, so it costs no α. Freeze champion code but refit any fitted state on Search_v5-train. Don't give champions a dual role. The Confirm slot goes to the best new candidate; the champion gets a replication test. Re-gating a champion that failed in Wave-4 is a second chance at the same claim, so report both waves.
>
> **(C) D9 and Confirm looks**
> - **D9 batching:** Totals of 0.204–0.255 s against ~0.20 s of spawn imply at most ~55 ms of compute per 48 vectors. Scaling by 10.67× gives at most ~0.59 s. That is the same order as spawn, not ≫. Scale higher (`D9-b2048`) and rerun all comparators symmetrically, checking max violation per vector in `float64` (not the batch mean) and guarding optimality gap.
> - **Look accounting:** Two looks at α = 0.025 each push FWER up to 0.05; spend `0.0125` per look (`effective_alpha = alpha / max_confirm_looks`). A look counts as soon as Confirm data is accessed. Log every look in a hash-chained ledger. No proposer activity after any look. At `0.0125` you need at least 7 independent clusters (`2^-k <= 0.0125`); cluster D5 by disjoint temporal regime and D4 by physical day on Wave-4-untouched sensors.

---

## Wave-5 (`autosota_governed_v5`) Engineering Remedies Implemented from Audit

1. **Content-Disjoint / Task-Refreshed Wave-5 Splits:**
   - **`D1` (`shapiq`):** Refit trees with new `random_state=5042` on strictly disjoint train prefix; sample explained instances strictly from Wave-4-untouched tail slices (`California[15000:18000]`, `Bike[15000:17379]`, `Adult[29000:32561]`) + fresh `SOUM` hypergraphs; candidate receives ONLY `(mask, v(mask))` scalars via 3-step `run-isolated` oracle.
   - **`D2`, `D3`, `D8` (`ACS`, `CMNN`, `Adult`):** Deduplicate identical feature-target rows by SHA-256 row hash before splitting; fit scalers strictly on `Train_v5`; for `D8`, construct genuinely diverse logistic posteriors across 3 distinct datasets (`ACS`, `COMPAS`, and `Adult-Census`, which Wave-4 `D8` never touched) with varied feature subsets and prior precisions.
   - **`D4` (`PEMS04`):** Select the **next 96 highway sensors (`rank 96..191`, 0% sensor overlap with Wave-4's `rank 0..95`)**, fit standardization strictly on `Train_v5` (`[0:8640]`), hard-cap candidate lookback at `L=12`, and cluster into **`12` non-overlapping physical days per split** (`gap >= 36 > L+H=24` steps between days).
   - **`D5` (`S&P 500`):** Partition `1,258` trading days into **`36` non-overlapping bimonthly temporal regimes** (`12` strictly disjoint time regimes per split `Search_v5 / Confirm_v5 / Holdout_v5` with 2-day buffer gaps), standardize and fit reference `GraphicalLasso(alpha=0.14)` **strictly inside each regime's own unmasked returns** (zero full-history leakage, `12` temporally independent clusters per split -> min exact sign-flip $p = 2^{-12} = 0.000244 \ll 0.0125$).
   - **`D6`, `D7`, `D9`:** Brand-new manifold/graph/QP problem matrices verified hash-disjoint from Wave-4; `--forbid-paths` blocks both `immutable_eval/` and `/scratch/autosota_governed_v4`.
2. **Parallel Paired Champion Arm (`w4_headless_champ`):**
   - `required_comparators` stays identical to Wave-4 (original paper + strong domain comparators).
   - Confirm slot goes strictly to the **best Wave-5 Cowork-originated candidate** (`Opus 5.5 Max` R1, `GPT 6.2 Astra Max` R1, or `Cowork R2 Crossover`), while `w4_headless_champ` (refit strictly on `Train_v5`) is evaluated in parallel on `Search_v5`, `Confirm_v5`, and `Holdout_v5` with a dedicated paired `gate-decision` head-to-head comparison.
3. **`D9-b2048` Batch Amortization + Per-Vector `float64` Max Violation + `alpha/K = 0.0125`:**
   - Scale `D9` to `B = 2048` vectors per QP instance (`32,768` vectors per split, `~2.5s` pure compute $\gg 0.20s$ spawn) symmetrically across all comparators and candidates, enforcing `max_feasibility_err <= 1e-4` over every vector in `float64`.
