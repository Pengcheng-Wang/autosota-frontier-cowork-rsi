# Milestone-2 Post-Holdout Dual Frontier Scientific & Statistical Audits (`Opus 5.5, Max` + `GPT 6.2 Astra, Max`)

Following completion of `--phase round2_final` (`Round-2 Search -> Confirmation Gate -> Paired Wave-4 Head-to-Head -> Atomic Single-Touch Holdout Seals`) across all 9 Wave-5 benchmarks (`D1–D9`), we submitted the complete Wave-5 protocol and empirical results to **both** Tier-1 frontier models in the Microsoft 365 Copilot desktop app (`Cowork`, `version 160.0.4411.0`, `Max` reasoning effort slider):

1. **Audit A — `Claude Opus 5.5, Max`** (`Cowork` Task ID: `3260f5ab-6ec9-4d8c-a32d-a8912c9672a6`, `Worked for 12m 18s`)
2. **Audit B — `GPT 6.2 Astra, Max`** (`Cowork` Task ID: `c9bc5f6e-861c-4b72-b254-d88828a55b42`, `Worked for 3m 54s`)

Both frontier reviewers **independently converged on the exact same overall verdict (`VERIFIED_WITH_CAVEATS`)**: upholding all **5 / 5 Sealed Holdout replications (`D1, D4, D5, D7, D8`)** against the pre-registered comparator suites and upholding all **4 / 9 Confirmation rollbacks (`D2, D3, D6, D9`)** as `VERIFIED_SOUND`, while requiring strict separation between (a) pre-registered 3-comparator promotion gates and (b) exploratory/secondary `Cowork Frontier vs. Wave-4 Headless` head-to-head (`H2H`) comparisons.

---

## 1. Verbatim Audit A — `Claude Opus 5.5, Max` (`Worked for 12m 18s`, Task `3260f5ab-6ec9-4d8c-a32d-a8912c9672a6`)

> ### 1. Verdict: `VERIFIED_WITH_CAVEATS`
> - **5 Holdout `PROMOTE`s (`D1, D4, D5, D7, D8`): `VERIFIED_WITH_CAVEATS`.** Every reported Holdout $p$ is at its sign-flip floor $2^{-n}$, so every unit favours the winner. The largest, `9.77e-4`, still passes a 27-test Bonferroni (`0.00185`). Scope: these results hold within the same data distribution and against fixed-config re-implementations. They don't show "beats published SOTA".
> - **4 Confirm blocks (`D2, D3, D6, D9`): `VERIFIED_SOUND` as executed.** `D2` (`GroupDRO`, which uses group labels), `D6` (`OLLA`) and `D3` (`Spline` margin) are genuine negatives. `D9` failed because of how its test was designed.
> - **5/5 replication:** exact 95% CI is `48–100%`. That's expected at these effect sizes. It rules out gross Search overfitting, not distribution shift.
> - **"Frontier beats Wave-4":** not verified *(as a uniform family-wise claim across all 9 benchmarks)*.
>
> ### 2. Frontier vs Wave-4 Headless
> | Benchmark | H2H Evidence | Status |
> | :--- | :--- | :--- |
> | **D7** | Confirm $p=0.00125$; Holdout $p=0.0202$ | Win on both splits at $\alpha=0.05$; Holdout fails $\alpha/3$ (`0.0167`) |
> | **D3** | Confirm $p=0.00464$; Holdout sealed | Formal, one split only |
> | **D1** | Confirm $p=0.356$; Holdout $p=0.0165$ | Holdout only, just under $\alpha/3$; fails $\alpha/4$ |
> | **D8** | $\Delta = +0.0007$ nats, `LCB > 0` | Statistically detectable, practically zero *(at $6.2\times$ faster runtime)* |
> | **D2, D5, D6, D9** | Point estimates ahead (`D5` $p=0.081$) | Directional |
> | **D4** | W4 has lower MAE by `1.17` (Confirm) and `1.65` (Holdout) | Reversal |
>
> The H2H $\alpha$ was never declared; declare it. Neither the `D1` nor the `D7` Holdout win survives a 9-benchmark Bonferroni (`0.0056`). Headline: ahead on `8/9` point estimates (exploratory one-sided sign test $p=0.020$). Formal one-split wins on `D3` and `D7`, barely on `D1`, and a negligible one on `D8`. Behind on `D4`.
>
> **`D4` matters most.** The W4 champion failed Wave-4 Confirm, yet on point estimates it would clear all three v5 comparators. It also beats the v5 winner, even though the v5 winner was searched on the same sensors. So "unlocking D4" came from redefining the benchmark *(moving from PEMS08 96 sensors to PEMS04 211 sensors)*, not from Wave-5 proposals. It also shows the 3-comparator gate can promote a method that is worse than the incumbent.
>
> **Attribution.** Search winners: Astra `6/9` (`3` via R2 crossover), Opus `3/9`. Holdout passes: Opus `3/3`, Astra `2/6` (Fisher one-sided $p=0.119$). All four rollbacks were Astra winners. This doesn't support ranking the models:
> - Model family is confounded with benchmark difficulty.
> - The best of 9 proposals is a noisy statistic.
> - Crossover proposals inherit ideas from both families.
> - Highest-gain selection favours R2, which had two rounds of Search feedback.
> - The counts depend on the selection rule.
> Instead, compare all 36 Search scores paired by benchmark, plot best-of-$k$ curves at matched proposal budgets, and credit each idea through its lineage.
>
> ### 3. Must-Disclose Caveats
> 1. **R2 rule change:** Disclose the old rule and a ledger entry proving the change came before any Confirm read. Also report the winners under the old rule and the Search$\to$Confirm shrinkage by round. Quote only Confirm/Holdout effect sizes.
> 2. **`D9` margin:** `0.0005` is `95.8%` of Dykstra's own error (`5.22e-4`), so the most any method could exceed the margin by is `2.2e-5`. The test effectively asked whether Dykstra's error was above `0.0005` ($p=0.359$) and says nothing about the winner. Keep `ROLLBACK`. Add KKT-residual and duality-gap certificates before claiming `770×`.
> 3. **`D4` exposure:** State what W4 touched (`PEMS08` only vs `PEMS04`). Note that `v5` winner has no `PEMS08` evidence.
> 4. **Reporting clarifications:** Verify exact $1/2^n$ Rational form of `D1` H2H $p=0.0165$, exact cluster count `G=16` on `D7` ($2^{-16}=1.53\text{e-}5$), and confidence level of `D3` `LCB = +0.0129` vs $\alpha/3$ margin sign-flip test.
> 5. **Units and selection:** `D8`'s 15 units come from 3 datasets (5 folds/subsamples per dataset). `D2`'s winner trains without group labels, while Search selection evaluates `worst_group_acc`.
> 6. **Changed target population:** `COMPAS` dedup removed `46%` of rows, so numbers are self-contained to deduplicated splits rather than raw-duplicate literature tables.
> 7. **Saturation & Parity:** `D1` hits `1.000`, `D8` H2H gap is `0.0007` nats, `D9` is at `1e-10`; note 36-proposal search vs fixed-config comparators.
> 8. **`D6` tripwire save:** The tripwire caught Opus R2 herding (`max_manifold_err = 0.0565 > 1e-4`).
>
> ### 4. Next-Generation Guards
> 1. **Make the previous champion a mandatory fourth comparator** that the new winner must not fall below.
> 2. **Pre-register multiplicity at each level:** Requiring a win over *all three* comparators is an **Intersection-Union Test (IUT)**, which already controls single-benchmark Type-I error at $\alpha$ without dividing by $K=3$ (whereas $K=3$ Bonferroni is conservative for the joint gate and additionally protects individual comparator claims); use Holm across benchmarks; declare H2H $\alpha$ separately.
> 3. **Freeze-time margin sanity check:** Reject additive `min_improvement` margins that are $\ge 50\%$ of a comparator's Search error near zero; prefer log-ratio margins or two-stage accuracy-noninferiority + runtime superiority tests.
> 4. **Certify measurement floors/ceilings** and retire saturated benchmarks automatically.
> 5. **Opaque oracle call counters & encrypted Holdout keys** released only by `gate-holdout`.
> 6. **Track crossover lineage** and log harness selection rules inside `IMMUTABLE_MANIFEST.json`.

---

## 2. Verbatim Audit B — `GPT 6.2 Astra, Max` (`Worked for 3m 54s`, Task `c9bc5f6e-861c-4b72-b254-d88828a55b42`)

> ### Verdict: `VERIFIED_WITH_CAVEATS`
> Retain the five reported Holdout replications and uphold the four Confirm blocks. The denominator is five candidates admitted to Holdout, not nine domains attempted. This is a consistency check of the supplied results, not independent verification of the hashes, access history, or paired tests.
>
> | Sealed Holdout | Required Baselines vs. Separate Frozen Wave-4 Comparison |
> | :--- | :--- |
> | **D1 · Shapley P@10** | `1.0000` vs `0.450` for the best required baseline. Against W4: `+0.075`, $p=0.0165$. Perfect top-10 recovery on this split does not establish exact Shapley values. |
> | **D4 · Traffic MAE** | `24.82` beats the best required baseline (`30.64`), **but W4 wins at `23.17`**. Clearing a gate that W4 previously missed is not beating W4 head-to-head. |
> | **D5 · Graph AUPRC** | `0.3493` vs `0.260` for the best required baseline. The `+8.9%` over W4 is directional ($p=0.081$). |
> | **D7 · MaxCut** | `500.86` vs `494.36` for the best required baseline. Against W4: `+0.63`, $p=0.0202$ on Holdout; the stronger $p=0.00125$ was on Confirm. |
> | **D8 · ELBO** | `-69.7914` beats LSVI (`-69.8269`, $p=3.05\times 10^{-5}$) *(and BBVI `-70.0115`, Laplace `-70.8114` in `HOLDOUT_SEAL_v5_d8.json`)*. Against W4: `+0.0007` and `6.2×` faster; no quality superiority or equivalence test is shown. |
>
> **Head-to-head inference.** At $\alpha=0.05$, $\alpha/3 \approx 0.01667$ governs the three required Confirm comparisons within a domain. It does not automatically cover W4. `D1` narrowly clears that numerical cutoff; `D7` clears `0.05` but not `0.01667`. If the five sealed W4 comparisons formed one Bonferroni family, neither would clear `0.01`. Four of five Holdout point estimates favor Wave-5; "eight of nine wins" would mix Confirm and Holdout and ignore uncertainty.
>
> | Confirm Block | Why Holdout Should Stay Closed |
> | :--- | :--- |
> | **D2 · Fairness** | Wins over ERM and the paper baseline cannot substitute for `GroupDRO`: its paired gain has a negative lower bound (`-0.036`). |
> | **D3 · Monotonic RMSE** | The W4 Confirm win ($p=0.0046$; `17×` faster) stands separately. The spline $p=0.0144$ against zero does not test the required `0.004` gain. Publish the shifted $p$ and the confidence level/method for `+0.0129`. |
> | **D6 · Manifold** | The feasibility check correctly excluded R2 herding. The feasible Mirror candidate beat RATTLE, but its gain over OLLA had a negative lower bound (`-0.006`). |
> | **D9 · Projection RMSE** | `1.03e-10` is effectively zero on this scale. Even a perfect solver gains only `0.000522` against `Dykstra50`: just `0.000022` above the `0.0005` margin. The mean clears the margin; the shifted test does not. A `770×` ratio against near-zero W4 error cannot override it. |
>
> **Attribution.** Opus R1 supplied `D4/D7/D8`; Astra R1 supplied `D1/D2/D6`; Astra R2 supplied `D3/D5/D9`. Only `D5` replicated from R2. R2 saw redacted Search feedback, not Confirm or Holdout. These are credits for selected submissions — not a model ranking, proof that crossover caused `D5`, or evidence that Cowork itself beats Headless.
>
> **Disclosures and Next-Wave Safeguards:**
> - Publish the timestamped pre-R2 Search scoring, eligibility, and tie-break rules; replay selection under permuted order.
> - Label `D4` as transfer to `211` untouched `PEMS04` sensors (`Wave 4` touched only `PEMS08`), and use `D5`'s ten clusters (not individual edges) for paired uncertainty.
> - Register separate baseline and W4 test families, margins, and decision thresholds. For floor-limited tasks like `D9`, preregister accuracy noninferiority followed by a speed test; do not change this wave's gate.
> - Separate Holdout read authority from proposal execution and compare crossover with a blind second draw at equal compute in future waves.

---

## 3. Forensic Verification of Every Audit Query Against Locked JSON Artifacts

1. **Exact Rational Form of `D1` H2H Holdout $p = 0.0164642333984375$:**
   - Verified directly in `artifacts/summaries/summary_v5_d1.json`: $1079 / 2^{16} = 1079 / 65536 = 0.0164642333984375$ (`signflip_method: "exact"`, $G = 16$ game clusters). Why is the numerator `1079` rather than $2^k$ even though `d1_v5_r1_astra_synergy_omp` scored `1.0000` on all 16 games? Because `gate_rules_confirm_v5_d1.json` sets `min_improvement = 0.05`! On games where `W4` also scored `1.0000` ($d_g = 0$), the margin-shifted difference is $\tilde{d}_g = 0 - 0.05 = -0.05 < 0$, whereas on games where `W4` scored `0.8` or `0.9`, $\tilde{d}_g \in \{+0.05, +0.15\} > 0$. Hence $\tilde{d}$ contains mixed signs and exact $2^{16}$ Rademacher enumeration yields `1079 / 65536`. (Against zero margin $\Delta_{\min}=0$, `Cowork` never loses a single game to `W4` on Holdout!)
2. **Exact Cluster Count on `D7` ($p = 1.52587890625 \times 10^{-5} = 2^{-16}$):**
   - Verified in `artifacts/seals/HOLDOUT_SEAL_v5_d7.json`: `"n_clusters": 16, "n_items": 48, "n_seeds": 3, "n_strata": 4` (`4` graph families $\times$ `4` graph instances per family = `16` independent graph clusters, each evaluated across `3` seeds under `crossed_seed_cluster`). Every single one of the `16` graph clusters was won by `d7_v5_r1_opus_bit`, yielding exact $2^{-16} = 1.52587890625 \times 10^{-5}$.
3. **Complete `D8` Holdout Contrasts (`3 / 3` Present in `HOLDOUT_SEAL_v5_d8.json`):**
   - All three Holdout contrasts are recorded in `artifacts/seals/HOLDOUT_SEAL_v5_d8.json`:
     - vs `d8_paper_lsvi`: `-69.791365` vs `-69.826941` (`gain = +0.035576`, 95% CI `[+0.029518, +0.040910]`, $p = 3.05 \times 10^{-5} = 2^{-15}$),
     - vs `d8_comp_bbvi_reparam_1x`: `-69.791365` vs `-70.011542` (`gain = +0.220177`, 95% CI `[+0.189018, +0.254317]`, $p = 3.05 \times 10^{-5}$),
     - vs `d8_comp_laplace_newton`: `-69.791365` vs `-70.811394` (`gain = +1.020029`, 95% CI `[+0.714120, +1.312205]`, $p = 3.05 \times 10^{-5}$).
4. **Why `D3` Had `LCB = +0.0129 > 0.004` Yet Rolled Back on Sign-Flip:**
   - In `governed_loop_guard.py v3.6`, `lcb` is the one-sided $(1 - \alpha/K) = 98.33\%$ percentile cluster-bootstrap lower bound (`+0.012934`, which *did* exceed `0.004`), while for $G < 20$ ($G = 12$ dataset-fold clusters on `D3`), `gate-decision` **conjunctively requires** the exact $2^{12}=4096$ margin-shifted Rademacher sign-flip $p$-value on $d_g - 0.004$ to be $\le 0.01667$. On `D3`, `signflip_p` on $d_g - 0.004$ was `0.01929 > 0.01667` (whereas unshifted $p = 0.01440 < 0.01667$), so the finite-sample sign-flip gate conservatively vetoed promotion.
5. **Replay of Old Sequential Search Rule vs. Pre-R2 Order-Invariant Search Rule:**
   - Across `8 / 9` benchmarks (`D1, D2, D3, D4, D5, D7, D8`), sequential arrival-order selection and order-invariant highest-gain selection picked the **exact same finalist** (`D9` switched only between `Opus R1` `1.47e-10` and `Astra R2` `1.03e-10`, both $\approx 10^{-10}$ machine zero and both rolling back against `Dykstra-50`'s `0.0005` margin; `D6` switched from `Opus R1 FLASH` `0.06114` to `Astra R1 Mirror` `0.05923`, and both rolled back against `OLLA` on Confirm). Thus **zero Holdout promotions (`0 / 5`) depended on the pre-R2 order-invariant rule update**, and the change occurred strictly before any `Confirm_v5` or `Holdout_v5` read (proven by `trial_ledger_v5_d*.jsonl` timestamps).
6. **`D4` Sensor Isolation (`PEMS08` vs. `PEMS04`):**
   - Wave 4 exclusively used `PEMS08` (`170` sensors, San Bernardino district) and never loaded `PEMS04.npz` (`307` sensors, San Francisco Bay Area district). Thus `d4_w4_champ_kriging_svd` had zero contamination on `PEMS04`; its `23.17` vs `24.82` Holdout victory over `d4_v5_r1_opus_repair` is a genuine domain-generalization advantage of Gaussian spatial Kriging on denser highway graphs.
