# Wave-5 Technical Study: Frontier `Cowork` (`Opus 5.5, Max` + `GPT 6.2 Astra, Max`) in Governed Recursive Self-Improvement

## 1. Motivation & Experimental Context

Across Waves 0–4 of our `AutoSOTA` Recursive Self-Improvement (RSI) investigation, we audited all `105` improved-paper repositories in Tsinghua University's `AutoSOTA` corpus, hardened `/governed-loop-optimization` to **`v3.6`** (`governed_loop_guard.py`, `90652db6649ad43cc3894f784c45e47d99f844807cda7c5ead0bd4c79ef1970a`), and demonstrated across 9 diverse scientific ML and combinatorial optimization domains (`D1–D9`) that multi-agent proposer diversity paired with strict two-stage Confirmation + Single-Touch Holdout gating eliminates false-positive discoveries.

However, during `Wave 4`, the Windows desktop hosting M365 Copilot `Cowork` had locked overnight (`session_active=false`), forcing Wave-4 proposals and reviews to use headless subagents rather than the desktop app's most capable frontier models. Once `Cowork` became available (`version 160.0.4411.0`), we executed **Wave 5 (`autosota_governed_v5`)** from scratch to rigorously re-run **both algorithmic proposals (Round 1 + Round 2 Crossover) and milestone scientific audits (Milestone 1 Pre-Flight + Milestone 2 Post-Holdout)** using M365 Copilot `Cowork`'s two flagship reasoning configurations:
- **`Claude Opus 5.5, Max`** (`Anthropic`, `Max` reasoning effort slider)
- **`GPT 6.2 Astra, Max`** (`OpenAI`, `Max` reasoning effort slider)

Crucially, because `Wave 4` had already unsealed its Protected Holdout sets on 5 of the 9 benchmarks (`D1, D5, D7, D8, D9`), re-optimizing on Wave-4 splits would violate **B2 (Single-Touch Holdout Hygiene)**. Wave 5 therefore constructed **brand-new disjoint 3-way splits (`Search_v5 / Confirm_v5 / Sealed Holdout_v5`)** with fresh cryptographic seeds (`20261118+`) and zero read access to Wave-4 evaluation files (`--forbid-paths immutable_eval /scratch/autosota_governed_v4`).

---

## 2. Milestone-1 Pre-Flight Audit (`Opus 5.5, Max`) & Empirical Data Hygiene

Before spending GPU compute on `24 × NVIDIA A100-80GB`, `Opus 5.5, Max` (`Cowork` thread `b44b1562-9d8e-4131-9f37-d37d18c92c46`) audited the proposed Wave-5 protocol and returned **`CONDITIONAL_GO`**, surfacing five structural leakage and power hazards that we verified empirically and fixed before `freeze-manifest`:

1. **Exact Feature-Row Deduplication Before Splitting (`D2` & `D8`):**
   - Direct inspection on `node-0` revealed massive duplicate feature rows in standard fairness/tabular benchmarks: `COMPAS` shrank from `4,937 -> 2,665` unique feature vectors (`46%` duplicates!) and `ACS` shrank from `29,949 -> 28,624` unique rows. Without pre-split deduplication, re-seeded random splits leak identical feature rows across `Search`, `Confirm`, and `Holdout`. Wave 5 deduplicated `COMPAS`, `ACS`, `Diabetes`, and `Ionosphere` prior to 3-way partitioning.
2. **Untouched Sensor Universe on `D4` (`PEMS04`):**
   - Instead of re-slicing Wave 4's `PEMS08` sensor set, Wave 5 moved `D4` to **`211` highway sensors in `PEMS04` (`307` total sensors) never touched in Wave 4**, partitioned into `12` temporally disjoint `400`-step blocks per split (`Search_v5`, `Confirm_v5`, `Holdout_v5`) separated by `96`-step (`8`-hour) guard buffers.
3. **Split-Isolated Ground Truth & Disjoint Temporal+Stock Clusters on `D5`:**
   - Wave 5 partitioned `400` `S&P 500` equities $\times$ `3` disjoint historical eras into **$G = 10$ stock-disjoint and time-disjoint `40`-stock blocks per split** (giving exact sign-flip resolution $2^{-10} = 9.77 \times 10^{-4} \ll \alpha/3 = 0.0167$), computing each split's reference precision topology strictly within that split's complete-return window.
4. **Spawn-Amortized Binary `.npz` Batching on `D9` (`D9-b2048`):**
   - Cold `python -I -B` startup takes `~0.20s`, and JSON serialization of millions of floats would dominate sub-second projection kernels. Wave 5 passed pre-allocated `.npz` batch paths (`B = 2048` $\times$ `12` internal repeats across `16` family-stratified clusters) so compute (`7–31s`) dwarfed spawn jitter (`< 3%`).
5. **Symmetric Pre-Registered Head-to-Head (`H2H`) Evaluation of Wave-4 Champions:**
   - Keeping `required_comparators` identical to Wave 4 (`K = 3`, Bonferroni $\alpha / 3 = 0.0167$) preserved apple-to-apple promotion standards while evaluating all 9 frozen Wave-4 Headless Champion `.py` scripts in parallel on the exact same `Search_v5`, `Confirm_v5`, and (when unsealed) `Holdout_v5` splits.

---

## 3. Per-Benchmark Algorithmic Mechanisms & Experimental Results (`D1–D9`)

### `D1` — Higher-Order Shapley Interaction (`k-SII`, Budget $\le 200$ Coalition Evaluations)
- **Wave-5 Winner:** `d1_v5_r1_astra_synergy_omp` (`Cowork GPT 6.2 Astra, Max`, Round 1)
- **Algorithmic Mechanism:** Exploits exact boundary-pair antithetic differences (`empty`, `full`, singletons, complements) plus cardinality-stratified ridge estimation to screen main effects, followed by **Orthogonal Matching Pursuit (OMP)** + focal pair probes on candidate synergy pairs within the strict `200`-evaluation cap.
- **Outcome:**
  - **Search (`Search_v5`):** `precision_at_10 = 1.0000`, `ksii_mse = 3.98e-5` (`8.4×` lower MSE than Wave-4's `3.35e-4`).
  - **Confirm (`Confirm_v5`):** `PROMOTE` vs. all 3 comparators (`0.9938` vs `0.2750` Antithetic MC, `0.3875` SVARM-IQ paper, `0.4063` AutoSOTA StratSize; all exact sign-flip $p = 1.53 \times 10^{-5}$).
  - **Sealed Holdout (`HOLDOUT_SEAL_v5_d1.json`):** **`1.0000` (100% perfect top-10 interaction recovery across all 16 Holdout game instances)** vs `0.2750` MC (`+0.7250`), `0.4500` SVARM-IQ (`+0.5500`), and `0.4000` AutoSOTA (`+0.6000`), all $p = 1.53 \times 10^{-5}$.
  - **H2H vs. Wave-4 Headless Champion (`d1_w4_champ_order3`):** **`1.0000` vs `0.9250`** on Sealed Holdout (`+7.50%`, one-sided bootstrap `LCB = +0.0121 > 0`, unshifted exact sign-flip $p = 0.0165$).

### `D2` — Worst-Group Fairness Without Group Labels (Deduplicated `ACS` + `COMPAS`)
- **Wave-5 Winner:** `d2_v5_r1_astra_boundary8` (`Cowork GPT 6.2 Astra, Max`, Round 1)
- **Algorithmic Mechanism:** Trains calibrated quadratic/MLP base conditional models and searches over an **8-direction latent error-boundary simplex** on inner OOF validation folds to maximize smoothed worst-latent-slice accuracy subject to `mean_acc >= 0.74`.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** Beat `ERM-MLP` (`0.6331` vs `0.3948`, `PROMOTE`, $p = 1.53 \times 10^{-5}$), beat the published `Latent-Score Reweight` paper (`0.6331` vs `0.5404`, `PROMOTE`, $p = 0.0025$), edged Wave-4's `d2_w4_champ_poly_minimax` (`0.6331` vs `0.6231`), and edged oracle-aligned `Group-DRO` (`0.6331` vs `0.6237`). Because the `+0.95 pp` margin over `Group-DRO` had `LCB = -0.0360` ($p = 0.418$), `Confirm` returned **`ROLLBACK`**, keeping `Holdout_v5` **100% sealed and untouched**.

### `D3` — Certified Monotonic Neural Regression (`0%` Monotonicity Violations)
- **Wave-5 Winner:** `d3_v5_r2_astra_crossover` (`Cowork GPT 6.2 Astra, Max`, **Round-2 Crossover**)
- **Algorithmic Mechanism:** Hybridizes `Astra R1`'s unconstrained-feature gated quartile splines with `Opus R1`'s nonnegative pairwise monotone hinge interactions $\max(0, x_a - \kappa_a)\max(0, x_b - \kappa_b)$ solved via projected FISTA (`c >= 0`), guaranteeing **exact `0.0%` finite-difference monotonicity violations by construction** in `1.43s` (`17×` faster than neural CMNN training).
- **Outcome:**
  - **Confirm (`Confirm_v5`):** Decisively beat published `CMNN-100ep` (`0.7375` vs `0.8639`, `PROMOTE`, $p = 0.00073$), beat `AutoSOTA CMNN-150ep` (`0.7375` vs `0.8674`, `PROMOTE`, $p = 0.00195$), **and formally `PROMOTE`d over Wave-4 Headless Champion `d3_w4_champ_convex_concave` (`0.7375` vs `0.7937`, `gain = +0.0562`, `LCB = +0.0256`, $p = 0.00464$ at `17×` speedup)**. Against `Monotone-Ridge-Spline`, it also won (`0.7375` vs `0.7767`, `LCB = +0.0129 > 0`, unshifted $p = 0.0144$), narrowly missing the margin-shifted $H_0: \mu_\Delta \le 0.004$ sign-flip threshold at $\alpha/3 = 0.0167$. `Holdout_v5` remained **100% untouched**.

### `D4` — Noisy Spatiotemporal Traffic Imputation (`PEMS04`, 211 Untouched Highway Sensors)
- **Wave-5 Winner:** `d4_v5_r1_opus_repair` (`Cowork Opus 5.5, Max`, Round 1 — `RePAIR`)
- **Algorithmic Mechanism:** Combines Huber/MAD robust spike-filtered low-rank Khatri-Rao/Hankel SVD background reconstruction with IRLS-$\ell_1$ temporal-Laplacian sensor regression, eliminating synthetic/sensor dropout spikes before fitting cross-sensor couplings.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** **`PROMOTE` across all 3 comparators** (`23.14` vs `48.71` Huber-Ridge, `48.10` RSTIB paper, and `28.41` Spatial-MLP with `LCB = +4.15`, $p = 2.44 \times 10^{-4}$), **unlocking a domain where Wave 4 had failed Confirmation**!
  - **Sealed Holdout (`HOLDOUT_SEAL_v5_d4.json`):** **`24.82` MAE** vs `50.36` Huber-Ridge (`+25.54`), `49.31` RSTIB (`+24.50`), and `30.64` Spatial-MLP (`+5.82`, 95% CI `[+4.84, +7.09]`, $p = 2.44 \times 10^{-4}$).
  - **Honest Reverse H2H Finding:** Evaluated in parallel on `PEMS04`'s `211` sensors, Wave 4's frozen `d4_w4_champ_kriging_svd` scored `23.17` vs `24.82` on Holdout (`21.97` vs `23.14` on Confirm) because spatial Gaussian-kernel Kriging benefited from `PEMS04`'s `2.2×` denser sensor network (`211` vs `96` sensors).

### `D5` — Graphical Lasso Under Informative Missingness (`S&P 500`, $G=10$ Disjoint Clusters)
- **Wave-5 Winner:** `d5_v5_r2_astra_crossover` (`Cowork GPT 6.2 Astra, Max`, **Round-2 Crossover**)
- **Algorithmic Mechanism:** Synthesizes propensity-calibrated Horvitz-Thompson inverse-probability weighting (IPW) + pairwise co-observation sample-size normalization $n_{ij}$ (`Opus` + `Astra` R2 insight) with eigenvalue-floored PSD correlation projection and multi-scale ADMM Graphical Lasso + debiased curvature scoring $|(2\Omega - \Omega R \Omega)_{ij}| \sqrt{n_{ij}} / \sqrt{\Omega_{ii}\Omega_{jj} + \Omega_{ij}^2}$.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** `PROMOTE` across all 3 comparators (`0.3278` vs `0.2449` Mean-Impute, `0.2451` Pairwise-Shrink, `0.2383` MSM Score-Matching paper; all $p = 9.77 \times 10^{-4}$).
  - **Sealed Holdout (`HOLDOUT_SEAL_v5_d5.json`):** **`0.3493` Edge AUPRC** vs `0.2596` Mean-Impute (`+0.0896`, 95% CI `[+0.0728, +0.1070]`), `0.2452` Pairwise-Shrink (`+0.1040`), and `0.2339` MSM paper (`+0.1154`), all $p = 9.77 \times 10^{-4}$.
  - **H2H vs. Wave-4 Headless Champion (`d5_w4_champ_multi_imp_stab`):** **`0.3493` vs `0.3207`** (`+0.0286` absolute / **`+8.9%` relative gain** on Holdout, $p = 0.081$).

### `D6` — Constrained Riemannian Manifold Sampling (`400`-Grad Cap, `err <= 1e-4`)
- **Search Tripwire Catch (`Opus R2`):** `d6_v5_r2_opus_crossover` (Mirror-SMC + energy-distance support-point herding) reached `sliced_w1_dist = 0.0555`, **but violated `max_manifold_err <= 1e-4` (`0.0565`)** due to off-manifold chordal drift during herding updates, and was automatically blocked by the Search tripwire gate!
- **Wave-5 Clean Winner:** `d6_v5_r1_astra_mirror_excursion` (`Cowork GPT 6.2 Astra, Max`, Round 1; exact manifold error `2.66e-15`).
- **Outcome:** Beat `RATTLE-HMC` (`0.0573` vs `0.2728`, `PROMOTE`), edged `Constrained-Langevin` (`0.0622`), edged the published `OLLA` paper (`0.0588`), and edged Wave-4's `d6_w4_champ_stein_equil` (`0.0576`), while `Confirm` properly blocked Holdout unsealing (`LCB = -0.0060` vs `OLLA`).

### `D7` — Combinatorial Ising MaxCut (`4` Graph Families $\times$ `3` Seeds)
- **Wave-5 Winner:** `d7_v5_r1_opus_bit` (`Cowork Opus 5.5, Max`, Round 1 — **Bifurcation-Initialized Tabu / BiT**)
- **Algorithmic Mechanism:** Evolves `64` parallel Kerr-nonlinear ballistic/discrete Simulated Bifurcation trajectories (`J = -W`, elastic walls at $|x|=1$) on GPU to populate diverse basins, then polishes top distinct cuts via batched exact incremental flip-gain $\Delta_i = s_i (Ws)_i$ Tabu search with aspiration.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** `PROMOTE` across all 3 comparators (`501.32` vs `489.22` RLSA paper, `489.94` AutoSOTA RLSA-Leap, `494.45` Spectral+Greedy; all $p = 1.53 \times 10^{-5}$) **AND formal `PROMOTE` in H2H Confirm over Wave-4 Champion `d7_w4_champ_dsb_memetic` (`501.32` vs `500.32`, `LCB = +0.372`, $p = 0.00125$)**.
  - **Sealed Holdout (`HOLDOUT_SEAL_v5_d7.json`):** **`500.86`** vs `487.57` RLSA paper (`+13.29`), `488.41` AutoSOTA (`+12.45`), `494.36` Spectral+Greedy (`+6.50`), all $p = 1.53 \times 10^{-5}$, **and beat Wave-4 Champion on Sealed Holdout (`500.86` vs `500.23`, `LCB = +0.170 > 0`, $p = 0.0202$)**.

### `D8` — Black-Box Variational Inference (`3` Deduplicated Posteriors)
- **Wave-5 Winner:** `d8_v5_r1_opus_qng_vi` (`Cowork Opus 5.5, Max`, Round 1)
- **Algorithmic Mechanism:** Initializes variational mode and precision via exact damped Newton/Laplace Hessian solve, then refines $(\mu, L)$ using randomized Sobol Quasi-Monte Carlo (`QMC`) reparameterization + Price/Stein-identity natural-gradient updates in **`0.51s` (`6.2×` faster than Wave 4)**.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** `PROMOTE` across all 3 comparators (`-72.3164` vs `-72.3647` LSVI paper, `-72.5506` BBVI-Reparam, `-73.5353` Laplace-Newton; all $p = 3.05 \times 10^{-5}$).
  - **Sealed Holdout (`HOLDOUT_SEAL_v5_d8.json`):** **`-69.7914` ELBO** vs `-69.8269` LSVI paper (`+0.0356` nats, 95% CI `[+0.0295, +0.0409]`, $p = 3.05 \times 10^{-5}$), `-70.0115` BBVI (`+0.2202`), `-70.8114` Laplace (`+1.0200`), and edged Wave-4 QMC-NatGrad (`-69.7914` vs `-69.7921`, `LCB = +0.00039 > 0`) at **`6.2×` speedup**.

### `D9` — Hard-Constrained Convex Projection (`PiNet`, Batched `.npz` `B=2048 × 12`)
- **Wave-5 Winner:** `d9_v5_r2_astra_crossover` (`Cowork GPT 6.2 Astra, Max`, **Round-2 Crossover**)
- **Algorithmic Mechanism:** Replaces 50-step iterative Douglas-Rachford / ADMM splitting with a batched GPU **Primal-Dual Active-Set KKT Cholesky projection** (`A x = b`, `l <= x <= u`) with a fallback 2-step feasibility polish, achieving **`1.03e-10` RMSE (`max_feasibility_err = 7.93e-10`) in `7.21s` (`3.8×` faster than baseline, `4.3×` faster than Wave-4 Vec-ADMM)**.
- **Outcome:**
  - **Confirm (`Confirm_v5`):** Crushed published `PiNet-DR50` (`1.03e-10` vs `0.3034`, `PROMOTE`), crushed `Overrelaxed-DR50` (`0.5461`, `PROMOTE`), beat Wave-4 `Vec-ADMM` (`1.03e-10` vs `7.94e-8`, `770×` lower RMSE), and beat `Dykstra-50` (`1.03e-10` vs `5.22e-4`, `5,000×` lower RMSE). Because `Dykstra-50` already sat at `5.22e-4` on simple box/simplex strata, subtracting the additive `min_improvement = 5e-4` margin left negative shifted deltas on those strata ($p = 0.359 \to \text{ROLLBACK}$), keeping `Holdout_v5` untouched and motivating relative/log-ratio margins near hard zero floors.

---

## 4. Milestone-2 Dual Frontier Post-Holdout Audits (`Opus 5.5, Max` + `GPT 6.2 Astra, Max`) & Next-Gen Guard Roadmap

Both **`Claude Opus 5.5, Max`** (`Cowork` Task `3260f5ab-6ec9-4d8c-a32d-a8912c9672a6`, `Worked for 12m 18s`) and **`GPT 6.2 Astra, Max`** (`Cowork` Task `c9bc5f6e-861c-4b72-b254-d88828a55b42`, `Worked for 3m 54s`) independently audited the full Wave-5 post-Holdout evidence (full verbatim transcripts in [`../cowork_audits_and_proposals/milestone2_dual_frontier_post_holdout_audits.md`](../cowork_audits_and_proposals/milestone2_dual_frontier_post_holdout_audits.md)) and converged on **`VERIFIED_WITH_CAVEATS`**:

1. **5/5 Sealed Holdout Replications (`D1, D4, D5, D7, D8`) & 4/9 Confirm Blocks (`D2, D3, D6, D9`) Upheld:**
   - Every reported Holdout $p$-value against the pre-registered 3-comparator suites hit its exact finite-sample Rademacher sign-flip floor $2^{-G}$ (`1.53e-5` on `D1` & `D7` [$G=16$], `3.05e-5` on `D8` [$G=15$], `2.44e-4` on `D4` [$G=12$], `9.77e-4` on `D5` [$G=10$]), clearing even a conservative `27`-test Bonferroni threshold (`0.00185`).
   - All 4 Confirm non-promotions (`D2` vs `GroupDRO`, `D3` vs `Monotone-Spline` margin, `D6` vs `OLLA` + Search catching `Opus R2` off-manifold herding `0.0565 > 1e-4`, and `D9` vs `Dykstra-50` additive zero-floor margin) were upheld as `VERIFIED_SOUND` — Holdout remained **100% sealed**.
2. **Rigorous Nuance on `Cowork Frontier vs. Wave-4 Headless` & the `D4` Sensor-District Lesson:**
   - Rather than conflating Confirm and Holdout into a single headline ratio, the two splits must be reported separately: on **Confirm (`9` domains)**, Wave-5 Cowork led in point estimate on `8/9` (`D3` formal win $p=0.00464$ at `17×` speedup; `D7` formal win $p=0.00125$; exploratory 9-domain sign test $p=0.020$), while `W4 Kriging-SVD` won `D4` (`21.97` vs `23.14`). On **Sealed Holdout (`5` unsealed domains)**, point estimates favored Wave-5 Cowork on `4/5` (`D1` `1.000` vs `0.925`, exact margin-shifted $p=1079/65536=0.0165$; `D7` `500.86` vs `500.23`, $p=0.0202$; `D5` `+8.9%` directional $p=0.081$; `D8` `+0.0007` nats at `6.2×` speedup), while `W4 Kriging-SVD` won `D4` (`23.17` vs `24.82`).
   - Crucially, `D4`'s Confirm+Holdout passage in Wave 5 resulted from moving from sparser `PEMS08` (`96` sensors in Wave 4) to denser untouched `PEMS04` (`211` sensors in Wave 5), where *both* `W4 Kriging-SVD` and `Opus RePAIR` beat `Spatial-MLP` (`30.64`).
3. **Concrete Next-Generation (`v3.7`) Guard Improvements Distilled from Both Frontier Audits:**
   - **Mandatory Incumbent Non-Inferiority/Superiority Comparator:** Automatically append any prior-wave champion to `required_comparators` so a newly promoted solver can never regress below a previous wave's incumbent (preventing the `D4` gap).
   - **Intersection-Union Test (IUT) Within Domain + Holm Across Domains:** Requiring simultaneous wins over all $K$ comparators is an IUT that already controls single-domain Type-I error at $\alpha$ without dividing by $K$, whereas Holm step-down across domains controls family-wise campaign error.
   - **Freeze-Time Zero-Floor Margin Guard:** Reject additive `min_improvement` values $\ge 50\%$ of any comparator's Search error near `0.0` (such as `D9` `0.0005` vs `Dykstra-50` `0.000522`), enforcing log-ratio margins or two-stage accuracy-noninferiority + speed superiority gates.
   - **Manifest-Locked Selection Rule & Encrypted Holdout Keys:** Hash the tournament orchestrator's Search tie-break rule inside `IMMUTABLE_MANIFEST.json` and release Holdout decryption keys exclusively inside `gate-holdout --phase reserve`.

