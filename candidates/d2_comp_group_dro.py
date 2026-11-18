import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
VARIANT = 'comp_group_dro'

def fit_and_predict(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)  # 0..3 demographic groups
    w_sc = np.asarray(bundle["w_score"], dtype=np.float64)
    sex_idx = int(bundle["sex_idx"])
    race_idx = int(bundle["race_idx"])
    g8_tr = g_tr * 2 + y_tr  # 0..7 demographic x label groups

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_tr))
    va_n = 2800
    va_i, tr_i = perm[:va_n], perm[va_n:]

    xm = np.mean(X_tr[tr_i], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[tr_i], axis=0, keepdims=True) > 1e-6, np.std(X_tr[tr_i], axis=0, keepdims=True), 1.0)
    Xtr_s = ((X_tr[tr_i] - xm) / xs).astype(np.float32)
    Xva_s = ((X_tr[va_i] - xm) / xs).astype(np.float32)
    ytr_f = y_tr[tr_i].astype(np.float32)
    gtr = g_tr[tr_i]
    g8tr = g8_tr[tr_i]
    wtr = np.clip(w_sc[tr_i], 0.05, 15.0).astype(np.float32)
    wtr /= np.mean(wtr)

    if VARIANT == "decision_shrunk_gdro_logit":
        # Proposer D: Shrunk 0.5 ERM + 0.5 GroupDRO logistic regression + validation threshold selection
        d = Xtr_s.shape[1]
        theta = np.zeros(d + 1, dtype=np.float64)
        Xb = np.hstack([Xtr_s.astype(np.float64), np.ones((len(Xtr_s), 1))])
        Xvb = np.hstack([Xva_s.astype(np.float64), np.ones((len(Xva_s), 1))])
        alpha_g = np.ones(8, dtype=np.float64) / 8.0
        w_d = wtr.astype(np.float64)
        m_adam = np.zeros_like(theta); v_adam = np.zeros_like(theta)
        y_pm = 2.0 * ytr_f.astype(np.float64) - 1.0
        for t in range(1, 201):
            margin = y_pm * (Xb @ theta)
            losses = np.log1p(np.exp(-np.clip(margin, -25, 25))) * w_d
            d_margin = -y_pm / (1.0 + np.exp(np.clip(margin, -25, 25))) * w_d
            grad_erm = (Xb.T @ d_margin) / len(Xb)
            grad_gdro = np.zeros_like(theta)
            l_g = np.zeros(8)
            for gv in range(8):
                mg = (g8tr == gv)
                if np.any(mg):
                    l_g[gv] = float(np.mean(losses[mg]))
                    grad_gdro += alpha_g[gv] * ((Xb[mg].T @ d_margin[mg]) / np.sum(mg))
            alpha_g *= np.exp(0.1 * l_g)
            alpha_g /= np.sum(alpha_g)
            grad = 0.5 * grad_erm + 0.5 * grad_gdro + 2e-3 * np.append(theta[:-1], 0.0)
            m_adam = 0.9 * m_adam + 0.1 * grad
            v_adam = 0.999 * v_adam + 0.001 * (grad ** 2)
            mh = m_adam / (1.0 - 0.9 ** t)
            vh = v_adam / (1.0 - 0.999 ** t)
            theta -= 0.02 * mh / (np.sqrt(vh) + 1e-8)

        p_va = 1.0 / (1.0 + np.exp(-np.clip(Xvb @ theta, -25, 25)))
        best_t, best_wg = 0.5, -1.0
        y_va, g8_va = y_tr[va_i], g8_tr[va_i]
        for t_cand in np.linspace(0.20, 0.80, 61):
            pred_v = (p_va >= t_cand).astype(np.int64)
            if np.mean(pred_v == y_va) < 0.745:
                continue
            wg = min(float((np.sum((pred_v == y_va) & (g8_va == gv)) + 1.0) / (np.sum(g8_va == gv) + 2.0)) for gv in range(8))
            if wg > best_wg:
                best_wg, best_t = wg, float(t_cand)

        outs = []
        for it in eval_items:
            Xev = ((np.asarray(it["X_eval"], dtype=np.float64) - xm) / xs)
            Xevb = np.hstack([Xev, np.ones((len(Xev), 1))])
            pev = 1.0 / (1.0 + np.exp(-np.clip(Xevb @ theta, -25, 25)))
            outs.append({"id": it["id"], "prediction": (pev >= best_t).astype(int).tolist()})
        return outs

    # Neural MLP models (ERM, Paper LatentScoreReweight, GroupDRO, Coordinator Group-Bayes Post-Calibration)
    torch.manual_seed(42)
    d_in = Xtr_s.shape[1]
    net = nn.Sequential(
        nn.Linear(d_in, 96),
        nn.SiLU(),
        nn.Linear(96, 48),
        nn.SiLU(),
        nn.Linear(48, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-2)
    xt = torch.tensor(Xtr_s)
    yt = torch.tensor(ytr_f)
    wt = torch.tensor(wtr)
    g8t = torch.tensor(g8tr, dtype=torch.long)
    xv = torch.tensor(Xva_s)

    q_dro = torch.ones(8) / 8.0
    best_state = None
    best_val_metric = -1e9

    n_epochs = 140
    batch_sz = 1024
    for ep in range(n_epochs):
        net.train()
        p_idx = torch.randperm(len(xt))
        for b0 in range(0, len(xt), batch_sz):
            sl = p_idx[b0:b0 + batch_sz]
            logits = net(xt[sl]).squeeze(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, yt[sl], reduction="none")
            if VARIANT == "base_erm":
                loss = torch.mean(bce)
            elif VARIANT == "paper_reweight":
                loss = torch.mean(bce * wt[sl])
            elif VARIANT == "comp_group_dro":
                g_losses = torch.zeros(8)
                for gv in range(8):
                    m = (g8t[sl] == gv)
                    if torch.any(m):
                        g_losses[gv] = torch.mean(bce[m])
                with torch.no_grad():
                    q_dro = q_dro * torch.exp(0.05 * g_losses)
                    q_dro = q_dro / torch.sum(q_dro)
                loss = torch.sum(q_dro * g_losses)
            elif VARIANT == "coord_group_bayes_calib":
                # Balanced inverse-group-prior + score-reweighted representation learning
                cnts = torch.bincount(g8t[sl], minlength=8).float().clamp(min=1.0)
                inv_p = 1.0 / torch.sqrt(cnts[g8t[sl]])
                loss = torch.mean(bce * inv_p * torch.clamp(wt[sl], 0.3, 3.0))
            else:
                raise ValueError(VARIANT)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % 20 == 0:
            net.eval()
            with torch.no_grad():
                lv = net(xv).squeeze(-1).numpy()
            pv = (lv >= 0.0).astype(np.int64)
            y_va, g8_va = y_tr[va_i], g8_tr[va_i]
            wg_v = min(float(np.mean(pv[g8_va == gv] == y_va[g8_va == gv])) for gv in range(8))
            val_score = float(np.mean(pv == y_va)) if VARIANT == "base_erm" else wg_v
            if val_score > best_val_metric:
                best_val_metric = val_score
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # For Coordinator candidate: learn 4 demographic-group Bayes logit thresholds tau_d (d in 0..3) on inner-val!
    tau_dem = np.zeros(4, dtype=np.float64)
    if VARIANT == "coord_group_bayes_calib":
        with torch.no_grad():
            lv = net(xv).squeeze(-1).numpy().astype(np.float64)
        y_va, g_va, g8_va = y_tr[va_i], g_tr[va_i], g8_tr[va_i]
        # Exact per-demographic-group threshold equalizing TPR (y=1) and TNR (y=0) within each (SEX, race) group!
        for d_grp in range(4):
            m0 = (g_va == d_grp) & (y_va == 0)
            m1 = (g_va == d_grp) & (y_va == 1)
            best_td, best_min2 = 0.0, -1.0
            for td in np.linspace(-2.2, 2.2, 181):
                tnr = float(np.mean(lv[m0] < td))
                tpr = float(np.mean(lv[m1] >= td))
                m2 = min(tnr, tpr)
                if m2 > best_min2:
                    best_min2, best_td = m2, float(td)
            tau_dem[d_grp] = best_td

    outs = []
    for it in eval_items:
        X_raw_ev = np.asarray(it["X_eval"], dtype=np.float64)
        g_ev = ((X_raw_ev[:, sex_idx] > 1.5).astype(np.int64) * 2 + (X_raw_ev[:, race_idx] > 0.5).astype(np.int64))
        Xev_s = torch.tensor(((X_raw_ev - xm) / xs).astype(np.float32))
        with torch.no_grad():
            lev = net(Xev_s).squeeze(-1).numpy().astype(np.float64)
        preds = (lev >= tau_dem[g_ev]).astype(int).tolist()
        outs.append({"id": it["id"], "prediction": preds})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = fit_and_predict(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
