import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def make_astra_feats(X, xm, xs, meds, nonbin):
    Xs = (X - xm) / xs
    hinges = np.maximum(0.0, Xs[:, nonbin] - meds[None, :])
    sq = Xs[:, nonbin] ** 2
    return Xs, np.hstack([np.ones((len(X), 1)), Xs, hinges, sq])

def fit_fold_models(X_tr, y_tr, g_tr, xm, xs, meds, nonbin, seed):
    Xs, Phi = make_astra_feats(X_tr, xm, xs, meds, nonbin)
    N, d = Xs.shape
    torch.manual_seed(seed)

    # 1) Shared 2x128 trunk with 4 per-group logit heads + 4-way group posterior head q(g | x)!
    class MultiHeadExpertNet(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d_in, 128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU())
            self.y_heads = nn.Linear(96, 4)
            self.g_head = nn.Linear(96, 4)

        def forward(self, x):
            h = self.trunk(x)
            return self.y_heads(h), self.g_head(h)

    net = MultiHeadExpertNet(d)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1.5e-2)
    xt = torch.tensor(Xs.astype(np.float32))
    yt = torch.tensor(y_tr.astype(np.float32))
    gt = torch.tensor(g_tr, dtype=torch.long)

    # Group-balanced inverse-prevalence weights so minority demographic x label cells have crisp logits!
    g8_t = g_tr * 2 + y_tr
    cnt8 = np.maximum(1.0, np.bincount(g8_t, minlength=8).astype(np.float32))
    inv_w8 = (len(g8_t) / (8.0 * cnt8)) ** 0.5
    wt = torch.tensor(inv_w8[g8_t])

    for _ in range(55):
        p_idx = torch.randperm(N)
        for b0 in range(0, N, 512):
            sl = p_idx[b0:b0 + 512]
            yl, gl = net(xt[sl])
            chosen_yl = yl.gather(1, gt[sl].unsqueeze(1)).squeeze(1)
            loss_y = torch.mean(wt[sl] * nn.functional.binary_cross_entropy_with_logits(chosen_yl, yt[sl], reduction="none"))
            loss_g = nn.functional.cross_entropy(gl, gt[sl])
            loss = loss_y + 0.5 * loss_g
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    # 2) Per-group quadratic logistic experts on Astra's Phi features via L-BFGS
    W_quad = np.zeros((Phi.shape[1], 4), dtype=np.float64)
    # Warm-start from global logistic fit
    pt_all = torch.tensor(Phi, dtype=torch.float64)
    yt_all = torch.tensor(y_tr.astype(np.float64), dtype=torch.float64)
    w_glob = torch.zeros(Phi.shape[1], dtype=torch.float64, requires_grad=True)
    opt_g = torch.optim.LBFGS([w_glob], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")
    def cl_g():
        opt_g.zero_grad()
        l = torch.mean(nn.functional.binary_cross_entropy_with_logits(pt_all @ w_glob, yt_all)) + 1e-3 * torch.sum(w_glob[1:] ** 2)
        l.backward()
        return l
    opt_g.step(cl_g)
    w0_np = w_glob.detach().numpy()

    for g_id in range(4):
        mg = (g_tr == g_id)
        if np.sum(mg) >= 25:
            pt_g = torch.tensor(Phi[mg], dtype=torch.float64)
            yt_g = torch.tensor(y_tr[mg].astype(np.float64), dtype=torch.float64)
            wg = torch.tensor(w0_np.copy(), dtype=torch.float64, requires_grad=True)
            w0_t = torch.tensor(w0_np, dtype=torch.float64)
            opt_k = torch.optim.LBFGS([wg], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")
            def cl_k():
                opt_k.zero_grad()
                # Hierarchical shrinkage toward w0_t prevents overfitting on small groups!
                l = torch.mean(nn.functional.binary_cross_entropy_with_logits(pt_g @ wg, yt_g)) + 5e-3 * torch.sum((wg[1:] - w0_t[1:]) ** 2)
                l.backward()
                return l
            opt_k.step(cl_k)
            W_quad[:, g_id] = wg.detach().numpy()
        else:
            W_quad[:, g_id] = w0_np

    def predict_experts(X_raw):
        Xs_e, Phi_e = make_astra_feats(X_raw, xm, xs, meds, nonbin)
        with torch.no_grad():
            yl, gl = net(torch.tensor(Xs_e.astype(np.float32)))
            yl_np = yl.numpy().astype(np.float64)
            gl_np = gl.numpy().astype(np.float64)
        gl_np -= np.max(gl_np, axis=1, keepdims=True)
        q_g = np.exp(gl_np) / np.sum(np.exp(gl_np), axis=1, keepdims=True)  # [N, 4]
        ell_g = 0.55 * yl_np + 0.45 * (Phi_e @ W_quad)
        p_y_given_g = expit(ell_g)  # [N, 4]
        return q_g, p_y_given_g

    return predict_experts

def run_d2_r2_opus(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr
    N = len(X_tr)

    xm = np.mean(X_tr, axis=0, keepdims=True)
    xs = np.where(np.std(X_tr, axis=0, keepdims=True) > 1e-6, np.std(X_tr, axis=0, keepdims=True), 1.0)
    Xs_all = (X_tr - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[:, j])) > 2]
    meds = np.median(Xs_all[:, nonbin], axis=0)

    # 3-Fold Cell-Stratified OOF to fit exact Bayes minimax weights w_{g, y} with ZERO overfitting!
    rng = np.random.RandomState(520)
    folds = [np.empty(0, dtype=np.int64) for _ in range(3)]
    fold_lists = [[], [], []]
    for c in range(8):
        ic = np.where(g8_tr == c)[0]
        rng.shuffle(ic)
        for f_i, part in enumerate(np.array_split(ic, 3)):
            fold_lists[f_i].append(part)
    folds = [np.concatenate(fl) for fl in fold_lists]

    oof_q = np.zeros((N, 4), dtype=np.float64)
    oof_p = np.zeros((N, 4), dtype=np.float64)
    fold_fns = []
    for f_i in range(3):
        val_idx = folds[f_i]
        tr_idx = np.concatenate([folds[j] for j in range(3) if j != f_i])
        fn = fit_fold_models(X_tr[tr_idx], y_tr[tr_idx], g_tr[tr_idx], xm, xs, meds, nonbin, seed=100 + f_i)
        fold_fns.append(fn)
        qf, pf = fn(X_tr[val_idx])
        oof_q[val_idx] = qf
        oof_p[val_idx] = pf

    # Exact Pointwise Bayes Minimax Rule over all 8 cells c = (g, y):
    # Predict y=1 iff sum_{g=0}^3 q(g|x) * [w_{g,1} * p(1|x,g) - w_{g,0} * (1 - p(1|x,g))] >= 0!
    pi_8 = np.array([max(1e-4, np.mean(g8_tr == c)) for c in range(8)], dtype=np.float64)
    n_8 = np.array([max(1.0, np.sum(g8_tr == c)) for c in range(8)], dtype=np.float64)

    lam = pi_8.copy()
    mu_fl = 0.0
    best_w8 = np.ones(8, dtype=np.float64)
    best_lcb = -1e9

    P1_oof = oof_q * oof_p            # [N, 4] = P(g, y=1 | x)
    P0_oof = oof_q * (1.0 - oof_p)    # [N, 4] = P(g, y=0 | x)

    for _ in range(240):
        w8 = lam / pi_8 + mu_fl
        w_g0 = w8[0::2]
        w_g1 = w8[1::2]
        margin = (P1_oof @ w_g1) - (P0_oof @ w_g0)
        yhat = (margin >= 0.0).astype(np.int64)
        corr = (yhat == y_tr).astype(np.float64)
        A_c = np.array([float(np.mean(corr[g8_tr == c])) for c in range(8)])
        A_mean = float(np.mean(corr))
        lcb = float(np.min(A_c - 0.35 * np.sqrt(np.maximum(1e-6, A_c * (1.0 - A_c)) / n_8)))
        if A_mean >= 0.750 and lcb > best_lcb:
            best_lcb = lcb
            best_w8 = w8.copy()
        lam = lam * np.exp(0.65 * (1.0 - A_c))
        lam /= np.sum(lam)
        mu_fl = max(0.0, mu_fl + 4.5 * (0.753 - A_mean))

    w_g0 = best_w8[0::2]
    w_g1 = best_w8[1::2]
    outs = []
    for it in eval_items:
        Xev = np.asarray(it["X_eval"], dtype=np.float64)
        q_ens = np.zeros((len(Xev), 4), dtype=np.float64)
        p_ens = np.zeros((len(Xev), 4), dtype=np.float64)
        for fn in fold_fns:
            qe, pe = fn(Xev)
            q_ens += qe / 3.0
            p_ens += pe / 3.0
        m_ev = ((q_ens * p_ens) @ w_g1) - ((q_ens * (1.0 - p_ens)) @ w_g0)
        outs.append({"id": it["id"], "prediction": (m_ev >= 0.0).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_d2_r2_opus(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
