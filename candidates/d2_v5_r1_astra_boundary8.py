import argparse, json
import numpy as np
import torch

torch.set_num_threads(1)

def expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def make_phi(X, xm, xs, meds, nonbin):
    Xs = (X - xm) / xs
    hinges = np.maximum(0.0, Xs[:, nonbin] - meds[None, :])
    sq = Xs[:, nonbin] ** 2
    return np.hstack([np.ones((len(X), 1)), Xs, hinges, sq])

def fit_logistic_lbfgs(Phi, y, reg=1e-3):
    xt = torch.tensor(Phi, dtype=torch.float64)
    yt = torch.tensor(y.astype(np.float64), dtype=torch.float64)
    w = torch.zeros(Phi.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=55, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = xt @ w
        loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(z, yt)) + 0.5 * reg * torch.sum(w[1:] ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()

def run_astra_d2(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    rng = np.random.RandomState(17)
    perm = rng.permutation(len(X_tr))
    n_fit = int(0.65 * len(X_tr))
    n_tune = int(0.85 * len(X_tr))
    i_fit, i_tune, i_aud = perm[:n_fit], perm[n_fit:n_tune], perm[n_tune:]

    xm = np.mean(X_tr[i_fit], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[i_fit], axis=0, keepdims=True) > 1e-6, np.std(X_tr[i_fit], axis=0, keepdims=True), 1.0)
    Xf_s = (X_tr[i_fit] - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[i_fit, j])) > 2]
    meds = np.median(Xf_s[:, nonbin], axis=0)

    Phi_f = make_phi(X_tr[i_fit], xm, xs, meds, nonbin)
    Phi_t = make_phi(X_tr[i_tune], xm, xs, meds, nonbin)
    Phi_a = make_phi(X_tr[i_aud], xm, xs, meds, nonbin)

    w_erm = fit_logistic_lbfgs(Phi_f, y_tr[i_fit], reg=1e-3)
    s_f = Phi_f @ w_erm
    sd_f = max(1e-6, float(np.std(s_f)))
    w_erm = w_erm / sd_f
    s_f = Phi_f @ w_erm

    G_mat = (Phi_f.T @ Phi_f) / len(Phi_f) + 0.01 * np.eye(Phi_f.shape[1])
    G_inv = np.linalg.inv(G_mat)
    r_f = 2.0 * y_tr[i_fit] - 1.0
    kern = np.exp(-0.5 * ((s_f / 0.35) ** 2))
    g8_f = g8_tr[i_fit]

    U_dirs = np.zeros((Phi_f.shape[1], 8), dtype=np.float64)
    for c in range(8):
        mc = (g8_f == c)
        if np.any(mc):
            rhs = np.mean((r_f[mc] * kern[mc])[:, None] * Phi_f[mc], axis=0)
            u = G_inv @ rhs
            rms = float(np.sqrt(np.mean((Phi_f @ u) ** 2)))
            if rms > 1e-8:
                u /= rms
            U_dirs[:, c] = u

    s0_t = Phi_t @ w_erm
    B_t = Phi_t @ U_dirs
    y_t, g8_t = y_tr[i_tune], g8_tr[i_tune]
    r_t = 2.0 * y_t - 1.0

    best_rule = (np.zeros(8), 0.0)
    best_aud_wg = -1.0

    for tau_temp in (0.12, 0.22):
        Bt_t = torch.tensor(B_t, dtype=torch.float64)
        s0_tt = torch.tensor(s0_t, dtype=torch.float64)
        r_tt = torch.tensor(r_t, dtype=torch.float64)
        g_masks = [torch.tensor(g8_t == c, dtype=torch.bool) for c in range(8)]
        prm = torch.zeros(9, dtype=torch.float64, requires_grad=True)
        opt_p = torch.optim.LBFGS([prm], lr=0.8, max_iter=45, line_search_fn="strong_wolfe")

        def closure_p():
            opt_p.zero_grad()
            al = prm[:8]
            b0 = prm[8]
            sc = s0_tt + Bt_t @ al + b0
            p_corr = torch.sigmoid(torch.clamp(r_tt * sc / tau_temp, -25.0, 25.0))
            c_accs = torch.stack([torch.mean(p_corr[m_c]) for m_c in g_masks])
            soft_wg = -0.05 * torch.log(torch.mean(torch.exp(-c_accs / 0.05)))
            mean_pen = 8.0 * torch.clamp(0.758 - torch.mean(p_corr), min=0.0) ** 2
            loss = -soft_wg + mean_pen + 0.01 * torch.sum(al ** 2)
            loss.backward()
            return loss

        opt_p.step(closure_p)
        al_hat = prm.detach().numpy()[:8]
        sc_t = s0_t + B_t @ al_hat
        sc_a = (Phi_a @ w_erm) + (Phi_a @ U_dirs) @ al_hat
        y_a, g8_a = y_tr[i_aud], g8_tr[i_aud]

        for cut in np.quantile(sc_t, np.linspace(0.15, 0.85, 81)):
            pr_t = (sc_t >= cut).astype(np.int64)
            if np.mean(pr_t == y_t) < 0.750:
                continue
            pr_a = (sc_a >= cut).astype(np.int64)
            acc_a = float(np.mean(pr_a == y_a))
            se_a = float(np.sqrt(acc_a * (1.0 - acc_a) / len(y_a)))
            if acc_a - 1.645 * se_a < 0.740:
                continue
            wg_t = min(float(np.mean(pr_t[g8_t == c] == y_t[g8_t == c])) for c in range(8))
            wg_a = min(float(np.mean(pr_a[g8_a == c] == y_a[g8_a == c])) for c in range(8))
            if 0.5 * (wg_t + wg_a) > best_aud_wg:
                best_aud_wg = 0.5 * (wg_t + wg_a)
                best_rule = (al_hat.copy(), float(cut))

    al_fin, cut_fin = best_rule
    outs = []
    for it in eval_items:
        Phi_e = make_phi(np.asarray(it["X_eval"], dtype=np.float64), xm, xs, meds, nonbin)
        sc_e = (Phi_e @ w_erm) + (Phi_e @ U_dirs) @ al_fin
        outs.append({"id": it["id"], "prediction": (sc_e >= cut_fin).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_astra_d2(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
