import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

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
        loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(xt @ w, yt)) + 0.5 * reg * torch.sum(w[1:] ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()

def run_d2_r2_astra(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    rng = np.random.RandomState(17)
    perm = rng.permutation(len(X_tr))
    n_fit = int(0.68 * len(X_tr))
    n_cal = int(0.86 * len(X_tr))
    i_fit, i_cal, i_aud = perm[:n_fit], perm[n_fit:n_cal], perm[n_cal:]

    xm = np.mean(X_tr[i_fit], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[i_fit], axis=0, keepdims=True) > 1e-6, np.std(X_tr[i_fit], axis=0, keepdims=True), 1.0)
    Xf_s = (X_tr[i_fit] - xm) / xs
    nonbin = [j for j in range(X_tr.shape[1]) if len(np.unique(X_tr[i_fit, j])) > 2]
    meds = np.median(Xf_s[:, nonbin], axis=0)

    Phi_f = make_phi(X_tr[i_fit], xm, xs, meds, nonbin)
    Phi_c = make_phi(X_tr[i_cal], xm, xs, meds, nonbin)
    Phi_a = make_phi(X_tr[i_aud], xm, xs, meds, nonbin)

    w_erm = fit_logistic_lbfgs(Phi_f, y_tr[i_fit], reg=1e-3)
    w_erm /= max(1e-6, float(np.std(Phi_f @ w_erm)))
    s0_f = Phi_f @ w_erm

    G_inv = np.linalg.inv((Phi_f.T @ Phi_f) / len(Phi_f) + 0.01 * np.eye(Phi_f.shape[1]))
    r_f = 2.0 * y_tr[i_fit] - 1.0
    kern = np.exp(-0.5 * ((s0_f / 0.35) ** 2))
    g8_f = g8_tr[i_fit]
    U_dirs = np.zeros((Phi_f.shape[1], 8), dtype=np.float64)
    for c in range(8):
        mc = (g8_f == c)
        if np.any(mc):
            u = G_inv @ np.mean((r_f[mc] * kern[mc])[:, None] * Phi_f[mc], axis=0)
            rms = float(np.sqrt(np.mean((Phi_f @ u) ** 2)))
            U_dirs[:, c] = u / max(1e-8, rms)

    B_f = Phi_f @ U_dirs
    B_c = Phi_c @ U_dirs
    B_a = Phi_a @ U_dirs
    s0_c = Phi_c @ w_erm
    s0_a = Phi_a @ w_erm

    # Fit Astra linear+quad minimax direction al_hat on cal
    y_c, g8_c = y_tr[i_cal], g8_tr[i_cal]
    r_c = 2.0 * y_c - 1.0
    prm = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    Bt_t = torch.tensor(B_c, dtype=torch.float64)
    s0_tt = torch.tensor(s0_c, dtype=torch.float64)
    r_tt = torch.tensor(r_c, dtype=torch.float64)
    g_masks = [torch.tensor(g8_c == c, dtype=torch.bool) for c in range(8)]
    opt_p = torch.optim.LBFGS([prm], lr=0.8, max_iter=45, line_search_fn="strong_wolfe")
    def clos():
        opt_p.zero_grad()
        sc = s0_tt + Bt_t @ prm[:8] + prm[8]
        pc = torch.sigmoid(torch.clamp(r_tt * sc / 0.16, -25.0, 25.0))
        ca = torch.stack([torch.mean(pc[m]) for m in g_masks])
        loss = 0.05 * torch.log(torch.mean(torch.exp(-ca / 0.05))) + 8.0 * torch.clamp(0.758 - torch.mean(pc), min=0.0) ** 2 + 0.01 * torch.sum(prm[:8] ** 2)
        loss.backward()
        return loss
    opt_p.step(clos)
    al_hat = prm.detach().numpy()[:8]
    b0_hat = float(prm.detach().numpy()[8])

    sA_f = s0_f + B_f @ al_hat + b0_hat
    sA_c = s0_c + B_c @ al_hat + b0_hat
    sA_a = s0_a + B_a @ al_hat + b0_hat
    sig_m = max(0.25, float(np.median(np.abs(sA_f))))

    # Margin-gated GroupDRO width-32 MLP correction: s(x) = s_A(x) + gamma * exp(-|s_A| / sig_m) * f_MLP([Xs, B])!
    torch.manual_seed(52)
    Z_in_f = np.hstack([Xf_s, B_f]).astype(np.float32)
    mlp = nn.Sequential(nn.Linear(Z_in_f.shape[1], 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
    nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=2e-2)

    zt_f = torch.tensor(Z_in_f)
    sA_ft = torch.tensor(sA_f.astype(np.float32))
    gate_ft = torch.exp(-torch.abs(sA_ft) / sig_m)
    yt_f = torch.tensor(y_tr[i_fit].astype(np.float32))
    g8_ft = torch.tensor(g8_f, dtype=torch.long)
    pi_f = np.array([max(1e-4, np.mean(g8_f == c)) for c in range(8)], dtype=np.float32)
    q_c = torch.ones(8) / 8.0

    for ep in range(36):
        p_idx = torch.randperm(len(zt_f))
        for b0 in range(0, len(zt_f), 512):
            sl = p_idx[b0:b0 + 512]
            logits = sA_ft[sl] + gate_ft[sl] * mlp(zt_f[sl]).squeeze(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, yt_f[sl], reduction="none")
            gl = torch.zeros(8)
            for c in range(8):
                mc = (g8_ft[sl] == c)
                if torch.any(mc):
                    gl[c] = torch.mean(bce[mc])
            with torch.no_grad():
                q_c = q_c * torch.exp(0.08 * gl)
                q_c = q_c / torch.sum(q_c)
            loss = torch.sum(q_c * gl)
            opt_m.zero_grad(); loss.backward(); opt_m.step()
    mlp.eval()

    def mlp_corr(X_raw, B_mat, sA_vec):
        Xs = (X_raw - xm) / xs
        Zi = torch.tensor(np.hstack([Xs, B_mat]).astype(np.float32))
        with torch.no_grad():
            f_out = mlp(Zi).squeeze(-1).numpy().astype(np.float64)
        return np.exp(-np.abs(sA_vec) / sig_m) * f_out

    m_c = mlp_corr(X_tr[i_cal], B_c, sA_c)
    m_a = mlp_corr(X_tr[i_aud], B_a, sA_a)
    y_a, g8_a = y_tr[i_aud], g8_tr[i_aud]

    best_cfg = (0.0, 0.0)
    best_wg = -1.0
    for gam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.35):
        sc_c = sA_c + gam * m_c
        sc_a = sA_a + gam * m_a
        for cut in np.quantile(sc_c, np.linspace(0.18, 0.82, 65)):
            prc = (sc_c >= cut).astype(np.int64)
            if np.mean(prc == y_c) < 0.750:
                continue
            pra = (sc_a >= cut).astype(np.int64)
            acc_a = float(np.mean(pra == y_a))
            if acc_a < 0.748:
                continue
            wgc = min(float(np.mean(prc[g8_c == c] == y_c[g8_c == c])) for c in range(8))
            wga = min(float(np.mean(pra[g8_a == c] == y_a[g8_a == c])) for c in range(8))
            score = 0.5 * (wgc + wga)
            if score > best_wg:
                best_wg = score
                best_cfg = (gam, float(cut))

    gam_fin, cut_fin = best_cfg
    outs = []
    for it in eval_items:
        Xev = np.asarray(it["X_eval"], dtype=np.float64)
        Phi_e = make_phi(Xev, xm, xs, meds, nonbin)
        B_e = Phi_e @ U_dirs
        sA_e = (Phi_e @ w_erm) + B_e @ al_hat + b0_hat
        sc_e = sA_e + gam_fin * mlp_corr(Xev, B_e, sA_e)
        outs.append({"id": it["id"], "prediction": (sc_e >= cut_fin).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_d2_r2_astra(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
