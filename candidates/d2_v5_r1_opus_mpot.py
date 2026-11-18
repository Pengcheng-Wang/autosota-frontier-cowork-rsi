import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

def fit_multiclass_l2(X, y, n_classes, reg=1e-3):
    N, d = X.shape
    xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    lin = nn.Linear(d, n_classes, dtype=torch.float64)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = lin(xt)
        loss = nn.functional.cross_entropy(logits, yt) + 0.5 * reg * torch.sum(lin.weight ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    W_hat = lin.weight.detach().numpy().T  # [d, C]
    b_hat = lin.bias.detach().numpy()      # [C]

    def pred_proba(Xe):
        lg = Xe @ W_hat + b_hat[None, :]
        lg -= np.max(lg, axis=1, keepdims=True)
        ex = np.exp(lg)
        return ex / np.sum(ex, axis=1, keepdims=True)

    return pred_proba

def run_mpot(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    g8_tr = g_tr * 2 + y_tr

    # Cell-stratified 75/25 fit/cal split
    rng = np.random.RandomState(502)
    fit_idx, cal_idx = [], []
    for c in range(8):
        idx_c = np.where(g8_tr == c)[0]
        rng.shuffle(idx_c)
        n_f = int(0.75 * len(idx_c))
        fit_idx.append(idx_c[:n_f])
        cal_idx.append(idx_c[n_f:])
    fit_i = np.concatenate(fit_idx)
    cal_i = np.concatenate(cal_idx)

    xm = np.mean(X_tr[fit_i], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[fit_i], axis=0, keepdims=True) > 1e-6, np.std(X_tr[fit_i], axis=0, keepdims=True), 1.0)
    Xf = (X_tr[fit_i] - xm) / xs
    Xc = (X_tr[cal_i] - xm) / xs
    yf, yc = y_tr[fit_i], y_tr[cal_i]
    gf, gc = g_tr[fit_i], g_tr[cal_i]
    g8c = g8_tr[cal_i]

    # Fit smooth ERM MLP logit model l(x) in PyTorch (fast, python -I native!)
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(Xf.shape[1], 96), nn.SiLU(), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-2)
    xt = torch.tensor(Xf.astype(np.float32))
    yt = torch.tensor(yf.astype(np.float32))
    for _ in range(95):
        p_idx = torch.randperm(len(xt))
        for b0 in range(0, len(xt), 1024):
            sl = p_idx[b0:b0 + 1024]
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[sl]).squeeze(-1), yt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    # Fit 4-class L2 softmax group posterior q(g | x, y) via SciPy L-BFGS-B
    XY_f = np.hstack([Xf, yf[:, None], Xf * (2.0 * yf[:, None] - 1.0)])
    q_proba_fn = fit_multiclass_l2(XY_f, gf, n_classes=4, reg=1e-3)

    def get_logit_and_q(X_s):
        with torch.no_grad():
            ell = net(torch.tensor(X_s.astype(np.float32))).squeeze(-1).numpy().astype(np.float64)
        n_r = len(X_s)
        XY0 = np.hstack([X_s, np.zeros((n_r, 1)), -X_s])
        XY1 = np.hstack([X_s, np.ones((n_r, 1)), X_s])
        q0 = np.clip(q_proba_fn(XY0), 1e-5, 1.0)
        q1 = np.clip(q_proba_fn(XY1), 1e-5, 1.0)
        return ell, q0, q1

    ell_c, q0_c, q1_c = get_logit_and_q(Xc)
    pi_hat = np.array([max(1e-4, np.mean(g8_tr == c)) for c in range(8)], dtype=np.float64)
    n_c = np.array([max(1.0, np.sum(g8c == c)) for c in range(8)], dtype=np.float64)

    lam = np.ones(8, dtype=np.float64) / 8.0
    mu_fl = 0.0
    best_a = np.ones(8, dtype=np.float64)
    best_lcb = -1e9

    for _ in range(250):
        a = lam / pi_hat + mu_fl
        a_g0 = a[0::2]
        a_g1 = a[1::2]
        t_a = np.log(q0_c @ a_g0) - np.log(q1_c @ a_g1)
        yhat = (ell_c > t_a).astype(np.int64)
        corr = (yhat == yc).astype(np.float64)
        A_c = np.array([float(np.mean(corr[g8c == c])) for c in range(8)])
        A_mean = float(np.mean(corr))
        lcb = float(np.min(A_c - 0.5 * np.sqrt(np.maximum(1e-6, A_c * (1.0 - A_c)) / n_c)))
        if A_mean >= 0.750 and lcb > best_lcb:
            best_lcb = lcb
            best_a = a.copy()
        lam = lam * np.exp(-2.2 * A_c)
        lam /= np.sum(lam)
        mu_fl = max(0.0, mu_fl + 5.0 * (0.754 - A_mean))

    a_g0 = best_a[0::2]
    a_g1 = best_a[1::2]
    outs = []
    for it in eval_items:
        Xev = (np.asarray(it["X_eval"], dtype=np.float64) - xm) / xs
        ell_e, q0_e, q1_e = get_logit_and_q(Xev)
        t_e = np.log(q0_e @ a_g0) - np.log(q1_e @ a_g1)
        outs.append({"id": it["id"], "prediction": (ell_e > t_e).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    preds = run_mpot(bundle, req["items"])
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds}, f)

if __name__ == "__main__":
    main()
