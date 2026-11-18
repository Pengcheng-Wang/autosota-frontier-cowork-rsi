import argparse, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
MODE = 'spectral_poly_minimax'

def run_d2_bc(bundle, eval_items):
    X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
    y_tr = np.asarray(bundle["y_train"], dtype=np.int64)
    g_tr = np.asarray(bundle["g_train"], dtype=np.int64)
    sex_idx = int(bundle["sex_idx"])
    race_idx = int(bundle["race_idx"])
    g8_tr = g_tr * 2 + y_tr

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_tr))
    va_i, tr_i = perm[:2800], perm[2800:]

    xm = np.mean(X_tr[tr_i], axis=0, keepdims=True)
    xs = np.where(np.std(X_tr[tr_i], axis=0, keepdims=True) > 1e-6, np.std(X_tr[tr_i], axis=0, keepdims=True), 1.0)
    Xtr_s = (X_tr[tr_i] - xm) / xs
    Xva_s = (X_tr[va_i] - xm) / xs
    ytr = y_tr[tr_i]
    yva = y_tr[va_i]
    gtr, gva = g_tr[tr_i], g_tr[va_i]
    g8tr, g8va = g8_tr[tr_i], g8_tr[va_i]

    if MODE == "spectral_poly_minimax":
        # Degree-2 interaction + quadratic expansion + primal-dual exponential/logistic minimax over 8 groups!
        def poly2(Z):
            d = Z.shape[1]
            quad = Z ** 2
            cross = [Z[:, i:i+1] * Z[:, j:j+1] for i in range(d) for j in range(i + 1, d)]
            return np.hstack([Z, quad, *cross, np.ones((len(Z), 1))])

        Ptr = poly2(Xtr_s)
        Pva = poly2(Xva_s)
        pm = np.mean(Ptr[:, :-1], axis=0, keepdims=True)
        ps = np.where(np.std(Ptr[:, :-1], axis=0, keepdims=True) > 1e-6, np.std(Ptr[:, :-1], axis=0, keepdims=True), 1.0)
        Ptr[:, :-1] = (Ptr[:, :-1] - pm) / ps
        Pva[:, :-1] = (Pva[:, :-1] - pm) / ps

        D_p = Ptr.shape[1]
        theta = np.zeros(D_p, dtype=np.float64)
        q_g = np.ones(8, dtype=np.float64) / 8.0
        y_pm = 2.0 * ytr - 1.0
        m_a, v_a = np.zeros(D_p), np.zeros(D_p)
        best_wg, best_theta = -1.0, theta.copy()

        for t in range(1, 221):
            z = np.clip(y_pm * (Ptr @ theta), -20.0, 20.0)
            # Smooth surrogate of 0-1 group error: log(1 + exp(-z))
            loss_i = np.log1p(np.exp(-z))
            dl_i = -y_pm / (1.0 + np.exp(z))
            lg = np.zeros(8)
            g_tot = 0.25 * (Ptr.T @ dl_i) / len(Ptr)
            for gv in range(8):
                mg = (g8tr == gv)
                lg[gv] = float(np.mean(loss_i[mg]))
                g_tot += 0.75 * q_g[gv] * ((Ptr[mg].T @ dl_i[mg]) / np.sum(mg))
            q_g *= np.exp(0.08 * lg)
            q_g /= np.sum(q_g)
            g_tot[:-1] += 3e-3 * theta[:-1]
            m_a = 0.9 * m_a + 0.1 * g_tot
            v_a = 0.999 * v_a + 0.001 * (g_tot ** 2)
            theta -= 0.015 * (m_a / (1 - 0.9 ** t)) / (np.sqrt(v_a / (1 - 0.999 ** t)) + 1e-8)

            if t % 20 == 0:
                pv = (Pva @ theta >= 0.0).astype(np.int64)
                if np.mean(pv == yva) >= 0.745:
                    wg = min(float(np.mean(pv[g8va == gv] == yva[g8va == gv])) for gv in range(8))
                    if wg > best_wg:
                        best_wg, best_theta = wg, theta.copy()

        outs = []
        for it in eval_items:
            Xev = (np.asarray(it["X_eval"], dtype=np.float64) - xm) / xs
            Pev = poly2(Xev)
            Pev[:, :-1] = (Pev[:, :-1] - pm) / ps
            outs.append({"id": it["id"], "prediction": (Pev @ best_theta >= 0.0).astype(int).tolist()})
        return outs

    # Explorer: Accuracy-deficit GroupDRO MLP + Temperature Calibration + Hierarchical Shrunk Group Thresholds!
    torch.manual_seed(142)
    net = nn.Sequential(nn.Linear(Xtr_s.shape[1], 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(ytr.astype(np.float32))
    g8t = torch.tensor(g8tr, dtype=torch.long)
    w_g = torch.ones(8) / 8.0

    for ep in range(130):
        net.train()
        p_i = torch.randperm(len(xt))
        for b0 in range(0, len(xt), 1024):
            sl = p_i[b0:b0 + 1024]
            lg = net(xt[sl]).squeeze(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(lg, yt[sl], reduction="none")
            acc_b = ((lg >= 0.0).float() == yt[sl]).float()
            gl = torch.zeros(8)
            ga = torch.ones(8) * 0.75
            for gv in range(8):
                m = (g8t[sl] == gv)
                if torch.any(m):
                    gl[gv] = torch.mean(bce[m])
                    ga[gv] = torch.mean(acc_b[m])
            with torch.no_grad():
                w_g = w_g * torch.exp(0.08 * (0.76 - ga))
                w_g = w_g / torch.sum(w_g)
            loss = 0.8 * torch.sum(w_g * gl) + 0.2 * torch.mean(bce)
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        l_va = net(torch.tensor(Xva_s, dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)

    # Global threshold + hierarchical group shrinkage tau_d = 0.75 * tau_d_opt + 0.25 * tau_global
    best_glob, best_g_sc = 0.0, -1.0
    for tg in np.linspace(-1.5, 1.5, 121):
        pv = (l_va >= tg).astype(np.int64)
        if np.mean(pv == yva) >= 0.745:
            wg = min(float(np.mean(pv[g8va == gv] == yva[g8va == gv])) for gv in range(8))
            if wg > best_g_sc:
                best_g_sc, best_glob = wg, float(tg)

    tau_d = np.full(4, best_glob)
    for dg in range(4):
        m0 = (gva == dg) & (yva == 0)
        m1 = (gva == dg) & (yva == 1)
        bt, bm = best_glob, -1.0
        for td in np.linspace(-2.0, 2.0, 161):
            sc = min(float(np.mean(l_va[m0] < td)), float(np.mean(l_va[m1] >= td)))
            if sc > bm:
                bm, bt = sc, float(td)
        tau_d[dg] = 0.80 * bt + 0.20 * best_glob

    outs = []
    for it in eval_items:
        Xr = np.asarray(it["X_eval"], dtype=np.float64)
        gev = ((Xr[:, sex_idx] > 1.5).astype(np.int64) * 2 + (Xr[:, race_idx] > 0.5).astype(np.int64))
        with torch.no_grad():
            lev = net(torch.tensor(((Xr - xm) / xs).astype(np.float32))).squeeze(-1).numpy()
        outs.append({"id": it["id"], "prediction": (lev >= tau_d[gev]).astype(int).tolist()})
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    bundle = json.loads(open(req["train_bundle_path"], encoding="utf-8").read())
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": run_d2_bc(bundle, req["items"])}, f)

if __name__ == "__main__":
    main()
