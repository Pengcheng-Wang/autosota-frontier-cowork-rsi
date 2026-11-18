import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VARIANT = 'autosota_cmnn_150ep'

class MonotonicLinearPaper(nn.Linear):
    def __init__(self, in_f, out_f, pre_act=nn.Identity()):
        super().__init__(in_f, out_f)
        self.act = pre_act
    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        return F.linear(self.act(x), w_pos, self.bias)

class PaperCMNN(nn.Module):
    def __init__(self, d_free, d_mono, hidden=32):
        super().__init__()
        self.free_net = nn.Sequential(nn.Linear(max(1, d_free), hidden), nn.ReLU(), nn.Linear(hidden, hidden)) if d_free > 0 else None
        in_m = d_mono + (hidden if d_free > 0 else 0)
        self.m1 = MonotonicLinearPaper(in_m, hidden, nn.Identity())
        self.m2 = MonotonicLinearPaper(hidden, hidden, nn.ReLU())
        self.m3 = MonotonicLinearPaper(hidden, 1, nn.ReLU())
        self.d_free = d_free
        self.hidden = hidden

    def forward(self, xf, xm):
        # Note: in Paper #77, free features pass through unconstrained pre_mono, then enter m1 whose weights on BOTH free_emb and xm are clamped >=0!
        if self.d_free > 0:
            hf = self.free_net(xf)
            z = torch.cat([hf, xm], dim=1)
        else:
            z = xm
        return self.m3(self.m2(self.m1(z))).squeeze(-1)

class DecisionCondHingeNet(nn.Module):
    """Proposer D: f(x) = h(x_free) + sum_{j,k} softplus(a_{jk}(x_free)) * softplus(x_{mono,j} - t_{jk})."""
    def __init__(self, d_free, d_mono, knots: torch.Tensor):
        super().__init__()
        self.d_free = d_free
        self.d_mono = d_mono
        self.register_buffer("knots", knots)  # [d_mono, K]
        K = knots.shape[1]
        if d_free > 0:
            self.backbone = nn.Sequential(nn.Linear(d_free, 48), nn.ReLU())
            self.head_h = nn.Linear(48, 1)
            self.head_a = nn.Linear(48, d_mono * K)
        else:
            self.h0 = nn.Parameter(torch.zeros(1))
            self.a0 = nn.Parameter(torch.zeros(d_mono * K))
        self.K = K

    def forward(self, xf, xm):
        basis = F.softplus(xm.unsqueeze(-1) - self.knots.unsqueeze(0))  # [N, d_mono, K]
        if self.d_free > 0:
            feat = self.backbone(xf)
            h = self.head_h(feat).squeeze(-1)
            a = F.softplus(self.head_a(feat)).view(-1, self.d_mono, self.K)
        else:
            h = self.h0.expand(len(xm))
            a = F.softplus(self.a0).view(1, self.d_mono, self.K)
        return h + torch.sum(a * basis, dim=(1, 2))

class CoordConvexConcaveMonotoneNet(nn.Module):
    """Proposer A: Convex + Concave + Sigmoid + Positive-Skip Monotone Basis + Positive-Weight Interaction Layer."""
    def __init__(self, d_free, d_mono, knots: torch.Tensor):
        super().__init__()
        self.d_free = d_free
        self.d_mono = d_mono
        self.register_buffer("knots", knots)  # [d_mono, K]
        K = knots.shape[1]
        n_basis = 3 * K + 1  # convex + concave + sigmoid + linear per monotone dim!
        if d_free > 0:
            self.free_mlp = nn.Sequential(nn.Linear(d_free, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
            self.base_out = nn.Linear(48, 1)
            self.gate_w = nn.Linear(48, d_mono * n_basis)
            self.inter_shift = nn.Linear(48, 24)
        else:
            self.b0 = nn.Parameter(torch.zeros(1))
            self.g0 = nn.Parameter(torch.zeros(d_mono * n_basis))
            self.s0 = nn.Parameter(torch.zeros(24))
        self.inter_pos_w = nn.Parameter(torch.randn(24, d_mono) * 0.1 - 1.5)
        self.inter_out_w = nn.Parameter(torch.randn(24) * 0.1 - 1.5)
        self.n_basis = n_basis

    def forward(self, xf, xm):
        diff = xm.unsqueeze(-1) - self.knots.unsqueeze(0)  # [N, d_mono, K]
        b_conv = F.softplus(diff)
        b_conc = -F.softplus(-diff)
        b_sig = torch.sigmoid(1.5 * diff)
        b_lin = xm.unsqueeze(-1)
        phi = torch.cat([b_conv, b_conc, b_sig, b_lin], dim=-1)  # [N, d_mono, 3K+1]

        if self.d_free > 0:
            h = self.free_mlp(xf)
            f0 = self.base_out(h).squeeze(-1)
            w_pos = F.softplus(self.gate_w(h)).view(-1, self.d_mono, self.n_basis) / self.n_basis
            shift = self.inter_shift(h)
        else:
            f0 = self.b0.expand(len(xm))
            w_pos = F.softplus(self.g0).view(1, self.d_mono, self.n_basis) / self.n_basis
            shift = self.s0.unsqueeze(0).expand(len(xm), -1)

        add_mono = torch.sum(w_pos * phi, dim=(1, 2))
        # Multi-dimensional monotone synergistic interaction via positive weights + tanh saturation (monotone non-decreasing!)
        W1 = F.softplus(self.inter_pos_w) / max(1, self.d_mono)
        w2 = F.softplus(self.inter_out_w) / 24.0
        inter_term = torch.tanh(xm @ W1.T + shift) @ w2
        return f0 + add_mono + inter_term

def train_predict_one(Xf_tr, Xm_tr, y_tr, seed, eval_batches):
    rng = np.random.RandomState(100 + seed)
    perm = rng.permutation(len(y_tr))
    n_va = max(25, int(0.18 * len(y_tr)))
    va_i, tr_i = perm[:n_va], perm[n_va:]

    if VARIANT == "comp_monotone_ridge_spline":
        # Projected Non-Negative Ridge on I-spline / hinge monotone basis + unconstrained Ridge on free features
        qs = np.quantile(Xm_tr[tr_i], np.linspace(0.1, 0.9, 7), axis=0).T  # [d_mono, 7]
        def feat_m(Xm):
            d = Xm[:, :, None] - qs[None, :, :]
            return np.hstack([Xm, np.maximum(0.0, d).reshape(len(Xm), -1), (-np.maximum(0.0, -d)).reshape(len(Xm), -1)])
        Pm_tr, Pm_va = feat_m(Xm_tr[tr_i]), feat_m(Xm_tr[va_i])
        Pf_tr = np.hstack([Xf_tr[tr_i], np.ones((len(tr_i), 1))])
        Pf_va = np.hstack([Xf_tr[va_i], np.ones((len(va_i), 1))])
        # Alternating exact Ridge on free + projected non-negative GD on monotone features
        wf = np.zeros(Pf_tr.shape[1])
        wm = np.zeros(Pm_tr.shape[1])
        Lf = np.linalg.inv(Pf_tr.T @ Pf_tr + 2.0 * np.eye(Pf_tr.shape[1])) @ Pf_tr.T
        step_m = 0.5 / (np.linalg.norm(Pm_tr, ord=2) ** 2 + 2.0)
        for _ in range(120):
            wf = Lf @ (y_tr[tr_i] - Pm_tr @ wm)
            res = Pf_tr @ wf + Pm_tr @ wm - y_tr[tr_i]
            wm = np.maximum(0.0, wm - step_m * (Pm_tr.T @ res + 1.0 * wm))
        outs = {}
        for bid, Xfe, Xme in eval_batches:
            Pfe = np.hstack([Xfe, np.ones((len(Xfe), 1))])
            Pme = feat_m(Xme)
            outs[bid] = (Pfe @ wf + Pme @ wm).tolist()
        return outs

    torch.manual_seed(1000 + seed)
    d_free, d_mono = Xf_tr.shape[1], Xm_tr.shape[1]
    qs = torch.tensor(np.quantile(Xm_tr[tr_i], np.linspace(0.1, 0.9, 8), axis=0).T, dtype=torch.float32)

    if VARIANT in ("paper_cmnn_100ep", "autosota_cmnn_150ep"):
        model = PaperCMNN(d_free, d_mono, hidden=32)
        epochs = 150 if VARIANT == "autosota_cmnn_150ep" else 100
        lr, wd = 1.5e-3, 1e-4
    elif VARIANT == "decision_cond_hinges":
        model = DecisionCondHingeNet(d_free, d_mono, qs)
        epochs, lr, wd = 110, 2.5e-3, 5e-3
    elif VARIANT == "coord_convex_concave":
        model = CoordConvexConcaveMonotoneNet(d_free, d_mono, qs)
        epochs, lr, wd = 120, 2.5e-3, 8e-3
    else:
        raise ValueError(VARIANT)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xft = torch.tensor(Xf_tr[tr_i], dtype=torch.float32)
    xmt = torch.tensor(Xm_tr[tr_i], dtype=torch.float32)
    yt = torch.tensor(y_tr[tr_i], dtype=torch.float32)
    xfv = torch.tensor(Xf_tr[va_i], dtype=torch.float32)
    xmv = torch.tensor(Xm_tr[va_i], dtype=torch.float32)
    yv = torch.tensor(y_tr[va_i], dtype=torch.float32)

    best_va = 1e18
    best_st = None
    bs = min(256, len(tr_i))
    for ep in range(epochs):
        model.train()
        perm_b = torch.randperm(len(tr_i))
        for b0 in range(0, len(tr_i), bs):
            sl = perm_b[b0:b0 + bs]
            pred = model(xft[sl], xmt[sl])
            loss = torch.mean((pred - yt[sl]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(torch.mean((model(xfv, xmv) - yv) ** 2).item())
            if vl < best_va:
                best_va = vl
                best_st = {k: v.clone() for k, v in model.state_dict().items()}

    if best_st is not None:
        model.load_state_dict(best_st)
    model.eval()
    outs = {}
    with torch.no_grad():
        for bid, Xfe, Xme in eval_batches:
            p = model(torch.tensor(Xfe, dtype=torch.float32), torch.tensor(Xme, dtype=torch.float32)).numpy()
            outs[bid] = p.tolist()
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    req = json.loads(open(args.input_json, encoding="utf-8").read())
    tr_shared = json.loads(open(req["train_datasets_path"], encoding="utf-8").read())

    # Group eval items by (dataset, seed) so we train once per (dataset, seed) instead of per block!
    groups = {}
    for it in req["items"]:
        key = (it["dataset"], int(it["seed"]))
        groups.setdefault(key, []).append((it["id"], np.asarray(it["X_eval_free"], dtype=np.float64), np.asarray(it["X_eval_mono"], dtype=np.float64)))

    all_preds = {}
    for (ds, sd), batches in groups.items():
        ds_tr = tr_shared[ds]
        res = train_predict_one(
            np.asarray(ds_tr["X_train_free"], dtype=np.float64),
            np.asarray(ds_tr["X_train_mono"], dtype=np.float64),
            np.asarray(ds_tr["y_train"], dtype=np.float64),
            sd,
            batches,
        )
        all_preds.update(res)

    preds_out = [{"id": it["id"], "prediction": all_preds[it["id"]]} for it in req["items"]]
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds_out}, f)

if __name__ == "__main__":
    main()
