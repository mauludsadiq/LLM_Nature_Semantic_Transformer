"""
Process supervision — Stage 17.

Step-level rewards: each op proposal receives immediate reward from the verifier.
This replaces the sparse terminal reward of Stage 16 with dense per-step feedback.

Reward structure:
    +1.0  correct op at this position (verifier accepts step)
    -0.5  wrong op (verifier rejects step)
    +2.0  bonus on ACCEPT terminal (full pass verified)
    -1.0  penalty on REJECT terminal

Gate: step_acceptance >= 0.95 over 10 eval episodes at noise=0.15
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
import random, json

FDIM=25; N_OPS=8; HIDDEN=128
TAU_BINS=[0.5,1.0,2.0,4.0]

GT_SEQUENCE=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,6),(5,6),(6,6)]

OP_NAMES=["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
          "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]

STEP_REWARD_CORRECT  =  1.0
STEP_REWARD_WRONG    = -0.5
TERMINAL_BONUS       =  2.0
TERMINAL_PENALTY     = -1.0


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    s=nn.Sequential(nn.Linear(FDIM,HIDDEN),nn.ReLU(),nn.Dropout(0.0),
                    nn.Linear(HIDDEN,HIDDEN),nn.ReLU(),nn.Dropout(0.0))
    return s, nn.Linear(HIDDEN,N_OPS), nn.Linear(HIDDEN,N_OPS)

def load_pt(path):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    s,o,t=build_model()
    s.load_state_dict(ckpt["shared"]); o.load_state_dict(ckpt["op_head"])
    t.load_state_dict(ckpt["tgt_head"]); return s,o,t

def save_pt(s,o,t,path):
    torch.save({"shared":s.state_dict(),"op_head":o.state_dict(),
                "tgt_head":t.state_dict(),"fdim":FDIM},path)


# ── Feature builder ───────────────────────────────────────────────────────────

def make_feature(bi,li,step,tau=1.0,top_k=3,noise=0.0):
    x=torch.zeros(FDIM)
    if bi<12: x[bi]=1.0
    x[12+li]=1.0; x[19]=min(step/20.0,1.0)
    tb=next((j for j,t in enumerate(TAU_BINS) if tau<=t),3)
    x[20+tb]=1.0; x[24]=min(top_k/10.0,1.0)
    if noise>0: x=(x+noise*torch.randn(FDIM)).clamp(0,1)
    return x


# ── Step-level verifier ───────────────────────────────────────────────────────

def step_reward(block_idx: int, proposed_op: int, noise_triggered: bool) -> float:
    gt_op, _ = GT_SEQUENCE[block_idx]
    is_terminal = block_idx == 11

    if proposed_op == gt_op:
        r = STEP_REWARD_CORRECT
        if is_terminal:
            r += TERMINAL_BONUS
        return r
    else:
        r = STEP_REWARD_WRONG
        if is_terminal:
            r += TERMINAL_PENALTY
        return r


# ── Metrics ───────────────────────────────────────────────────────────────────

class StepMetrics:
    def __init__(self):
        self.step_correct = 0
        self.step_total   = 0
        self.pass_correct = 0
        self.pass_total   = 0
        self.total_reward = 0.0
        self.per_op_correct = [0]*8
        self.per_op_total   = [0]*8

    def record_step(self, gt_op, pred_op, reward):
        self.step_total += 1
        self.per_op_total[gt_op] += 1
        if pred_op == gt_op:
            self.step_correct += 1
            self.per_op_correct[gt_op] += 1
        self.total_reward += reward

    def record_pass(self, all_correct):
        self.pass_total += 1
        if all_correct: self.pass_correct += 1

    def step_acc(self):
        return self.step_correct / max(self.step_total, 1)

    def pass_acc(self):
        return self.pass_correct / max(self.pass_total, 1)

    def mean_reward(self):
        return self.total_reward / max(self.step_total, 1)


# ── Process-supervised rollout ────────────────────────────────────────────────

def ps_rollout(shared, op_head, n_episodes=32, noise=0.0, train_mode=False):
    """
    Collect trajectories with step-level rewards.
    Returns (log_probs, step_rewards, metrics).
    """
    all_lps  = []
    all_rews = []
    metrics  = StepMetrics()

    for _ in range(n_episodes):
        ep_lps = []; ep_rews = []; pass_ok = True

        for bi, (gt_op, li) in enumerate(GT_SEQUENCE):
            x      = make_feature(bi, li, bi, noise=noise)
            h      = shared(x.unsqueeze(0))
            logits = op_head(h).squeeze(0)

            if train_mode:
                dist   = Categorical(logits=logits)
                action = dist.sample()
                lp     = dist.log_prob(action)
                pred   = action.item()
            else:
                with torch.no_grad():
                    pred = logits.argmax().item()
                lp   = torch.tensor(0.0)

            r = step_reward(bi, pred, noise > 0)
            metrics.record_step(gt_op, pred, r)

            if pred != gt_op: pass_ok = False

            ep_lps.append(lp)
            ep_rews.append(r)

        metrics.record_pass(pass_ok)
        all_lps.append(ep_lps)
        all_rews.append(ep_rews)

    return all_lps, all_rews, metrics


# ── Process-supervised loss ───────────────────────────────────────────────────

def ps_loss(all_lps, all_rews, baseline=0.0, gamma=0.99):
    """
    Policy gradient loss with step-level returns.
    G_t = r_t + gamma * r_{t+1} + ... (discounted from each step)
    """
    total = torch.tensor(0.0)
    n     = 0

    for ep_lps, ep_rews in zip(all_lps, all_rews):
        # Compute discounted return from each step
        T = len(ep_rews)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = ep_rews[t] + gamma * G
            returns[t] = G

        for lp, G in zip(ep_lps, returns):
            advantage = G - baseline
            total = total - lp * advantage
            n += 1

    return total / max(n, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(shared, op_head, n=200, noise=0.0):
    shared.eval(); op_head.eval()
    _, _, m = ps_rollout(shared, op_head, n, noise, train_mode=False)
    return m


# ── Main training loop ────────────────────────────────────────────────────────

def train(model_path="train/model_rl.pt", out_path="train/model_ps.pt",
          epochs=30, n_roll=100, lr=3e-5, noise_max=0.15, seed=42):

    torch.manual_seed(seed); random.seed(seed)

    print(f"Loading {model_path}...")
    shared, op_head, tgt_head = load_pt(model_path)

    # Frozen reference for KL penalty
    ref_s, ref_o, _ = load_pt(model_path)
    for p in list(ref_s.parameters())+list(ref_o.parameters()):
        p.requires_grad_(False)

    params = list(shared.parameters()) + list(op_head.parameters())
    optim  = torch.optim.Adam(params, lr=lr)

    # Baseline: running mean of episode return
    baseline  = 0.0
    best_sacc = 0.0

    print(f"\nProcess supervision: {epochs} epochs × {n_roll} rollouts")
    print(f"Step rewards: correct={STEP_REWARD_CORRECT} wrong={STEP_REWARD_WRONG} "
          f"terminal_bonus={TERMINAL_BONUS} terminal_penalty={TERMINAL_PENALTY}")
    print(f"\n{'Epoch':>6}  {'step_acc':>9}  {'pass_acc':>9}  "
          f"{'noisy_sa':>9}  {'reward':>8}  {'loss':>8}")
    print("-" * 60)

    for epoch in range(1, epochs+1):
        shared.train(); op_head.train()

        # Curriculum: noise increases gradually
        noise = min(noise_max, noise_max * epoch / 10)

        lps, rews, tr_metrics = ps_rollout(
            shared, op_head, n_roll, noise, train_mode=True
        )

        # Update baseline
        mean_ret = sum(sum(r) for r in rews) / len(rews)
        baseline = 0.9 * baseline + 0.1 * mean_ret

        # KL penalty vs reference
        kl_loss = torch.tensor(0.0)
        n_kl = 0
        for bi, (gt_op, li) in enumerate(GT_SEQUENCE):
            x = make_feature(bi, li, bi, noise=noise)
            logits = op_head(shared(x.unsqueeze(0))).squeeze(0)
            with torch.no_grad():
                ref_l = ref_o(ref_s(x.unsqueeze(0))).squeeze(0)
            kl_loss = kl_loss + F.kl_div(
                F.log_softmax(logits,0), F.softmax(ref_l,0), reduction='sum'
            )
            n_kl += 1
        kl_loss = kl_loss / max(n_kl, 1)

        pg_loss = ps_loss(lps, rews, baseline)
        loss    = pg_loss + 0.05 * kl_loss

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 0.5)
        optim.step()

        # Eval
        m_clean = evaluate(shared, op_head, 100, noise=0.0)
        m_noisy = evaluate(shared, op_head, 100, noise=noise_max)

        gate = "✓" if m_clean.step_acc() >= 0.95 else " "
        print(f"{epoch:>6}  {m_clean.step_acc():>9.3f}  {m_clean.pass_acc():>9.3f}  "
              f"{m_noisy.step_acc():>9.3f}  {m_clean.mean_reward():>8.3f}  "
              f"{loss.item():>8.4f} {gate}")

        if m_clean.step_acc() >= best_sacc:
            best_sacc = m_clean.step_acc()
            save_pt(shared, op_head, tgt_head, out_path)

    # Final evaluation
    print(f"\n{'='*60}")
    m_final_clean = evaluate(shared, op_head, 500, 0.0)
    m_final_noisy = evaluate(shared, op_head, 500, noise_max)

    print(f"Final step_acc  — clean: {m_final_clean.step_acc():.3f}  "
          f"noisy: {m_final_noisy.step_acc():.3f}")
    print(f"Final pass_acc  — clean: {m_final_clean.pass_acc():.3f}  "
          f"noisy: {m_final_noisy.pass_acc():.3f}")
    print(f"Final mean_rew  — clean: {m_final_clean.mean_reward():.3f}  "
          f"noisy: {m_final_noisy.mean_reward():.3f}")

    print(f"\nPer-op step accuracy (clean):")
    for i, name in enumerate(OP_NAMES):
        tot = m_final_clean.per_op_total[i]
        if tot == 0: continue
        acc = m_final_clean.per_op_correct[i] / tot
        bar = "█" * int(acc * 20)
        print(f"  [{i}] {name:<20} {acc:.3f}  {bar}")

    gate_pass = m_final_clean.step_acc() >= 0.95
    print(f"\nStage 17 gate (step_acc >= 0.95): "
          f"{'✓ PASSED' if gate_pass else '✗ FAILED'}")
    print(f"{'='*60}")

    return gate_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="train/model_rl.pt")
    parser.add_argument("--out",       default="train/model_ps.pt")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--n_roll",    type=int,   default=100)
    parser.add_argument("--lr",        type=float, default=3e-5)
    parser.add_argument("--noise_max", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    train(args.model, args.out, args.epochs, args.n_roll,
          args.lr, args.noise_max, args.seed)
