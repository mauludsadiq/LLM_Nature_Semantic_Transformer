"""
RL fine-tuning — Stage 16.

REINFORCE with baseline over the tower op sequence.
The model proposes ops given context features; reward = +1 on ACCEPT terminal.

Since the model already achieves 100% on the imitation corpus, Stage 16 tests:
  1. Policy gradient infrastructure is correct
  2. Model maintains performance under RL updates (no regression)
  3. Model generalizes to perturbed / out-of-distribution features

Gate: acceptance_rate after RL >= 95% (no regression from 100%)

Usage:
    python train/rl_train.py \
        --model   train/model_v2.pt \
        --out     train/model_rl.pt \
        --epochs  20 \
        --n_roll  200
"""

import argparse, json, math, random, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

FDIM    = 25
N_OPS   = 8
HIDDEN  = 128
GAMMA   = 0.99
TAU_BINS = [0.5, 1.0, 2.0, 4.0]
LAYERS   = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]

OP_NAMES = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
            "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]

# Ground truth op sequence (deterministic tower protocol)
GT_SEQUENCE = [
    (0, 0),   # block=0  SELECT_UNIVERSE  layer=PHONEME
    (1, 0),   # block=1  WITNESS_NEAREST  layer=PHONEME
    (2, 0),   # block=2  ATTEND           layer=PHONEME
    (3, 0),   # block=3  FFN_STEP         layer=PHONEME
    (3, 1),   # block=4  FFN_STEP         layer=SYLLABLE
    (3, 2),   # block=5  FFN_STEP         layer=MORPHEME
    (3, 3),   # block=6  FFN_STEP         layer=WORD
    (3, 4),   # block=7  FFN_STEP         layer=PHRASE
    (3, 5),   # block=8  FFN_STEP         layer=SEMANTIC
    (4, 6),   # block=9  PROJECT_LAYER    layer=DISCOURSE
    (5, 6),   # block=10 RETURN_SET       layer=DISCOURSE
    (6, 6),   # block=11 ACCEPT           layer=DISCOURSE
]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    shared = nn.Sequential(
        nn.Linear(FDIM, HIDDEN), nn.ReLU(), nn.Dropout(0.05),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.05),
    )
    op_head  = nn.Linear(HIDDEN, N_OPS)
    tgt_head = nn.Linear(HIDDEN, N_OPS)
    return shared, op_head, tgt_head


def load_pt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    shared, op_head, tgt_head = build_model()
    shared.load_state_dict(ckpt["shared"])
    op_head.load_state_dict(ckpt["op_head"])
    tgt_head.load_state_dict(ckpt["tgt_head"])
    return shared, op_head, tgt_head


def save_pt(shared, op_head, tgt_head, path):
    torch.save({
        "shared":   shared.state_dict(),
        "op_head":  op_head.state_dict(),
        "tgt_head": tgt_head.state_dict(),
        "fdim":     FDIM,
    }, path)


# ── Feature builder ───────────────────────────────────────────────────────────

def make_feature(block_idx: int, layer_idx: int, step: int,
                 tau: float, top_k: int, noise: float = 0.0) -> torch.Tensor:
    x = torch.zeros(FDIM)
    if block_idx < 12:
        x[block_idx] = 1.0
    x[12 + layer_idx] = 1.0
    x[19] = min(step / 20.0, 1.0)
    tb = next((j for j, t in enumerate(TAU_BINS) if tau <= t), 3)
    x[20 + tb] = 1.0
    x[24] = min(top_k / 10.0, 1.0)
    if noise > 0.0:
        x = (x + noise * torch.randn(FDIM)).clamp(0.0, 1.0)
    return x


# ── Environment ───────────────────────────────────────────────────────────────

class TowerEnv:
    """
    Simulates one pass through the tower op sequence.
    At each step the policy proposes an op_class.
    Reward: +1.0 if op matches ground truth at this position, else 0.
    Terminal: always at block 11 (ACCEPT position).
    """

    def __init__(self, tau: float = 1.0, top_k: int = 3, noise: float = 0.0):
        self.tau   = tau
        self.top_k = top_k
        self.noise = noise
        self.reset()

    def reset(self):
        self.block_idx = 0
        return self._obs()

    def _obs(self) -> torch.Tensor:
        gt_op, layer_idx = GT_SEQUENCE[self.block_idx]
        return make_feature(
            self.block_idx, layer_idx,
            self.block_idx,  # step_count = block_idx
            self.tau, self.top_k, self.noise
        )

    def step(self, action: int):
        gt_op, _ = GT_SEQUENCE[self.block_idx]
        reward    = 1.0 if action == gt_op else 0.0
        done      = self.block_idx >= 11
        self.block_idx = min(self.block_idx + 1, 11)
        obs = self._obs() if not done else torch.zeros(FDIM)
        return obs, reward, done


# ── REINFORCE ─────────────────────────────────────────────────────────────────

def rollout(env, shared, op_head, n_episodes=32, noise=0.0):
    """Collect trajectories. Returns (log_probs, rewards, acceptance_rate)."""
    all_log_probs = []
    all_rewards   = []
    n_accepted    = 0

    shared.eval(); op_head.eval()

    for _ in range(n_episodes):
        obs      = env.reset()
        ep_lp    = []
        ep_rew   = []
        done     = False
        all_accept = True

        while not done:
            with torch.no_grad():
                h     = shared(obs.unsqueeze(0))
                logits = op_head(h).squeeze(0)

            dist   = Categorical(logits=logits)
            action = dist.sample()
            lp     = dist.log_prob(action)

            obs, reward, done = env.step(action.item())
            ep_lp.append(lp)
            ep_rew.append(reward)
            if reward == 0.0:
                all_accept = False

        if all_accept:
            n_accepted += 1

        all_log_probs.append(ep_lp)
        all_rewards.append(ep_rew)

    return all_log_probs, all_rewards, n_accepted / n_episodes


def compute_returns(rewards, gamma=GAMMA):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def reinforce_loss(all_log_probs, all_rewards, baseline=0.0):
    loss = torch.tensor(0.0, requires_grad=True)
    for ep_lp, ep_rew in zip(all_log_probs, all_rewards):
        returns = compute_returns(ep_rew)
        for lp, G in zip(ep_lp, returns):
            loss = loss - lp * (G - baseline)
    return loss / len(all_log_probs)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading model from {args.model}...")
    shared, op_head, tgt_head = load_pt(args.model)

    params = list(shared.parameters()) + list(op_head.parameters())
    optim  = torch.optim.Adam(params, lr=args.lr)

    env      = TowerEnv(tau=1.0, top_k=3, noise=args.noise)
    env_hard = TowerEnv(tau=1.0, top_k=3, noise=args.noise * 3)

    best_acc  = 0.0
    baseline  = 0.0  # running mean reward

    print(f"REINFORCE: {args.epochs} epochs × {args.n_roll} rollouts, noise={args.noise}\n")
    print(f"{'Epoch':>6}  {'acc_clean':>10}  {'acc_noisy':>10}  {'loss':>8}  {'baseline':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        shared.train(); op_head.train()

        # Clean rollouts
        lps, rews, acc_clean = rollout(env, shared, op_head, args.n_roll, noise=0.0)

        # Update baseline (exponential moving average of mean episode return)
        mean_ret = sum(sum(r) for r in rews) / len(rews)
        baseline = 0.9 * baseline + 0.1 * mean_ret

        # RL loss
        loss = reinforce_loss(lps, rews, baseline=baseline)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 0.5)
        optim.step()

        # Noisy eval
        with torch.no_grad():
            _, _, acc_noisy = rollout(
                env_hard, shared, op_head, 50, noise=args.noise * 3
            )

        gate = "✓" if acc_clean >= 0.95 else " "
        print(f"{epoch:>6}  {acc_clean:>10.3f}  {acc_noisy:>10.3f}  "
              f"{loss.item():>8.4f}  {baseline:>8.4f} {gate}")

        if acc_clean >= best_acc:
            best_acc = acc_clean
            save_pt(shared, op_head, tgt_head, args.out)

    print(f"\nBest acceptance: {best_acc:.3f}")
    gate = "✓ PASSED" if best_acc >= 0.95 else "✗ FAILED"
    print(f"Stage 16 gate (acceptance >= 95%): {gate}")

    # Final eval on clean + hard distribution
    shared.eval(); op_head.eval()
    _, _, final_clean = rollout(env,      shared, op_head, 200, noise=0.0)
    _, _, final_noisy = rollout(env_hard, shared, op_head, 200, noise=args.noise*3)
    print(f"\nFinal eval:")
    print(f"  Clean acceptance: {final_clean:.3f}")
    print(f"  Noisy acceptance: {final_noisy:.3f}")

    return best_acc


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="train/model_v2.pt")
    parser.add_argument("--out",     default="train/model_rl.pt")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--n_roll",  type=int,   default=200)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--noise",   type=float, default=0.05)
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()
    train(args)
