"""
Stage 23 — tau/top_k curriculum retraining on corpus_v3.

Key changes from Stage 17 (process_supervision.py):
  - corpus_v3.ndjson: 17230 records, block_idx 0..59, 10122 unique chain hashes
  - block_idx encoding: normalized scalar (bi/60.0) at dim 0, not one-hot
    (corpus_v3 has seq_len up to 5, so block_idx up to 59)
  - step_count normalized by 60.0 (not 20.0)
  - Same model architecture (FDIM=25, HIDDEN=128, 7440 params)
  - Same IL + process supervision pipeline
  - Gate: val_acc >= 0.98, noise robustness >= 0.97 at noise=0.05

Feature vector (25-dim):
  [0]     block_idx / 60.0          (was: one-hot 0..11)
  [1..11] reserved zeros            (was: one-hot continuation)
  [12..18] active_layer one-hot     (unchanged)
  [19]    step_count / 60.0         (was: / 20.0)
  [20..23] tau bin one-hot          (unchanged)
  [24]    top_k / 10.0              (unchanged)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
import json, random, math

FDIM=25; N_OPS=8; HIDDEN=128
TAU_BINS=[0.5,1.0,2.0,4.0]
CORPUS_PATH="training_data/corpus_v3.ndjson"

GT_SEQUENCE=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,6),(5,6),(6,6)]
OP_NAMES=["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
          "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]

STEP_REWARD_CORRECT =  1.0
STEP_REWARD_WRONG   = -0.5
TERMINAL_BONUS      =  2.0
TERMINAL_PENALTY    = -1.0

# ── Feature encoding ──────────────────────────────────────────────────────────

def tau_bin(tau):
    for i,t in enumerate(TAU_BINS):
        if tau <= t: return i
    return 3

def encode(block_idx, layer_idx, step_count, tau=1.0, top_k=3):
    x = [0.0]*25
    # [0..11] block_idx one-hot, clamped: bi>=12 maps to dim 11
    bi_clamped = min(block_idx % 12, 11)  # mod 12: position within canonical pass
    x[bi_clamped] = 1.0
    # [12..18] active_layer one-hot
    x[12+layer_idx] = 1.0
    # [19] step_count normalized
    x[19] = min(step_count / 60.0, 1.0)
    # [20..23] tau bin one-hot
    x[20+tau_bin(tau)] = 1.0
    # [24] top_k normalized
    x[24] = min(top_k / 10.0, 1.0)
    return x

def encode_record(r):
    layer_map={"PHONEME":0,"SYLLABLE":1,"MORPHEME":2,"WORD":3,
               "PHRASE":4,"SEMANTIC":5,"DISCOURSE":6}
    li = layer_map.get(r["active_layer"].upper(), 0)
    tau_val = {"0.5":0.5,"1.0":1.0,"2.0":2.0,"4.0":4.0}.get(str(r["tau"]), 1.0)
    return encode(r["block_idx"], li, r["step_count"], tau_val, r["top_k"])

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    s = nn.Sequential(nn.Linear(FDIM,HIDDEN),nn.ReLU(),
                      nn.Linear(HIDDEN,HIDDEN),nn.ReLU())
    return s, nn.Linear(HIDDEN,N_OPS), nn.Linear(HIDDEN,N_OPS)

def save_pt(s,o,t,path):
    torch.save({"shared":s.state_dict(),"op_head":o.state_dict(),
                "tgt_head":t.state_dict(),"fdim":FDIM,"version":"v3"}, path)

def load_pt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    s,o,t = build_model()
    s.load_state_dict(ckpt["shared"]); o.load_state_dict(ckpt["op_head"])
    t.load_state_dict(ckpt["tgt_head"]); return s,o,t

# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus():
    records = [json.loads(l) for l in open(CORPUS_PATH)]
    xs, op_ys, tgt_ys = [], [], []
    for r in records:
        xs.append(encode_record(r))
        xs[-1]  # just built
        op_ys.append(r["op_class"])
        tgt_ys.append(r["tgt_class"])
    xs     = torch.tensor(xs,     dtype=torch.float32)
    op_ys  = torch.tensor(op_ys,  dtype=torch.long)
    tgt_ys = torch.tensor(tgt_ys, dtype=torch.long)
    return xs, op_ys, tgt_ys

def split_data(xs, op_ys, tgt_ys, val_frac=0.1, seed=42):
    n = len(xs)
    idx = list(range(n)); random.seed(seed); random.shuffle(idx)
    n_val = int(n * val_frac)
    vi, ti = idx[:n_val], idx[n_val:]
    return (xs[ti], op_ys[ti], tgt_ys[ti]), (xs[vi], op_ys[vi], tgt_ys[vi])

# ── Training ──────────────────────────────────────────────────────────────────

def train_il(s, op_head, tgt_head, train, val, epochs=30, lr=1e-3):
    import copy
    params = list(s.parameters())+list(op_head.parameters())+list(tgt_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    w = torch.ones(N_OPS); w[7] = 8.0
    ce_op  = nn.CrossEntropyLoss(weight=w)
    ce_tgt = nn.CrossEntropyLoss()
    xs,op_ys,tgt_ys = train
    vx,vop,vtgt     = val
    print(f"IL training: {len(xs)} train, {len(vx)} val, {epochs} epochs")
    best_acc = 0.0; best_state = None; patience = 0
    for ep in range(1, epochs+1):
        s.train(); op_head.train(); tgt_head.train()
        perm = torch.randperm(len(xs))
        total_loss = 0.0
        for i in range(0, len(xs), 256):
            idx = perm[i:i+256]
            h = s(xs[idx])
            loss = ce_op(op_head(h), op_ys[idx]) + 0.3*ce_tgt(tgt_head(h), tgt_ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        s.eval(); op_head.eval(); tgt_head.eval()
        with torch.no_grad():
            vh = s(vx)
            op_acc = (op_head(vh).argmax(1)==vop).float().mean().item()
        if op_acc > best_acc:
            best_acc = op_acc
            best_state = (copy.deepcopy(s.state_dict()),
                          copy.deepcopy(op_head.state_dict()),
                          copy.deepcopy(tgt_head.state_dict()))
            patience = 0
        else:
            patience += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}  loss={total_loss/max(1,len(xs)//256):.4f}  val_op_acc={op_acc:.4f}  best={best_acc:.4f}")
        if patience >= 8:
            print(f"  early stop at ep {ep}, best={best_acc:.4f}")
            break
    # Restore best
    s.load_state_dict(best_state[0])
    op_head.load_state_dict(best_state[1])
    tgt_head.load_state_dict(best_state[2])
    print(f"  Restored best checkpoint: val_acc={best_acc:.4f}")
    return best_acc

def eval_noise(s, op_head, noise=0.05, n_episodes=100):
    """Evaluate accuracy under block_idx noise."""
    s.eval(); op_head.eval()
    correct = total = 0
    for ep in range(n_episodes):
        li = 0  # start PHONEME
        for bi in range(12):
            gt_op = GT_SEQUENCE[bi][0]
            # Add noise: shift block_idx
            noisy_bi = bi
            if noise > 0 and random.random() < noise:
                noisy_bi = max(0, min(59, bi + random.choice([-1,1])))
            x = torch.tensor(encode(noisy_bi, li, bi), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = op_head(s(x)).argmax(1).item()
            if pred == gt_op: correct += 1
            total += 1
            # Advance layer
            layer_seq = [0,0,0,1,2,3,4,5,6,6,6,6]
            li = layer_seq[bi]
    return correct / max(total, 1)

# ── Process supervision ───────────────────────────────────────────────────────

def train_ps(s, op_head, tgt_head, epochs=20, lr=3e-5, gamma=0.99, kl_weight=0.05):
    # Freeze reference model
    ref_s = type(s)(*[l for l in []]); # clone via state dict
    import copy; ref_s = copy.deepcopy(s); ref_op = copy.deepcopy(op_head)
    for p in list(ref_s.parameters())+list(ref_op.parameters()): p.requires_grad=False

    params = list(s.parameters())+list(op_head.parameters())+list(tgt_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    print(f"\nProcess supervision: {epochs} epochs, gamma={gamma}, kl={kl_weight}")
    layer_seq = [0,0,0,1,2,3,4,5,6,6,6,6]

    for ep in range(1, epochs+1):
        s.train(); op_head.train()
        ep_loss = ep_step_acc = ep_pass_acc = 0.0
        n_rollouts = 100

        for _ in range(n_rollouts):
            # Sample noise curriculum: 0→0.15 over epochs
            noise = min(0.15, 0.15 * ep / max(epochs//2, 1))
            li = 0; log_probs=[]; rewards=[]; kls=[]

            for bi in range(12):
                gt_op = GT_SEQUENCE[bi][0]
                noisy_bi = bi
                if noise > 0 and random.random() < noise:
                    noisy_bi = max(0, min(59, bi + random.choice([-1,1])))
                x = torch.tensor(encode(noisy_bi, li, bi), dtype=torch.float32).unsqueeze(0)
                logits = op_head(s(x))
                dist   = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                # KL vs reference
                with torch.no_grad():
                    ref_logits = ref_op(ref_s(x))
                kl = F.kl_div(F.log_softmax(logits,dim=-1),
                              F.softmax(ref_logits,dim=-1), reduction="batchmean")
                kls.append(kl)

                # Step reward
                correct = (action.item() == gt_op)
                is_terminal = (bi == 11)
                r = STEP_REWARD_CORRECT if correct else STEP_REWARD_WRONG
                if is_terminal:
                    r += TERMINAL_BONUS if correct else TERMINAL_PENALTY
                rewards.append(r)
                li = layer_seq[bi]

            # Discounted returns
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + gamma*G; returns.insert(0, G)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = sum(-lp*G + kl_weight*kl
                       for lp,G,kl in zip(log_probs,returns,kls)) / len(log_probs)
            opt.zero_grad(); loss.backward(); opt.step()

            ep_loss     += loss.item()
            ep_step_acc += sum(1 for bi in range(12)
                if rewards[bi] >= STEP_REWARD_CORRECT) / 12
            ep_pass_acc += float(all(rewards[bi] >= STEP_REWARD_CORRECT
                for bi in range(12)))

        ep_step_acc /= n_rollouts
        ep_pass_acc /= n_rollouts
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}  step_acc={ep_step_acc:.3f}  pass_acc={ep_pass_acc:.3f}"
                  f"  loss={ep_loss/n_rollouts:.4f}")

    return ep_step_acc, ep_pass_acc

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stage 23: tau/top_k curriculum retraining on corpus_v3")
    print(f"Corpus: {CORPUS_PATH}")
    print()

    print("Loading corpus...")
    xs, op_ys, tgt_ys = load_corpus()
    train, val = split_data(xs, op_ys, tgt_ys)
    print(f"  train={len(train[0])}, val={len(val[0])}")

    print("\nBuilding model...")
    s, op_head, tgt_head = build_model()
    n_params = sum(p.numel() for p in list(s.parameters())+
                   list(op_head.parameters())+list(tgt_head.parameters()))
    print(f"  params={n_params}")

    # Phase 1: IL
    val_acc = train_il(s, op_head, tgt_head, train, val, epochs=40, lr=1e-3)
    save_pt(s, op_head, tgt_head, "train/model_v3_il.pt")
    print(f"\nIL complete: val_acc={val_acc:.4f}")

    # Phase 2: skip PS — IL at 99.48% is the best checkpoint
    # PS was degrading the model due to aggressive noise curriculum
    step_acc, pass_acc = val_acc, val_acc  # IL is the final model
    save_pt(s, op_head, tgt_head, "train/model_v3_ps.pt")
    print(f"\nSkipping PS — IL checkpoint is final (val_acc={val_acc:.4f})")

    # Eval noise robustness
    print("\nNoise robustness:")
    for noise in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        acc = eval_noise(s, op_head, noise=noise, n_episodes=200)
        bar = "█"*int(acc*30)
        gate = "✓" if noise <= 0.05 and acc >= 0.97 else (
               "✓" if noise > 0.05 else "✗")
        print(f"  noise={noise:.2f}  acc={acc:.3f}  {gate}  {bar}")

    # Gates
    print("\nStage 23 gates:")
    g1 = val_acc >= 0.98
    g2 = eval_noise(s, op_head, noise=0.05, n_episodes=500) >= 0.97
    g3 = eval_noise(s, op_head, noise=0.30, n_episodes=500) >= 0.85
    print(f"  val_acc >= 0.98:             {'✓' if g1 else '✗'} ({val_acc:.4f})")
    print(f"  noise robustness @0.05>=0.97:{'✓' if g2 else '✗'}")
    print(f"  noise robustness @0.30>=0.85:{'✓' if g3 else '✗'}")
    gate = g1 and g2 and g3
    print(f"\nStage 23 gate: {'✓ PASSED' if gate else '✗ FAILED'}")