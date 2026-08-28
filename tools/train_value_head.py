# -*- coding: utf-8 -*-
"""RL 前准备：value head 监督预训练。

- 第一遍：扫 60 天 transfer_records，记录每局最终分数（最后一条 scores）
- 第二遍：流式随机抽样 ~1.2M 决策点，build_features + value 标签
  value = (final_score[seat] - 25000) / 1000   （分数量纲 [-25,+25]）
- 冻结 trunk + 全部动作头，只训 value head（MSE），cosine LR
- 产出 checkpoints/sl/rl/value_pretrain.pt（transfer_final 权重 + 新 value head）

日志 -> logs/value_train.txt（[value] step ... loss ... corr ...）
"""
import gzip, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "C:/agentwork/src")
from model.features import BINARY_HEADS, build_features, feature_channels
from model.model import MultiHeadRiichiNet, get_device

RECORDS = "C:/agentwork/data/processed/transfer_records/records-*.jsonl.gz"
CKPT = "C:/agentwork/checkpoints/sl/transfer/transfer_final.pt"
OUT = "C:/agentwork/checkpoints/sl/rl/value_pretrain.pt"
LOG = "C:/agentwork/logs/value_train.txt"
N_TARGET = 1_200_000
BATCH = 1024
LR = 3e-4
VALUE_SCALE = 1000.0
VALUE_MEAN = 25000.0


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    files = sorted(glob.glob(RECORDS))
    t0 = time.time()
    # ---- 第一遍：每局最终分数 ----
    final = {}
    total = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                final[rec["game_id"]] = rec["scores"]  # 覆盖 = 最后一条
                total += 1
    n_games = len(final)
    log("PASS1 done: %d records, %d games, %.0fs"
        % (total, n_games, time.time() - t0))

    # ---- 模型：冻结全部，只训 value head ----
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    ck = torch.load(CKPT, map_location="cpu")
    cfg = ck.get("config") or {}
    model = MultiHeadRiichiNet(feature_channels("full"),
                               channels=int(cfg.get("channels", 256)),
                               n_blocks=int(cfg.get("blocks", 50)),
                               binary_heads=BINARY_HEADS)
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad = False
    for p in model.value.parameters():
        p.requires_grad = True
    model.to(device)
    opt = torch.optim.AdamW(model.value.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    total_steps = max(1, int(N_TARGET / BATCH))
    warmup = min(100, total_steps // 10)
    log("value train: steps=%d batch=%d device=%s n_games=%d"
        % (total_steps, BATCH, device, n_games))

    rate = N_TARGET / max(total, 1)
    rng = random.Random(0)
    steps = 0
    run_loss = 0.0
    run_corr = []
    t1 = time.time()

    # ---- 第二遍：流式抽样训练 ----
    # value head 需要 trunk 输出 h，只能走 model(x) 完整 forward 取 logits["value"]
    def train_batch(x, v):
        nonlocal steps, run_loss
        if steps < warmup:
            lr = LR * (steps + 1) / warmup
        else:
            prog = min((steps - warmup) / max(total_steps - warmup, 1), 1.0)
            lr = LR * 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x, masks={}, candidates={})
            loss = F.mse_loss(logits["value"], v)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.value.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss += float(loss.item())
        steps += 1
        with torch.no_grad():
            run_corr.append((float(logits["value"][:64].mean().item()), float(v[:64].mean().item())))
        if steps % 200 == 0:
            dl = run_loss / 200
            run_loss = 0.0
            log("[value] step %d/%d loss=%.4f lr=%.2e elapsed=%.0fs"
                % (steps, total_steps, dl, lr, time.time() - t1))

    x_buf, v_buf = [], []
    n_used = 0
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if rng.random() > rate:
                    continue
                rec = json.loads(line)
                sc = final.get(rec["game_id"])
                if sc is None:
                    continue
                seat = rec["seat"]
                v = (sc[seat] - VALUE_MEAN) / VALUE_SCALE
                x_buf.append(build_features(rec, "full")[:, :, 0])  # (C,34)
                v_buf.append(v)
                n_used += 1
                if len(x_buf) >= BATCH:
                    train_batch(
                        torch.from_numpy(np.stack(x_buf)).to(device),
                        torch.tensor(v_buf, dtype=torch.float32, device=device))
                    x_buf, v_buf = [], []
                if steps >= total_steps:
                    break
        if steps >= total_steps:
            break
    if x_buf and steps < total_steps:
        train_batch(torch.from_numpy(np.stack(x_buf)).to(device),
                    torch.tensor(v_buf, dtype=torch.float32, device=device))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {**cfg, "value_mean": VALUE_MEAN, "value_scale": VALUE_SCALE,
                           "value_n_samples": n_used, "value_base": CKPT},
                "heads": BINARY_HEADS}, OUT)
    log("VALUE_DONE steps=%d samples=%d -> %s (%.0fs)"
        % (steps, n_used, OUT, time.time() - t0))


if __name__ == "__main__":
    main()
