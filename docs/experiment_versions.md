# Experiment Versions — 实验版本登记

> 本文档记录 Kamikaze 各实验版本的冻结信息，确保「任何一个数字都能追到来源」。
> 追查链路（feedback3 §五）：
>
> `Result → Manifest → Commit → Config → Checkpoint → Seed → Evaluation Protocol`

## 关键声明（feedback3 §十九）

> **Tags are reproducibility freezes rather than claims about historical feature introduction points.**

本仓库 git 历史仅 9 个文档修改 commit，无法按「某功能首次稳定出现的 commit」精确回溯。因此下列 tag 均为**可复现冻结点**（统一冻结到当前 HEAD `e9992ca`），其含义是「复现该实验时 checkout 到这个 commit + 这个 config + 这个 checkpoint」，**而非**声称该 tag 对应特性的最早引入时刻（例如 `rl-v1-event-attn` 不代表 event-attn 在 `e9992ca` 首次出现）。

## 冻结总表

| tag | 冻结 commit | config | 模型/checkpoint | 说明 |
|---|---|---|---|---|
| `sl-v1` | `e9992ca` | （无 RL config，SL 监督学习） | `checkpoints/sl/transfer/transfer_final.pt` | SL 基线（模仿人类），未随仓库分发 |
| `rl-v1` | `e9992ca` | `configs/rl_v1.json` | `models/kamikaze_rl_v1_fp16.pt`（已随仓库分发） | 基准 PPO 自对弈（副露流） |
| `rl-v1-event-attn` | `e9992ca` | `configs/rl_v1_event_attn.json` | `checkpoints/sl/rl/rl_v1_event_attn.pt` | Ablation：事件因果注意力 ON（唯一 diff = `model.use_event_attn=true`） |
| `rl-v1-reward` | `e9992ca` | `configs/rl_v1_reward_full.json` | `checkpoints/sl/rl/rl_v1_reward_full.pt` | Ablation：reward shaping（+`agari_bonus`/`deal_in_penalty`） |

## 每个实验 = 一个 config + 一个 commit + 一个 checkpoint + 一组 seeds（feedback2 §十一）

| 要素 | 记录位置 | 示例 |
|---|---|---|
| config | `configs/<name>.json`（含 `_meta.frozen_from`） | `rl_v1.json` → `e9992ca` |
| commit | `_meta.frozen_from` + `git rev-parse HEAD` | `e9992ca` |
| checkpoint | `runtime.checkpoints` 下的 `*.pt`（D2 记录路径） | `models/kamikaze_rl_v1_fp16.pt` |
| seeds | `evaluation.pilot.seeds` / `evaluation.full.seeds` | pilot 3 seeds / full 5 seeds |

## 评估协议指针

- 评估指标/统计/两阶段流程：`docs/evaluation_protocol.md` §9
- 防泄漏规则：`docs/evaluation_protocol.md` §10
- 评估执行脚本：`tools/eval_vs_sl.py`（results.json 含 `git_commit` / `config_hash` / `checkpoint_a/b`，自动落盘 manifest）
- GPU 环境部署：`docs/gpu_runbook.md`

## 待办（需 commit 授权）

1. commit 当前工作区（D1 路径清零、configs/、src/utils/、tools/d0_regression.py 及 D8–D10 骨架）。
2. 在 commit 后打 tag：
   ```bash
   git tag sl-v1
   git tag rl-v1
   git tag rl-v1-event-attn
   git tag rl-v1-reward
   ```
3. 若后续（GPU 机）在真实训练/评估中产出新 checkpoint，追加新行并指向新 commit（此时才能把 tag 从 freeze 语义升级为 feature-introduction 语义）。