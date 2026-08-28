# RL 阶段损失函数定义（v2，2026-08-25 终局奖励经网页抓取核实）

参照：Suphx (arXiv:2003.13590) §3 + docs/training_plan.md。
代码：src/model/rl_losses.py / rl_reward.py / train_rl.py。
终局奖励数据源（已抓取存档 research/）：萌娘百科·精算点数、天凤规则书、游民星空·雀魂段位点。

## 0. 终局奖励（精算点数，真实规则）
精算点数 = (终局点数 − 基点)/1000 + 马点；一位 = −(其他三位之和)（自动含 オカ 与 供托）。

| 规则 | 基点 | 马点 uma | 舍入 | 说明 |
|---|---|---|---|---|
| 天凤 tenhou | 30000 返し | (+20, +10, −10, −20) | 千位四舍五入 | 25000起・30000返・供托归1位・同点按东1局顺位 |
| 雀魂 majsoul | 25000 | (+15, +5, −5, −15) | 否 | 段位点公式：PT=(S−25000)/1000+马点+段位分 |
| M.League mleague | 30000 返し | (+50, +10, −10, −30) | 否 | 顺位点 + 持ち点差 |

- 双目标：**避4**（马点不对称 + 4位点差多为负）与**赢且赢大**（千点差=1pt）。
- 代码：rl_reward.settlement_pt / final_reward；已与百科原例逐位吻合（+57/+4/−20/−42）。

## 1. 策略损失（PPO 裁剪代理目标）
L_policy = E[ -min( r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t ) ]，ε=0.2
r_t = exp(log π_new(a_t|s_t) − log π_old(a_t|s_t))；A_t = GAE 优势。
代码：rl_losses.ppo_policy_loss。

## 2. GAE 优势估计
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
A_t = Σ_l (γλ)^l·δ_{t+l}，γ=0.99，λ=0.95；R_t = A_t + V(s_t)。
代码：rl_losses.gae。

## 3. 熵正则 + 自适应系数
H(π) = −Σ_a π(a|s)·log π(a|s)
α ← α + β·(H_target − H̄(π))，β 小步长，H_target≈1.5。
代码：rl_losses.entropy_loss / adaptive_entropy_coef。

## 4. 价值损失（critic = ValueHead）
L_value = E[ (V(s_t) − R_t)² ]。代码：rl_losses.value_loss。

## 5. 全局奖励预测器（GRU Φ）
- 结构：2 层 GRU(512) → 2×FC → 终局奖励标量（rl_reward.RewardPredictor）。
- 输入：每轮 12 维特征（本轮得分/累计分/庄位/本场/立直棒/场风位/半庄段/相对均分/得分夹取/是否和牌/有立直/负分）。
- 训练：L_reward = MSE(Φ(x_1..k), R_final)，R_final = settlement_pt（终局精算点数）。
- 每轮奖励：r_k = Φ(x_1..k) − Φ(x_1..k−1)（rl_losses.per_round_reward）。
- 作用：把"终局精算（避4+打点）"分配到每一轮，解决"单轮得分≠动作好坏"。

## 6. 神谕引导（可选）
完美特征 × Bernoulli(γ_t)，γ_t: 1→0；γ_t=0 后 LR/10 + 重要性权重阈值。
代码：rl_losses.oracle_dropout_mask。

## 7. 总损失
L_total = L_policy + vf_coef·L_value − entropy_coef·H(π)   （默认 1.0 / 0.5 / 0.01）

## 8. 自对弈（train_rl.py）
- 策略 = MultiHeadRiichiNet(284ch, 50块, 多头+value)；SL 权重初始化（transfer）。
- 对手池：历史 checkpoint 混战防坍缩（骨架期=随机对手）。
- 采样：弃牌在合法牌种 softmax；立直/吃/碰/杠经 BinaryHead+候选牌 one-hot；和牌规则式。
- 超参：lr 3e-4、clip 0.2、clip_grad 1.0、γ 0.99、λ 0.95、entropy_target 1.5、vf_coef 0.5。

## 9. 验收
- rl_losses 单测 16/16（PPO 裁剪/熵/GAE/奖励 MSE/掩码）。
- train_rl --smoke：微型网络 + 1 局 CPU 全链路（212 步，loss 有限）。
- settlement_pt 与萌娘百科示例逐位一致。
