# -*- coding: utf-8 -*-
"""B1 复查: Φ 差分奖励尺度 vs value（judge 复验判据）。ratio ∈ [0.2,5] 通过。"""
import sys, json
sys.path.insert(0, "C:/agentwork/src")
import numpy as np, torch
from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from model.rl_reward import RewardPredictor, _round_features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = RiichiPolicy("checkpoints/sl/rl/value_pretrain.pt", seed=0, device=device)
    phi = RewardPredictor().to(device)
    phi.load_state_dict(torch.load("checkpoints/sl/rl/reward_predictor.pt", map_location="cpu")["model"])
    phi.eval()
    vals, phi_diffs = [], []
    for g in range(3):
        game = RiichiGame(RiichiConfig(), seed=g)
        prev_score = None; rd=[]; cs=[]; dl=[]; hb=[]; st=[]; ri=[]; prev_phi=None
        for _ in range(10000):
            if game.phase == "game_end": break
            obs = game.state.get_observation()
            vals.append(policy.value(obs))
            action = policy.act(obs) if game.turn == 0 else game.random_action()
            res = game.step(action)
            if res.get("round_end") or res.get("game_end"):
                o0 = game.state.get_observation(seat=0)
                score = o0["scores"][0]
                delta = (score - prev_score) if prev_score is not None else 0.0
                rd.append(delta); cs.append(score); dl.append(o0.get("oya",0))
                hb.append(o0.get("honba",0)); st.append(o0.get("riichi_sticks",0)); ri.append(o0.get("round",0))
                R = _round_features(rd, cs, dl, hb, st, ri)
                with torch.no_grad():
                    phi_k = float(phi(R.to(device)).item())
                if prev_phi is not None:
                    phi_diffs.append(phi_k - prev_phi)
                prev_phi = phi_k; prev_score = score
    v = np.array(vals); p = np.array(phi_diffs)
    # GAE 关心分布宽度匹配：std 比值为主判据（mean 比受零均值分布偏置影响）
    ratio_std = float(p.std() / max(v.std(), 1e-9))
    ratio_mean = float(abs(p.mean()) / max(abs(v.mean()), 1e-9))
    out = {"phi_diff_mean": float(p.mean()), "phi_diff_std": float(p.std()),
           "value_mean": float(v.mean()), "value_std": float(v.std()),
           "ratio_std": ratio_std, "ratio_mean": ratio_mean,
           "pass": bool(0.2 <= ratio_std <= 5.0)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    open("logs/scale_check_phi.json", "w", encoding="utf-8").write(json.dumps(out))

if __name__ == "__main__":
    main()