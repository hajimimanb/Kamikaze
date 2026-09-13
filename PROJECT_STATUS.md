# Project Status

> Status: **Frozen at the engineering-validation milestone.**
> Last updated: 2026-09-13

## Completed

- Riichi Mahjong rule engine (Tenhou-aligned, mjai protocol)
- Tenhou game-log acquisition & preprocessing pipeline
- SL initialization + Φ reward modeling + self-play PPO training pipeline
- Opponent pool / historical policy / causal event attention
- Evaluation framework (Mean Rank primary, Bootstrap 95% CI)
- Reproducibility infrastructure (unified paths, one config ↔ commit ↔ checkpoint ↔ seeds)
- Regression gate (D0) + unit tests

## Not Completed

- Full RL retraining (beyond the earlier ~43% development run)
- SL-vs-RL large-scale statistical evaluation (multi-seed)
- Ablations (event-attention / reward / opponent pool)
- Human behavioral analysis / strategy shift

## Reason

Large-scale RL retraining and scientific evaluation were deferred because the
available compute budget and project time allocation are prioritized toward the
primary undergraduate research track. No attempt is made to claim these were
completed.

## Scientific Claim

**No claim is made that the RL policy outperforms the SL baseline.**

The earlier development run reported a mean rank of approximately 2.50 against
SL over a limited sample; this was **not** reproduced under a finalized
multi-seed protocol and is therefore **not** treated as a scientific conclusion.