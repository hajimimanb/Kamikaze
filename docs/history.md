# History — Historical Development Record

> ⚠️ **These results are a historical development snapshot, NOT a scientific conclusion.**

Early development ran on a single GPU and reached about 43% of the planned
10,000-game schedule. The figures below were captured **during that partial run**
and were **not** reproduced under the finalized multi-seed evaluation protocol.
See [`PROJECT_STATUS.md`](../PROJECT_STATUS.md).

## Development run (至 2026-08-28, ~43% of plan)

| Item | Value |
|---|---|
| Plan / done | 10,000 games / 500 epochs; **4,293 games / 214 epochs (~43%)** |
| Stability | ratio ≈ 1.000, KL 0.001–0.14, clip < 5% (no divergence) |

## Partial model stats (4,293 games / 49,544 rounds)

| Metric | Value |
|---|---|
| Win style (model own) | tile-open ron/tsumo **75.0%** / riichi + tanyao 23.5% / silent 1.5% |
| Open calls | 1.48 / round (aggressive kuitan play) |
| Win / deal-in | 19.3% / 14.5% per round |
| Avg win / deal-in points | **5,164 / 4,587** |
| **vs-SL mean rank** | **2.50** (n=728, evenness 2.5) |
| vs-old-version mean rank | 2.62 (n=1,030) |

**vs-SL sliding evaluation** (independent eval process, 100-game window):
pt-weighted win rate 0.27–0.79, per-100-game pt **−1533 ~ +2433**, rank 0.25–0.53
— mixed, not converged.

> **Interpretation restraint:** per the review, these numbers are kept in this
> auxiliary document rather than the README front matter, and are explicitly
> **not** treated as evidence that RL outperforms SL.