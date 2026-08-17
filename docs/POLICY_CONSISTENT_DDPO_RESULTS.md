# Policy-Consistent Episode-Q DDPO Results

## Method

The upgraded reward-model dataset uses strict counterfactual closed-loop replay:

1. State Diffusion samples eight candidate state chunks for one context.
2. The single-step IDM decodes one current bid from each chunk.
3. AuctionNet executes that bid and exposes the real next state.
4. The current policy replans after every later state until the episode ends.
5. Candidates in one context share diffusion noise during the continuation, so
   their return difference is attributable to the candidate intervention.

The Episode-Q ensemble predicts terminal reward, cost, and CompetitionScore from
the policy context and candidate state chunk. DDPO uses the direct Score head,
ensemble uncertainty, support penalty, adaptive NTP anchoring, and reference
denoiser MSE. Best-of-N is not used at deployment; all reported policy results
use one sampled state chunk and one executed IDM bid per decision.

## Data

- RM train: AuctionNet periods 7-24, 5,755 context groups.
- RM validation: period 25, 320 groups.
- RM test: periods 26-27, 619 groups.
- Candidates per RM group: 8.
- Policy horizon: 3 future states.
- AuctionNet only provides independent market replay files for periods 7-27;
  periods 0-6 are therefore excluded from strict closed-loop collection.

## Reward Model

The selected five-member ensemble uses `rank_weight=0.5`.

| Split | Score pairwise accuracy | Top-1 regret | Uncertainty/error correlation |
|---|---:|---:|---:|
| Period 25 | 66.43% | 1.56 | 0.357 |
| Period 26-27 | 61.07% | 2.19 | 0.528 |

Without common random numbers, the pilot test pairwise accuracy was 47.72% and
top-1 regret was 24.70. Sharing continuation noise reduced the average
within-context Score standard deviation from 22.91 to 4.23 before scaling the
dataset, which removed rollout randomness from the counterfactual labels.

## DDPO Selection

One DDPO iteration was selected on period 25. All variants use `group_size=8`,
two PPO epochs, `clip=0.05`, uncertainty LCB, adaptive NTP loss, and reference
MSE.

| Learning rate | Period 25 Continuous Score |
|---|---:|
| Base policy | 306.38 |
| 1e-7 | 306.68 |
| 3e-7 | 306.83 |
| 3e-7 + CPA/pacing constraints | 306.84 |
| **1e-6** | **308.08** |

For the selected `1e-6` update, the clip fraction was 10.55%, validation NTP
loss was 0.02453, and reference-denoiser MSE was `4.69e-7`.

## Final Test

Full AuctionNet periods 26-27, 96 advertiser episodes, seed 20260805:

| Method | Continuous Score | Continuous Reward | Aggregate CPA | CPA violation | Budget utilization |
|---|---:|---:|---:|---:|---:|
| Base N=1 | 324.27 | 33,317.29 | 6.853 | 22.92% | 86.88% |
| **Episode-Q DDPO N=1** | **325.61** | **33,487.10** | 6.882 | **22.92%** | **87.42%** |

Paired advertiser-level bootstrap for Continuous Score:

- Mean improvement: **+1.34** per episode.
- 95% CI: **[+0.42, +2.39]**.
- Two-sided bootstrap p-value: **0.0025**.

Continuous Reward improves by +1.77 per episode with 95% CI
[+0.75, +2.90]. Budget utilization improves by +0.53 percentage points. The
continuous CPA violation rate is unchanged; raw aggregate CPA increases by
0.029.

## Alternating Round 2

Round 2 refreshes the counterfactual dataset under the Round-1 policy before
performing another DDPO update. The Episode-Q input adds a scalar policy-version
condition. Training uses 70% Round-1-policy groups and 30% Base-policy replay;
validation and test labels use only the Round-1 continuation policy.

The refreshed five-member RM reaches:

| Split | Score pairwise accuracy | Top-1 regret | Uncertainty/error correlation |
|---|---:|---:|---:|
| Period 25 | 69.79% | 0.95 | 0.385 |
| Period 26-27 | 64.61% | 1.99 | 0.368 |

Period-25 Continuous Score increases from 308.08 after Round 1 to 309.39 after
Round 2. Three new AuctionNet replay seeds are used for the final comparison,
giving 288 seed/advertiser episodes:

| Method | Continuous Score | Mean continuous reward | Aggregate CPA | CPA violation | Budget utilization |
|---|---:|---:|---:|---:|---:|
| Base N=1 | 321.96 | 344.28 | **6.823** | **23.61%** | 86.58% |
| Round-1 DDPO N=1 | 323.12 | 346.00 | 6.848 | **23.61%** | 87.09% |
| **Round-2 DDPO N=1** | **324.07** | **347.41** | 6.873 | 23.96% | **87.53%** |

Advertiser-cluster bootstrap, averaging the three replay seeds within each
advertiser before resampling:

- Round 1 minus Base: `+1.17`, 95% CI `[+0.50, +1.86]`, `p=0.00044`.
- Round 2 minus Round 1: `+0.95`, 95% CI `[+0.20, +1.75]`, `p=0.0122`.
- Round 2 minus Base: `+2.12`, 95% CI `[+0.73, +3.57]`, `p=0.0022`.

Round 2 raises aggregate CPA by 0.038 versus Base. Mean continuous CPA
violation increases by 0.35 percentage points, with a cluster-bootstrap
interval of `[0.00, 1.04]` percentage points; this safety tradeoff should be
reported with the score gain.
