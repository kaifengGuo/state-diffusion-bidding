# DDPO Without Best-of-N

## Protocol

- Base policy: State Diffusion with planning horizon `H=3` and denoising steps
  `K=5`.
- Reward: frozen Dynamic Transformer RM members `[0, 4]`.
- DDPO updates: diffusion denoiser only; RM and single-step IDM are frozen.
- DDPO data: Period 24 contexts, with eight on-policy denoising trajectories per
  context.
- Regularization: NTP on Period 24 and reference-denoiser epsilon MSE.
- Selection: learning rate, NTP weight, and update iteration selected on Period
  25.
- Locked test: Period 26-27, 96 advertisers, evaluated once after selection.
- Inference: `N=1`; there is no candidate ranking or Best-of-N selection.

The training group is not Best-of-N. Every sampled trajectory contributes a
group-normalized terminal-reward advantage to the DDPO-IS update.

## Period 25 Selection

The no-DDPO validation Continuous Score is `306.38`.

| Configuration | Iter 1 | Iter 3 | Iter 5 |
|---|---:|---:|---:|
| `lr=1e-7`, NTP `1.0` | 306.67 | 306.97 | 307.46 |
| `lr=3e-7`, NTP `1.0` | 306.87 | 307.52 | 308.54 |
| `lr=3e-7`, NTP `0.5` | 306.87 | 307.48 | 308.52 |
| **`lr=1e-6`, NTP `0.5`** | 307.79 | 309.50 | **310.40** |

The selected checkpoint gains `+4.02`, with advertiser-paired bootstrap 95% CI
`[-0.87, +8.71]`.

## Locked Period 26-27 Test

| Method | Continuous Score | Reward | Aggregate CPA | CPA Ratio | CPA violation | Budget utilization |
|---|---:|---:|---:|---:|---:|---:|
| State Diffusion, `N=1` | 324.27 | 33,317.29 | **6.853** | **0.876** | **22.92%** | 86.88% |
| State Diffusion + DDPO, `N=1` | **327.77** | **33,933.74** | 6.998 | 0.888 | 25.00% | **89.07%** |

DDPO improves mean Continuous Score by `+3.50`. The advertiser-paired bootstrap
95% CI is `[-0.01, +7.22]` using 200,000 resamples; the two-sided bootstrap
`p=0.0506`. The improvement is borderline rather than statistically conclusive.
DDPO increases reward and budget utilization while making CPA slightly more
aggressive.

## Implementation Classification

This implementation is DDPO-IS with group-normalized advantages:

1. Sample the full reverse-diffusion trajectory from the current policy.
2. Record each stochastic transition and its old Gaussian log probability.
3. Score the completed state chunk with the frozen Transformer RM ensemble.
4. Normalize rewards among trajectories generated from the same context.
5. Recompute new transition log probabilities and optimize the PPO-style
   clipped importance-sampling objective.
6. Add adaptive NTP anchoring and reference-denoiser epsilon MSE to limit reward
   overoptimization.

The reference-denoiser term is an MSE regularizer, not an exact analytical KL.
