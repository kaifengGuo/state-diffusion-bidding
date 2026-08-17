# State-Space Diffusion Bidding

Core implementation of a receding-horizon bidding policy for AuctionNet-style offline evaluation:

```text
recent states/actions/rewards
        |
        v
state diffusion policy -- sample N future state chunks
        |
        v
Transformer reward model -- rank candidate chunks
        |
        v
single-step inverse dynamics model -- decode one bid
        |
        v
execute one bid, observe the next state, and replan
```

This repository intentionally contains only model, training, evaluation, and test code. It does not include datasets, model weights, logs, server utilities, credentials, or internal infrastructure configuration.

## Method

- **State Diffusion:** generates an `H`-step future state chunk in one denoising process.
- **Dynamic Transformer RM:** scores variable-length state chunks and supports ensemble ranking.
- **Single-step IDM:** maps the current state plus the selected state chunk to one executable bid.
- **Best-of-N inference:** ranks multiple state-chunk candidates, executes one bid, and replans at the next tick.
- **Optional DDPO-IS:** post-trains every stochastic reverse-diffusion transition with
  Transformer-RM reward, a clipped importance-sampling objective, adaptive NTP
  anchoring, and reference-denoiser regularization.

DDPO and Best-of-N are independent. DDPO changes the policy parameters during
post-training; Best-of-N ranks samples at inference. The released DDPO experiment
uses one sample at inference (`N=1`).

## Final AuctionNet Result

Locked on Period 25 and evaluated once on Period 26-27:

| Method | Continuous Score | Continuous Reward | CPA Ratio | CPA violation | Budget utilization |
|---|---:|---:|---:|---:|---:|
| State Diffusion + RM + IDM | **331.62** | **34,647.97** | **0.898** | **27.08%** | 91.33% |
| Tuned CBD | 309.36 | 33,111.91 | 0.948 | 35.42% | **91.68%** |

Advertiser-paired bootstrap difference: `+22.26`, 95% CI `[+9.20, +35.92]`.

Locked policy configuration:

```text
history length = 4
planning horizon H = 3
diffusion steps K = 5
Best-of-N = 2
RM ensemble members = [0, 4]
uncertainty beta = 0
support penalty = 0
action scale = 1
```

## Repository Layout

```text
src/train_state_diffusion.py              state-chunk diffusion training
src/train_single_step_idm.py              single-bid inverse dynamics training
src/train_transformer_state_chunk_rm.py   dynamic Transformer RM training
src/train_transformer_state_ddpo.py       DDPO-IS policy post-training
src/evaluate_state_chunk_best_of_n.py     receding-horizon Best-of-N evaluation
src/evaluate_auctionnet_offline.py         deterministic AuctionNet replay
baselines/evaluate_cbd.py                  adapter for an external CBD checkout
tools/summarize_hparam_sweep.py            summary and paired-bootstrap utility
tests/                                     focused unit tests
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
pytest -q
```

AuctionNet is an external dependency and is not vendored here. Download it from its official repository and prepare the RL training CSV expected by the training scripts.

## Data Contract

The processed training CSV must contain:

```text
deliveryPeriodIndex
advertiserNumber
advertiserCategoryIndex
budget
CPAConstraint
timeStepIndex
state
action
reward_continuous
done
```

`state` is a 16-dimensional tuple/string compatible with AuctionNet's logged state format.

## Train

Set local paths without committing them:

```bash
export PYTHONPATH="$PWD/src"
export DATA_CSV=/path/to/training_data_all-rlData.csv
```

Train the state diffusion policy:

```bash
python src/train_state_diffusion.py \
  --csv-path "$DATA_CSV" \
  --output-dir outputs/state_h3_k5 \
  --history-length 4 \
  --horizon 3 \
  --diffusion-steps 5
```

Train the single-step IDM:

```bash
python src/train_single_step_idm.py \
  --csv-path "$DATA_CSV" \
  --output-dir outputs/idm_h3 \
  --horizon 3
```

Train a five-member dynamic Transformer RM:

```bash
python src/train_transformer_state_chunk_rm.py \
  --csv-path "$DATA_CSV" \
  --state-checkpoint-dir outputs/state_h3_k5 \
  --output-dir outputs/transformer_rm \
  --ensemble-size 5
```

Post-train State Diffusion with DDPO-IS while keeping the RM and IDM frozen:

```bash
python src/train_transformer_state_ddpo.py \
  --csv-path "$DATA_CSV" \
  --state-checkpoint-dir outputs/state_h3_k5 \
  --state-rm-checkpoint-dir outputs/transformer_rm \
  --ensemble-members 0 4 \
  --output-dir outputs/state_h3_k5_ddpo \
  --rl-periods 24 \
  --ntp-periods 24 \
  --validation-periods 25 \
  --iterations 5 \
  --contexts-per-iteration 128 \
  --group-size 8 \
  --ppo-epochs 2 \
  --learning-rate 1e-6 \
  --ppo-clip 0.05 \
  --reference-weight 0.1 \
  --ntp-base-weight 0.5
```

Each context produces a group of on-policy denoising trajectories during
training. All trajectories contribute group-normalized advantages and PPO
updates; the group is not a Best-of-N selector. The scalar RM reward of the
completed state chunk is broadcast to its stochastic denoising transitions,
following the terminal-reward treatment used by DDPO-IS.

## Evaluate

```bash
python src/evaluate_state_chunk_best_of_n.py \
  --auctionnet-root /path/to/AuctionNet \
  --state-checkpoint-dir outputs/state_h3_k5 \
  --idm-checkpoint-dir outputs/idm_h3 \
  --state-rm-checkpoint-dir outputs/transformer_rm \
  --periods 26 27 \
  --candidate-counts 2 \
  --candidate-pool-size 2 \
  --ensemble-members 0 4 \
  --uncertainty-beta 0 \
  --support-penalty 0 \
  --action-scale 1 \
  --ranking-objective reward \
  --output-dir results/final
```

Only the first bid decoded from the selected state chunk is executed. The policy replans after observing the next state.

To evaluate the DDPO policy without Best-of-N, point
`--state-checkpoint-dir` at the DDPO output and set both candidate arguments to
one:

```bash
python src/evaluate_state_chunk_best_of_n.py \
  --auctionnet-root /path/to/AuctionNet \
  --state-checkpoint-dir outputs/state_h3_k5_ddpo \
  --idm-checkpoint-dir outputs/idm_h3 \
  --state-rm-checkpoint-dir outputs/transformer_rm \
  --periods 26 27 \
  --candidate-counts 1 \
  --candidate-pool-size 1 \
  --ensemble-members 0 4 \
  --output-dir results/ddpo_n1
```

The locked Period 26-27 `N=1` comparison was `324.27` before DDPO and
`327.77` after DDPO. The paired bootstrap delta is `+3.50`, with 95% CI
`[-0.01, +7.22]`; this is a promising but borderline result. See
[`docs/DDPO_RESULTS.md`](docs/DDPO_RESULTS.md) for the full protocol and risk
metrics.

## Optional CBD Adapter

The CBD adapter expects a separate checkout and checkpoint. No third-party CBD source or weights are included.

```bash
pip install -r requirements-cbd.txt
python baselines/evaluate_cbd.py --help
```

## Security

Do not commit data, checkpoints, environment files, browser cookies, API tokens, SSH keys, or machine-specific paths. The included `.gitignore` blocks the common forms of these artifacts.
