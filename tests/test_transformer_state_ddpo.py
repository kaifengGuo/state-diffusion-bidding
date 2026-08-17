import math

import torch

from train_transformer_state_chunk_rm import DynamicStateTransformerRewardModel
from train_transformer_state_ddpo import (
    adaptive_loss_weights,
    clipped_ppo_loss,
    grouped_advantages,
    transformer_rm_scores,
)


def test_grouped_advantages_are_normalized_per_context():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 8.0, 8.0, 8.0])
    advantages = grouped_advantages(rewards, group_size=4).reshape(2, 4)
    torch.testing.assert_close(
        advantages.mean(1), torch.zeros(2), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(advantages[1], torch.zeros(4), atol=1e-6, rtol=0)


def test_clipped_ppo_loss_uses_old_and_new_transition_log_probs():
    old = torch.zeros(2, 1)
    new = torch.tensor([[math.log(1.5)], [math.log(0.5)]], requires_grad=True)
    advantage = torch.tensor([[1.0], [-1.0]])
    loss, ratio = clipped_ppo_loss(new, old, advantage, clip=0.2)
    torch.testing.assert_close(ratio.detach().flatten(), torch.tensor([1.5, 0.5]))
    torch.testing.assert_close(loss.detach(), torch.tensor(-0.2))
    loss.backward()
    assert new.grad is not None


def test_adaptive_weights_strengthen_ntp_for_high_rm_risk():
    low = adaptive_loss_weights(
        torch.zeros(4), torch.zeros(4), 2.0, 1.0, 0.5, 0.5
    )
    high = adaptive_loss_weights(
        torch.ones(4), torch.ones(4), 2.0, 1.0, 0.5, 0.5
    )
    assert high[0] < low[0]
    assert high[1] > low[1]


def test_transformer_rm_scores_accept_runtime_horizon_and_ensemble():
    torch.manual_seed(13)
    models = [
        DynamicStateTransformerRewardModel(
            state_dim=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        ).eval()
        for _ in range(2)
    ]
    metadata = {
        "history_length": 2,
        "horizon": 4,
        "state_dim": 2,
        "keep_state_indices": [0, 1],
        "state_clip": 3.0,
    }
    context = torch.zeros(4, 2 * 16 + 2 * 2 + 4)
    chunks = torch.randn(4, 3 * 2)

    robust, mean, uncertainty, support = transformer_rm_scores(
        models, metadata, context, chunks, 0.5, 0.2
    )

    assert robust.shape == mean.shape == uncertainty.shape == support.shape == (4,)
    assert torch.all(uncertainty >= 0)
    torch.testing.assert_close(robust, mean - 0.5 * uncertainty - 0.2 * support)
