import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train_policy_aligned_state_ddpo import (
    TRANSFORMER_EPISODE_Q_MODEL,
    adaptive_loss_weights,
    compose_episode_q_inputs,
    compose_transformer_episode_q_inputs,
    constrained_episode_q_scores,
    grouped_advantages,
)


class PolicyAlignedStateDDPOTest(unittest.TestCase):
    def test_grouped_advantages_are_normalized_per_context(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 8.0, 8.0, 8.0, 8.0])
        advantages = grouped_advantages(rewards, group_size=4).reshape(2, 4)
        torch.testing.assert_close(advantages.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(advantages[1], torch.zeros(4), atol=1e-6, rtol=0)

    def test_uncertainty_moves_weight_from_rl_to_ntp(self):
        low = adaptive_loss_weights(
            torch.tensor([0.01]), torch.tensor([0.0]), 2.0, 1.0, 0.1, 0.4
        )
        high = adaptive_loss_weights(
            torch.tensor([1.0]), torch.tensor([0.5]), 2.0, 1.0, 0.1, 0.4
        )
        self.assertGreater(low[0], high[0])
        self.assertLess(low[1], high[1])

    def test_constrained_score_penalizes_predicted_cpa_violation(self):
        predictions = torch.log1p(
            torch.tensor(
                [
                    [10.0, 10.0, 10.0],
                    [10.0, 30.0, 10.0],
                ]
            )
        )

        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                return predictions.to(inputs.device)

        metadata = {
            "target_mode": "absolute",
            "target_transform": "log1p_nonnegative",
            "target_mean": [0.0, 0.0, 0.0],
            "target_std": [1.0, 1.0, 1.0],
            "context_dim": 6,
            "state_chunk_dim": 2,
            "input_dim": 8,
        }
        context = torch.zeros(2, 6)
        context[:, -4:] = torch.tensor([100.0, 2.0, 0.0, 0.0])
        chunks = torch.zeros(2, 2)
        normalizer = SimpleNamespace(
            condition_mean=np.zeros(4, dtype=np.float32),
            condition_std=np.ones(4, dtype=np.float32),
        )

        objective, _, _, _, cpa_ratio, budget_utilization = (
            constrained_episode_q_scores(
                [FixedModel(), FixedModel()],
                metadata,
                torch.zeros(8),
                torch.ones(8),
                context,
                chunks,
                normalizer,
                uncertainty_beta=0.0,
                support_penalty=0.0,
                cpa_violation_weight=1.0,
                budget_shortfall_weight=0.0,
                budget_util_target=0.9,
            )
        )

        torch.testing.assert_close(cpa_ratio, torch.tensor([0.5, 1.5]))
        torch.testing.assert_close(budget_utilization, torch.tensor([0.1, 0.3]))
        self.assertGreater(float(objective[0]), float(objective[1]))

        derived_objective = constrained_episode_q_scores(
            [FixedModel(), FixedModel()],
            metadata,
            torch.zeros(8),
            torch.ones(8),
            context,
            chunks,
            normalizer,
            uncertainty_beta=0.0,
            support_penalty=0.0,
            cpa_violation_weight=0.0,
            budget_shortfall_weight=0.0,
            budget_util_target=0.9,
            score_source="reward_cost",
        )[0]
        self.assertGreater(float(derived_objective[0]), float(derived_objective[1]))

    def test_policy_version_is_appended_for_snapshot_conditioned_rm(self):
        context = torch.zeros(3, 4)
        chunks = torch.zeros(3, 2)
        metadata = {
            "context_dim": 4,
            "state_chunk_dim": 2,
            "policy_version_dim": 1,
            "input_dim": 7,
        }
        features = compose_episode_q_inputs(context, chunks, metadata, 1.5)
        self.assertEqual(tuple(features.shape), (3, 7))
        torch.testing.assert_close(features[:, -1], torch.full((3,), 1.5))

    def test_transformer_episode_q_uses_state_tokens_and_auxiliary_version(self):
        context = torch.arange(40, dtype=torch.float32).reshape(2, 20)
        chunks = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        metadata = {
            "history_length": 1,
            "keep_state_indices": [0, 1],
            "context_dim": 20,
            "state_chunk_dim": 4,
            "policy_version_dim": 1,
            "aux_dim": 5,
            "aux_mean": [0.0] * 5,
            "aux_std": [1.0] * 5,
        }
        states, mask, auxiliary = compose_transformer_episode_q_inputs(
            context, chunks, metadata, policy_version=1.5
        )
        self.assertEqual(tuple(states.shape), (2, 3, 2))
        self.assertTrue(bool(mask.all()))
        torch.testing.assert_close(states[:, 0], context[:, :2])
        torch.testing.assert_close(states[:, 1:].reshape(2, 4), chunks)
        torch.testing.assert_close(auxiliary[:, -1], torch.full((2,), 1.5))

    def test_constrained_score_accepts_transformer_episode_q(self):
        class FixedTransformer(torch.nn.Module):
            def forward(self, states, valid_mask, auxiliary, history_length):
                self.seen = (states.shape, valid_mask.shape, auxiliary.shape, history_length)
                return torch.zeros(len(states), 3, device=states.device)

        model = FixedTransformer()
        context = torch.zeros(2, 20)
        context[:, -4:] = torch.tensor([100.0, 2.0, 0.0, 0.0])
        chunks = torch.zeros(2, 4)
        metadata = {
            "model": TRANSFORMER_EPISODE_Q_MODEL,
            "target_mode": "absolute",
            "target_transform": "log1p_nonnegative",
            "target_mean": [1.0, 1.0, 1.0],
            "target_std": [1.0, 1.0, 1.0],
            "history_length": 1,
            "keep_state_indices": [0, 1],
            "context_dim": 20,
            "state_chunk_dim": 4,
            "policy_version_dim": 1,
            "aux_dim": 5,
            "aux_mean": [0.0] * 5,
            "aux_std": [1.0] * 5,
        }
        normalizer = SimpleNamespace(
            condition_mean=np.zeros(4, dtype=np.float32),
            condition_std=np.ones(4, dtype=np.float32),
        )
        result = constrained_episode_q_scores(
            [model, model],
            metadata,
            torch.zeros(24),
            torch.ones(24),
            context,
            chunks,
            normalizer,
            uncertainty_beta=0.0,
            support_penalty=0.0,
            cpa_violation_weight=0.0,
            budget_shortfall_weight=0.0,
            budget_util_target=0.9,
        )
        self.assertEqual(tuple(result[0].shape), (2,))
        self.assertEqual(model.seen, (torch.Size([2, 3, 2]), torch.Size([2, 3]), torch.Size([2, 5]), 1))


if __name__ == "__main__":
    unittest.main()
