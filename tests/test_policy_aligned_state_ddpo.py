import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train_policy_aligned_state_ddpo import (
    adaptive_loss_weights,
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


if __name__ == "__main__":
    unittest.main()
