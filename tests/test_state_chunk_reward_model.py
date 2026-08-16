import unittest

import torch

from train_state_chunk_reward_model import (
    StateChunkRewardModel,
    robust_state_chunk_cpa_scores,
    robust_state_chunk_scores,
)


class StateChunkRewardModelTest(unittest.TestCase):
    def test_scores_state_chunks_without_bid_inputs(self):
        models = [StateChunkRewardModel(76, 56, 32) for _ in range(3)]
        cond = torch.randn(5, 76)
        chunks = torch.randn(5, 56)
        score, diagnostics = robust_state_chunk_scores(
            models,
            cond,
            chunks,
            state_clip=2.5,
            uncertainty_beta=0.5,
            support_penalty=0.2,
        )
        self.assertEqual(tuple(score.shape), (5,))
        self.assertEqual(tuple(diagnostics["mean"].shape), (5,))
        self.assertEqual(tuple(diagnostics["uncertainty"].shape), (5,))

    def test_cpa_score_uses_reward_and_predicted_cost(self):
        models = [StateChunkRewardModel(76, 56, 32) for _ in range(3)]
        cond = torch.randn(4, 76)
        chunks = torch.randn(4, 56)
        score, diagnostics = robust_state_chunk_cpa_scores(
            models,
            cond,
            chunks,
            predicted_cost=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            cpa_constraint=torch.full((4,), 2.0),
            return_mean=1.0,
            return_std=0.5,
            state_clip=2.5,
            uncertainty_beta=0.5,
            support_penalty=0.2,
        )
        self.assertEqual(tuple(score.shape), (4,))
        self.assertEqual(tuple(diagnostics["predicted_reward"].shape), (4,))


if __name__ == "__main__":
    unittest.main()
