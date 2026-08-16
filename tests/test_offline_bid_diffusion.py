import unittest

import numpy as np
import pandas as pd
import torch

from train_offline_bid_diffusion import DiffusionPolicy, Normalizer, build_windows
from evaluate_auctionnet_offline import build_state, competition_score, enforce_budget


def state(values):
    return "(" + ",".join(str(x) for x in values) + ")"


class OfflineBidDiffusionTest(unittest.TestCase):
    def test_causal_window_alignment(self):
        rows = []
        base = np.arange(16, dtype=np.float32)
        for t in range(9):
            rows.append({
                "deliveryPeriodIndex": 7, "advertiserNumber": 1,
                "advertiserCategoryIndex": 2, "budget": 100, "CPAConstraint": 10,
                "timeStepIndex": t, "state": state(base + t), "action": 10 + t,
                "reward_continuous": 100 + t, "done": int(t == 8),
            })
        arrays, stats = build_windows(pd.DataFrame(rows), 2, 2, "reward_continuous")
        np.testing.assert_allclose(arrays["states"][0, :, 0], [1, 2])
        np.testing.assert_allclose(arrays["past_actions"][0], [10, 11])
        np.testing.assert_allclose(arrays["future_actions"][0], [12, 13])
        self.assertEqual(arrays["future_returns"][0], 205)
        self.assertEqual(stats["valid_windows"], 5)

    def test_action_transform_round_trip(self):
        normalizer = Normalizer()
        normalizer.action_mean = 1.2
        normalizer.action_std = 0.7
        values = np.array([[0.0, 1.0, 10.0, 30.0]], dtype=np.float32)
        np.testing.assert_allclose(normalizer.decode_actions(normalizer.encode_actions(values)), values, rtol=1e-5, atol=1e-5)

    def test_diffusion_sampling_records_reverse_steps(self):
        model = DiffusionPolicy(cond_dim=6, bid_dim=4, hidden_dim=32, steps=5)
        cond = torch.randn(3, 6)
        sample, records = model.sample(cond, record=True)
        self.assertEqual(tuple(sample.shape), (3, 4))
        self.assertEqual(len(records), 4)
        self.assertEqual(tuple(records[0]["old_log_prob"].shape), (3,))

    def test_auctionnet_score_penalizes_cpa_violation(self):
        self.assertEqual(competition_score(10, 50, 5), 10)
        self.assertAlmostEqual(competition_score(10, 100, 5), 2.5)

    def test_online_state_uses_latest_three_ticks(self):
        pvalue_history = [np.full((2, 2), value, dtype=np.float32) for value in [1, 2, 3, 4]]
        bid_history = [np.full(2, value, dtype=np.float32) for value in [1, 2, 3, 4]]
        auction_history = [
            np.column_stack([np.full(2, value), np.zeros(2), np.zeros(2)])
            for value in [1, 2, 3, 4]
        ]
        impression_history = [
            np.column_stack([np.ones(2), np.full(2, value)]) for value in [1, 2, 3, 4]
        ]
        market_history = [np.full(2, value, dtype=np.float32) for value in [1, 2, 3, 4]]
        state = build_state(
            4,
            100,
            75,
            np.full(3, 5, dtype=np.float32),
            pvalue_history,
            bid_history,
            auction_history,
            impression_history,
            market_history,
        )
        self.assertAlmostEqual(state[3], 3.0)
        self.assertAlmostEqual(state[8], 3.0)
        self.assertEqual(state[14], 6)
        self.assertEqual(state[15], 8)

    def test_budget_enforcement_never_overspends(self):
        bids = np.full(4, 10, dtype=np.float32)
        prices = np.asarray([4, 4, 4, 4], dtype=np.float32)
        priority = np.asarray([0.1, 0.2, 0.3, 0.4])
        _, status, costs = enforce_budget(bids, prices, 9, priority)
        self.assertLessEqual(float(costs.sum()), 9)
        self.assertEqual(int(status.sum()), 2)


if __name__ == "__main__":
    unittest.main()
