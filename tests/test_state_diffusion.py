import unittest

import numpy as np
import pandas as pd

from train_state_diffusion import KEEP_STATE_INDICES, build_windows


def state(values):
    return "(" + ",".join(str(value) for value in values) + ")"


class StateDiffusionTest(unittest.TestCase):
    def test_future_state_and_action_alignment(self):
        rows = []
        for timestep in range(9):
            rows.append(
                {
                    "deliveryPeriodIndex": 7,
                    "advertiserNumber": 1,
                    "advertiserCategoryIndex": 2,
                    "budget": 100,
                    "CPAConstraint": 8,
                    "timeStepIndex": timestep,
                    "state": state(np.arange(16) + timestep),
                    "action": 10 + timestep,
                    "reward_continuous": float(timestep),
                    "done": int(timestep == 8),
                }
            )
        arrays, stats = build_windows(pd.DataFrame(rows), type("Cfg", (), {"history_length": 4, "horizon": 4, "reward_column": "reward_continuous"})())
        self.assertEqual(stats["valid_windows"], 1)
        np.testing.assert_allclose(arrays["states"][0, :, 0], [1, 2, 3, 4])
        np.testing.assert_allclose(arrays["future_states"][0, :, 0], [5, 6, 7, 8])
        np.testing.assert_allclose(arrays["past_actions"][0], [10, 11, 12, 13])
        np.testing.assert_allclose(arrays["future_rewards"][0], [4, 5, 6, 7])
        self.assertEqual(arrays["future_returns"][0], 22)

    def test_bid_stat_dimensions_are_excluded(self):
        self.assertNotIn(2, KEEP_STATE_INDICES.tolist())
        self.assertNotIn(3, KEEP_STATE_INDICES.tolist())
        self.assertEqual(len(KEEP_STATE_INDICES), 14)


if __name__ == "__main__":
    unittest.main()
