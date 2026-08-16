import unittest

import torch

from train_single_step_idm import SingleActionMLP


class SingleStepIDMTest(unittest.TestCase):
    def test_outputs_one_action_for_a_full_state_chunk(self):
        model = SingleActionMLP(input_dim=84, hidden_dim=32)
        output = model(torch.randn(7, 84))
        self.assertEqual(tuple(output.shape), (7, 1))


if __name__ == "__main__":
    unittest.main()
