import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train_policy_aligned_state_chunk_rm import (
    compose_state_chunk_features,
    decode_single_actions,
)


class IdentityStateNormalizer:
    def decode_state(self, states):
        return states


class IdentityIDMNormalizer:
    def encode_states(self, states):
        return states

    def encode_conditions(self, conditions):
        return conditions

    def decode_actions(self, actions):
        return actions


class RecordingIDM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(self, inputs):
        self.last_input_shape = tuple(inputs.shape)
        return torch.arange(len(inputs), dtype=inputs.dtype, device=inputs.device)[:, None]


class PolicyAlignedStateChunkRMTest(unittest.TestCase):
    def test_feature_contract_is_context_plus_state_chunk_only(self):
        context = np.zeros((3, 76), dtype=np.float32)
        chunks = np.zeros((3, 56), dtype=np.float32)
        features = compose_state_chunk_features(context, chunks, horizon=4)
        self.assertEqual(features.shape, (3, 132))

    def test_single_step_idm_decodes_exactly_one_action(self):
        model = RecordingIDM()
        actions = decode_single_actions(
            generated=torch.zeros(3, 56),
            current_states=np.zeros((3, 16), dtype=np.float32),
            conditions=np.zeros((3, 4), dtype=np.float32),
            state_cfg=SimpleNamespace(horizon=4),
            state_normalizer=IdentityStateNormalizer(),
            idm=model,
            idm_normalizer=IdentityIDMNormalizer(),
            device=torch.device("cpu"),
        )
        self.assertEqual(model.last_input_shape, (3, 84))
        self.assertEqual(actions.shape, (3,))
        np.testing.assert_array_equal(actions, np.arange(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
