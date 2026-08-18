import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train_policy_aligned_state_chunk_rm import (
    append_policy_version_features,
    compose_state_chunk_features,
    decode_single_actions,
    select_active_candidate_indices,
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

    def test_policy_version_is_broadcast_to_every_candidate(self):
        features = np.zeros((2, 3, 4), dtype=np.float32)
        result = append_policy_version_features(
            features, np.asarray([0.0, 1.0], dtype=np.float32)
        )
        self.assertEqual(result.shape, (2, 3, 5))
        np.testing.assert_array_equal(result[0, :, -1], np.zeros(3))
        np.testing.assert_array_equal(result[1, :, -1], np.ones(3))

    def test_active_selection_keeps_anchors_uncertainty_and_diversity(self):
        member_scores = np.asarray(
            [
                [5.0, -5.0, 4.0, 0.0, 0.1, 0.2],
                [5.0, -5.0, -4.0, 0.0, 0.1, 0.2],
                [5.0, -5.0, 0.0, 0.0, 0.1, 0.2],
            ],
            dtype=np.float32,
        )
        chunks = np.asarray(
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0]],
            dtype=np.float32,
        )
        selected = select_active_candidate_indices(member_scores, chunks, 4)
        self.assertEqual(len(np.unique(selected)), 4)
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        self.assertIn(2, selected)


if __name__ == "__main__":
    unittest.main()
