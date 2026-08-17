import numpy as np
import torch

from train_policy_aligned_transformer_episode_q import (
    PolicyConditionedTransformerEpisodeQ,
    build_transformer_inputs,
)


def build_model():
    return PolicyConditionedTransformerEpisodeQ(
        state_dim=14,
        aux_dim=7,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()


def test_transformer_episode_q_accepts_runtime_chunk_lengths():
    model = build_model()
    auxiliary = torch.zeros(3, 7)
    short = model(
        torch.zeros(3, 5, 14), torch.ones(3, 5, dtype=torch.bool), auxiliary, 2
    )
    long = model(
        torch.zeros(3, 8, 14), torch.ones(3, 8, dtype=torch.bool), auxiliary, 2
    )
    assert short.shape == long.shape == (3, 3)


def test_masked_tokens_do_not_change_transformer_episode_q_output():
    torch.manual_seed(17)
    model = build_model()
    states = torch.randn(2, 6, 14)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    auxiliary = torch.randn(2, 7)
    changed = states.clone()
    changed[:, -1] = 1000.0
    first = model(states, mask, auxiliary, history_length=2)
    second = model(changed, mask, auxiliary, history_length=2)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_flat_policy_features_become_state_tokens_and_versioned_auxiliary():
    features = np.zeros((2, 3, 80), dtype=np.float32)
    versions = np.asarray([0.0, 1.0], dtype=np.float32)
    states, masks, auxiliary = build_transformer_inputs(
        features,
        versions,
        context_dim=38,
        state_chunk_dim=42,
        history_length=2,
    )
    assert states.shape == (2, 3, 5, 14)
    assert masks.shape == (2, 3, 5)
    assert auxiliary.shape == (2, 3, 7)
    np.testing.assert_array_equal(auxiliary[0, :, -1], np.zeros(3))
    np.testing.assert_array_equal(auxiliary[1, :, -1], np.ones(3))
