import numpy as np
import torch

import evaluate_state_chunk_best_of_n as evaluator
from train_transformer_state_chunk_rm import (
    DynamicStateTransformerRewardModel,
    build_transformer_sequences,
)


def test_dynamic_horizon_labels_and_masks_are_aligned():
    history = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    future = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    rewards = np.asarray([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=np.float32)
    sequences, masks, returns, sources = build_transformer_sequences(
        history, future, rewards, history_length=2, horizon=3
    )

    assert sequences.shape == (6, 5, 3)
    assert masks.sum(axis=1).tolist() == [3, 4, 5, 3, 4, 5]
    assert returns.tolist() == [1.0, 3.0, 7.0, 3.0, 8.0, 15.0]
    assert sources.tolist() == [0, 0, 0, 1, 1, 1]


def test_padding_values_do_not_change_transformer_score():
    torch.manual_seed(7)
    model = DynamicStateTransformerRewardModel(
        state_dim=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    states = torch.randn(2, 5, 3)
    mask = torch.tensor([[True, True, True, False, False]]).repeat(2, 1)
    changed = states.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 1000

    with torch.inference_mode():
        expected = model(states, mask)
        actual = model(changed, mask)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_transformer_scoring_uses_runtime_horizon():
    class ShapeCheckingModel(torch.nn.Module):
        def forward(self, states, valid_mask):
            assert states.shape == (4, 5, 3)
            assert valid_mask.shape == (4, 5)
            return states.mean(dim=(1, 2))

    cond = torch.randn(4, 2 * evaluator.STATE_DIM)
    state_chunk = torch.randn(4, 3 * 3)
    scores, diagnostics = evaluator.robust_transformer_state_chunk_scores(
        [ShapeCheckingModel()],
        cond,
        state_chunk,
        {
            "history_length": 2,
            "horizon": 4,
            "state_dim": 3,
            "keep_state_indices": [0, 1, 2],
            "state_clip": 10.0,
        },
        uncertainty_beta=0.0,
        support_penalty=0.0,
    )

    assert scores.shape == (4,)
    assert diagnostics["uncertainty"].shape == (4,)
