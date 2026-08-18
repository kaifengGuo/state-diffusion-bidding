import numpy as np
import torch

from evaluate_auctionnet_offline import TickData
from train_policy_aligned_episode_q_model import (
    append_decision_state,
    competition_scores_from_heads,
    rollout_candidate_outcomes,
    rollout_candidate_group_outcomes,
    sample_with_grouped_noise,
)
from train_state_replay_ddpo import EpisodeTemplate, ReplayState


def make_tick(time_index: int) -> TickData:
    return TickData(
        time_index=time_index,
        pvalues=np.asarray([1.0], dtype=np.float32),
        pvalue_sigmas=np.asarray([0.0], dtype=np.float32),
        market_prices=np.asarray([0.5], dtype=np.float32),
        logged_bids=np.asarray([0.5], dtype=np.float32),
        potential_conversions=np.asarray([1], dtype=np.int8),
        drop_priority=np.asarray([0.0], dtype=np.float32),
    )


def test_candidate_continuation_is_replanned_without_mutating_prefix():
    template = EpisodeTemplate(
        period=24,
        advertiser=7,
        budget=10.0,
        cpa=10.0,
        category=1,
        ticks=[make_tick(0), make_tick(1)],
    )
    prefix = ReplayState.create(template, group_id=3)
    append_decision_state(prefix, 0)
    calls = []

    def continuation(replays, time_index):
        calls.append((time_index, len(replays)))
        return np.ones(len(replays), dtype=np.float32)

    rewards, costs, scores = rollout_candidate_outcomes(
        prefix,
        decision_time=0,
        candidate_alphas=np.asarray([0.0, 1.0], dtype=np.float32),
        continuation_actions=continuation,
    )

    np.testing.assert_allclose(rewards, [1.0, 2.0])
    np.testing.assert_allclose(costs, [0.5, 1.0])
    np.testing.assert_allclose(scores, rewards)
    assert calls == [(1, 2)]
    assert prefix.total_continuous_reward == 0.0
    assert prefix.total_cost == 0.0
    assert len(prefix.history_bids) == 0
    assert len(prefix.state_history) == 1


def test_continuation_must_return_one_action_per_active_replay():
    template = EpisodeTemplate(
        period=24,
        advertiser=7,
        budget=10.0,
        cpa=10.0,
        category=1,
        ticks=[make_tick(0), make_tick(1)],
    )
    prefix = ReplayState.create(template, group_id=3)
    append_decision_state(prefix, 0)

    try:
        rollout_candidate_outcomes(
            prefix,
            decision_time=0,
            candidate_alphas=np.asarray([0.0, 1.0], dtype=np.float32),
            continuation_actions=lambda replays, time_index: np.asarray([1.0]),
        )
    except ValueError as error:
        assert "one action per active replay" in str(error)
    else:
        raise AssertionError("invalid continuation output should fail")


def test_candidate_groups_share_one_batched_continuation_call():
    template = EpisodeTemplate(
        period=24,
        advertiser=7,
        budget=10.0,
        cpa=10.0,
        category=1,
        ticks=[make_tick(0), make_tick(1)],
    )
    prefixes = [ReplayState.create(template, group_id=index) for index in range(2)]
    for prefix in prefixes:
        append_decision_state(prefix, 0)
    calls = []

    def continuation(replays, time_index):
        calls.append((time_index, len(replays)))
        return np.ones(len(replays), dtype=np.float32)

    rewards, costs, scores = rollout_candidate_group_outcomes(
        prefixes,
        decision_time=0,
        candidate_alphas=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        continuation_actions=continuation,
    )

    np.testing.assert_allclose(rewards, [[1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(costs, [[0.5, 1.0], [1.0, 0.5]])
    np.testing.assert_allclose(scores, rewards)
    assert calls == [(1, 4)]


def test_grouped_diffusion_sampling_uses_common_random_numbers():
    class FakePolicy:
        bid_dim = 2
        steps = 3

        @staticmethod
        def model_stats(x, timesteps, cond):
            return x + cond[:, :2], torch.ones_like(x), torch.zeros_like(x)

    torch.manual_seed(17)
    condition = torch.zeros(4, 2)
    samples = sample_with_grouped_noise(
        FakePolicy(), condition, np.asarray([0, 0, 1, 1], dtype=np.int64)
    )

    torch.testing.assert_close(samples[0], samples[1])
    torch.testing.assert_close(samples[2], samples[3])
    assert not torch.allclose(samples[0], samples[2])


def test_competition_score_is_reconstructed_from_reward_and_cost_heads():
    rewards = np.asarray([[10.0, 10.0]], dtype=np.float32)
    costs = np.asarray([[10.0, 30.0]], dtype=np.float32)
    scores = competition_scores_from_heads(
        rewards, costs, np.asarray([2.0], dtype=np.float32)
    )

    np.testing.assert_allclose(scores[0, 0], 10.0)
    np.testing.assert_allclose(scores[0, 1], 10.0 * (2.0 / 3.0) ** 2)
