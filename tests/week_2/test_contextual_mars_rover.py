"""Tests for the contextual MarsRover (Level 3)."""

from __future__ import annotations

import numpy as np
import pytest
from rl_exercises.week_2.contextual_experiment import (
    build_average_mdp,
    build_joint_mdp,
    train_hidden,
    train_specialized,
    train_visible,
)
from rl_exercises.week_2.contextual_mars_rover import (
    ContextualMarsRover,
    MarsRoverContext,
    make_context_grid,
    round_robin,
)


def _factory(n_states: int = 7):
    """Small helper so each test doesn't have to re-spell the env config."""

    def make(c: MarsRoverContext) -> ContextualMarsRover:
        return ContextualMarsRover(n_states=n_states, context=c, horizon=12, seed=0)

    return make


def test_set_context_updates_rewards_and_transitions():
    # Switching the context should rebuild rewards and T accordingly.
    env = ContextualMarsRover(n_states=5)
    env.set_context(MarsRoverContext(goal_position=1, goal_reward=7.0))
    assert env.rewards == [0.0, 7.0, 0.0, 0.0, 0.0]
    T = env.get_transition_matrix()
    assert T.shape == (5, 2, 5)
    assert np.isclose(T.sum(axis=-1), 1.0).all()


def test_round_robin_cycles():
    # Round robin schedule should just repeat the contexts in order.
    contexts = make_context_grid([2, 3], [5.0])
    schedule = list(round_robin(contexts, n_episodes=5))
    assert len(schedule) == 5
    assert schedule[0] == schedule[2] == schedule[4]
    assert schedule[1] == schedule[3]


def test_average_mdp_shapes():
    # Sanity-check shapes and that rows of T still sum to 1.
    contexts = make_context_grid([2, 3], [5.0, 10.0])
    T_avg, R_avg = build_average_mdp(contexts, _factory())
    assert T_avg.shape == (7, 2, 7)
    assert R_avg.shape == (7, 2)
    assert np.isclose(T_avg.sum(axis=-1), 1.0).all()


def test_joint_mdp_indexing_keeps_context_static():
    # In the joint MDP, transitions must never cross between context slices,
    # otherwise the context wouldn't really be static within an episode.
    contexts = make_context_grid([2, 4], [5.0])
    T, R, nS, nC = build_joint_mdp(contexts, _factory())
    assert nS == 7 and nC == 2
    for ci in range(nC):
        for s in range(nS):
            j = s * nC + ci
            mass_other_ctx = sum(
                T[j, :, s_next * nC + cj].sum()
                for s_next in range(nS)
                for cj in range(nC)
                if cj != ci
            )
            assert np.isclose(mass_other_ctx, 0.0)


def test_visible_recovers_per_context_optimum_on_training():
    """Visible should match the specialised return on every training context.

    I'm not comparing actions directly: VI can break ties differently between
    the joint and per-context formulations, but the achievable return has to
    match.
    """
    from rl_exercises.week_2.contextual_experiment import evaluate_per_context

    contexts = make_context_grid([2, 3, 4], [5.0, 10.0])
    factory = _factory()
    visible = train_visible(contexts, factory, gamma=0.95, seed=0)
    specialized = train_specialized(contexts, factory, gamma=0.95, seed=0)
    spec_returns = evaluate_per_context(
        specialized, contexts, factory, episodes=3, seed=0
    )
    vis_returns = evaluate_per_context(visible, contexts, factory, episodes=3, seed=0)
    for c in contexts:
        assert np.isclose(vis_returns[c].mean(), spec_returns[c].mean())


def test_hidden_policy_reachable_via_average_mdp():
    # Hidden has one policy total, so it can't match the specialised one
    # everywhere - I want to make sure at least one mismatch exists.
    contexts = make_context_grid([2, 3, 4], [5.0, 10.0])
    hidden = train_hidden(contexts, _factory(), gamma=0.95, seed=0)
    assert hidden.pi.shape == (7,)
    specialized = train_specialized(contexts, _factory(), gamma=0.95, seed=0)
    mismatches = sum(
        not np.array_equal(hidden.pi, specialized.pi_per_context[c]) for c in contexts
    )
    assert mismatches >= 1


def test_invalid_action_raises():
    env = ContextualMarsRover()
    env.reset()
    with pytest.raises(RuntimeError):
        env.step(-1)
