"""Contextual MarsRover env + helpers for the Level 3 exercise.

A contextual MDP (cMDP, Benjamins et al. 2023) is a family of MDPs that share
states and actions but whose dynamics, rewards, or initial state can change
with the context. For MarsRover I picked two static context features:

- goal_position: which cell holds the reward. The optimal policy depends on
  this, so a context-oblivious agent can't be optimal everywhere.
- goal_reward: how big the reward is. With gamma < 1 this only scales returns,
  the optimal policy is the same regardless. I picked it on purpose as a
  "free" feature that a context-oblivious agent could still get right.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence, SupportsFloat

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class MarsRoverContext:
    """Static context for one MarsRover instance.

    Attributes
    ----------
    goal_position : int
        Cell that gives the reward. Optimal policy depends on this.
    goal_reward : float
        Reward at the goal cell. With gamma < 1 this just scales returns,
        so the optimal policy is the same regardless.
    """

    goal_position: int
    goal_reward: float

    def as_array(self) -> np.ndarray:
        """Flatten to a numpy array. Used for nearest-neighbour distance lookups."""
        return np.array([self.goal_position, self.goal_reward], dtype=float)


class ContextualMarsRover(gym.Env):
    """1D MarsRover where the context controls the reward layout.

    Same dynamics as the original MarsRover (left=0, right=1, optional action
    flipping). Only the reward function changes with the context: goal cell
    gets ``context.goal_reward``, everything else is 0.

    Parameters
    ----------
    n_states : int, default=7
        Number of cells.
    context : MarsRoverContext or None
        Initial context. If None, defaults to goal at the rightmost cell with
        reward 10 (matches the original env).
    action_success_prob : float, default=1.0
        Probability the requested action actually gets executed (else flipped).
        Default is deterministic.
    horizon : int, default=12
        Episode length.
    seed : int or None
        Seed for the action-flip RNG.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_states: int = 7,
        context: MarsRoverContext | None = None,
        action_success_prob: float = 1.0,
        horizon: int = 12,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= action_success_prob <= 1.0:
            raise ValueError("action_success_prob must lie in [0, 1].")
        if n_states < 2:
            raise ValueError("n_states must be at least 2.")

        self.rng = np.random.default_rng(seed)
        self.n_states = int(n_states)
        self.action_success_prob = float(action_success_prob)
        self.horizon = int(horizon)

        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(2)

        # MDP buffers, kept in sync with the active context.
        self.states = np.arange(self.n_states)
        self.actions = np.arange(2)
        self.P = np.full((self.n_states, 2), self.action_success_prob, dtype=float)

        self.set_context(
            context
            if context is not None
            else MarsRoverContext(goal_position=self.n_states - 1, goal_reward=10.0)
        )

        self.current_steps = 0
        self.position = self._initial_position()

    # ----------------------------------------------------------------- context

    def set_context(self, context: MarsRoverContext) -> None:
        """Switch the active context and rebuild the reward / transition tables."""
        self.context = context
        self.rewards = self._build_rewards()
        self.transition_matrix = self.T = self.get_transition_matrix()

    def _build_rewards(self) -> list[float]:
        r = np.zeros(self.n_states, dtype=float)
        gp = int(np.clip(self.context.goal_position, 0, self.n_states - 1))
        r[gp] = float(self.context.goal_reward)
        return r.tolist()

    def _initial_position(self) -> int:
        # Start in the middle so the optimal direction actually depends on
        # where the goal is (otherwise the start would already encode it).
        return self.n_states // 2

    # -------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset to the start state and return it together with the context info."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_steps = 0
        self.position = self._initial_position()
        return self.position, {"context": self.context.as_array()}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """One step under the active context."""
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        p = float(self.P[self.position, action])
        a_used = action if self.rng.random() < p else 1 - action
        self.position = self.get_next_state(self.position, a_used)

        reward = float(self.rewards[self.position])
        terminated = False
        truncated = self.current_steps >= self.horizon
        return (
            self.position,
            reward,
            terminated,
            truncated,
            {"context": self.context.as_array()},
        )

    # --------------------------------------------------------------- model

    def get_next_state(self, state: int, action: int) -> int:
        """Deterministic next state for a given executed action."""
        if action not in (0, 1):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")
        if action == 0:
            return max(0, int(state) - 1)
        return min(self.n_states - 1, int(state) + 1)

    def get_transition_matrix(self) -> np.ndarray:
        """Build T[s, a, s'] under the current dynamics."""
        T = np.zeros((self.n_states, 2, self.n_states), dtype=float)
        for s in range(self.n_states):
            for a in range(2):
                p = float(self.P[s, a])
                T[s, a, self.get_next_state(s, a)] += p
                T[s, a, self.get_next_state(s, 1 - a)] += 1.0 - p
        return T

    def get_reward_per_action(self) -> np.ndarray:
        """R[s, a] = expected reward of taking a in s under the active context."""
        T = self.get_transition_matrix()
        r = np.array(self.rewards, dtype=float)
        return np.einsum("san,n->sa", T, r)

    # ---------------------------------------------------------------- render

    def render(self, mode: str = "human") -> None:
        """Print one line with where the rover is and what context is active."""
        print(
            f"[ContextualMarsRover] pos={self.position}, steps={self.current_steps}, "
            f"goal={self.context.goal_position}, goal_reward={self.context.goal_reward}"
        )


# ----------------------------------------------------------------- helpers


def make_context_grid(
    goal_positions: Iterable[int], goal_rewards: Iterable[float]
) -> list[MarsRoverContext]:
    """Cartesian product over the two context features. Order: (gp, gr)."""
    return [
        MarsRoverContext(int(gp), float(gr))
        for gp in goal_positions
        for gr in goal_rewards
    ]


def round_robin(
    contexts: Sequence[MarsRoverContext], n_episodes: int
) -> Iterator[MarsRoverContext]:
    """Cycle through the given contexts for n_episodes episodes.

    This is the training schedule the paper uses in Section 6.2: every context
    is visited the same number of times.
    """
    if not contexts:
        raise ValueError("`contexts` must be non-empty.")
    for i in range(int(n_episodes)):
        yield contexts[i % len(contexts)]
