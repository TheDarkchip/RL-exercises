"""Level 3 experiment: VI on the contextual MarsRover.

I train three tabular agents and evaluate each on three context splits
(train / val / test), roughly following Section 6 of Benjamins et al. (2023):

- hidden:      VI on the average MDP over the training contexts. The agent
               doesn't see the context, so this is what round-robin training
               looks like in expectation. Gives a single policy pi : S -> A.
- visible:     VI on the joint MDP over (state, context_idx). One policy per
               training context. For unseen contexts at eval time I just
               look up the nearest training context in feature space, since
               tabular VI doesn't generalise on its own.
- specialized: a separate VI per evaluated context. This is the upper bound
               (one policy per cMDP slice) and lets me read off the
               optimality gap directly.
"""

from __future__ import annotations

from typing import Callable, Sequence

import os
import pathlib
from dataclasses import dataclass, field

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from rich import print as printr
from rich.table import Table
from rl_exercises.week_2.contextual_mars_rover import (
    ContextualMarsRover,
    MarsRoverContext,
    make_context_grid,
)
from rl_exercises.week_2.value_iteration import value_iteration

EnvFactory = Callable[[MarsRoverContext], ContextualMarsRover]


# ------------------------------------------------------------- MDP construction


def build_average_mdp(
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
) -> tuple[np.ndarray, np.ndarray]:
    """Average T and R over the given contexts.

    With round-robin training and a uniform context distribution, an agent
    that doesn't see the context effectively learns from this averaged MDP.
    """
    if not contexts:
        raise ValueError("`contexts` must be non-empty.")
    T_acc: np.ndarray | None = None
    R_acc: np.ndarray | None = None
    for c in contexts:
        env = env_factory(c)
        T_c = env.get_transition_matrix()
        R_c = env.get_reward_per_action()
        T_acc = T_c if T_acc is None else T_acc + T_c
        R_acc = R_c if R_acc is None else R_acc + R_c
    n = len(contexts)
    return T_acc / n, R_acc / n  # type: ignore[operator]


def build_joint_mdp(
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Joint MDP over (state, context_idx) with the context fixed within an episode.

    Joint state index is ``s * n_contexts + ci``. Returns (T, R, n_states, n_contexts).
    """
    if not contexts:
        raise ValueError("`contexts` must be non-empty.")
    nC = len(contexts)
    template = env_factory(contexts[0])
    nS = template.observation_space.n
    nA = template.action_space.n
    nJ = nS * nC

    T = np.zeros((nJ, nA, nJ), dtype=float)
    R = np.zeros((nJ, nA), dtype=float)
    for ci, c in enumerate(contexts):
        env = env_factory(c)
        T_c = env.get_transition_matrix()
        R_c = env.get_reward_per_action()
        for s in range(nS):
            j = s * nC + ci
            R[j, :] = R_c[s, :]
            for a in range(nA):
                for s_next in range(nS):
                    T[j, a, s_next * nC + ci] = T_c[s, a, s_next]
    return T, R, nS, nC


# ----------------------------------------------------------------------- agents


@dataclass
class HiddenAgent:
    """One state-only policy. Doesn't know about the context."""

    pi: np.ndarray  # shape (n_states,)

    def act(self, state: int, context: MarsRoverContext | None = None) -> int:
        return int(self.pi[int(state)])


@dataclass
class VisibleAgent:
    """One policy per training context, with nearest-neighbour fallback for new ones.

    For training contexts the action is just a table lookup. For unseen
    contexts I find the closest training context (Euclidean distance in
    normalised feature space) and reuse its policy. It's a crude stand-in
    for what a neural concat-agent would do via interpolation.
    """

    train_contexts: list[MarsRoverContext]
    pis: np.ndarray  # shape (n_contexts, n_states)
    feature_scale: np.ndarray = field(repr=False)

    def _nearest_idx(self, context: MarsRoverContext) -> int:
        target = context.as_array() / self.feature_scale
        train = (
            np.stack([c.as_array() for c in self.train_contexts], axis=0)
            / self.feature_scale
        )
        return int(np.argmin(np.linalg.norm(train - target, axis=1)))

    def act(self, state: int, context: MarsRoverContext) -> int:
        return int(self.pis[self._nearest_idx(context), int(state)])


@dataclass
class SpecializedAgent:
    """One optimal policy per evaluated context. Used as the upper-bound reference."""

    pi_per_context: dict[MarsRoverContext, np.ndarray]

    def act(self, state: int, context: MarsRoverContext) -> int:
        return int(self.pi_per_context[context][int(state)])


# ---------------------------------------------------------------------- training


def train_hidden(
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
    gamma: float,
    seed: int,
) -> HiddenAgent:
    """Train the hidden agent: VI on the averaged MDP."""
    T, R = build_average_mdp(contexts, env_factory)
    _, pi = value_iteration(T=T, R_sa=R, gamma=gamma, seed=seed)
    return HiddenAgent(pi=pi)


def train_visible(
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
    gamma: float,
    seed: int,
) -> VisibleAgent:
    """Train the visible agent: VI on the joint (state, context) MDP."""
    T, R, nS, nC = build_joint_mdp(contexts, env_factory)
    _, pi_joint = value_iteration(T=T, R_sa=R, gamma=gamma, seed=seed)
    pis = np.empty((nC, nS), dtype=int)
    for ci in range(nC):
        for s in range(nS):
            pis[ci, s] = int(pi_joint[s * nC + ci])
    feats = np.stack([c.as_array() for c in contexts], axis=0)
    feature_scale = np.maximum(feats.max(axis=0) - feats.min(axis=0), 1.0)
    return VisibleAgent(
        train_contexts=list(contexts), pis=pis, feature_scale=feature_scale
    )


def train_specialized(
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
    gamma: float,
    seed: int,
) -> SpecializedAgent:
    """Train one policy per context. The upper-bound reference."""
    pi_per_context: dict[MarsRoverContext, np.ndarray] = {}
    for c in contexts:
        env = env_factory(c)
        T = env.get_transition_matrix()
        R = env.get_reward_per_action()
        _, pi = value_iteration(T=T, R_sa=R, gamma=gamma, seed=seed)
        pi_per_context[c] = pi
    return SpecializedAgent(pi_per_context=pi_per_context)


# ---------------------------------------------------------------- evaluation


def evaluate_per_context(
    agent: HiddenAgent | VisibleAgent | SpecializedAgent,
    contexts: Sequence[MarsRoverContext],
    env_factory: EnvFactory,
    episodes: int,
    seed: int,
) -> dict[MarsRoverContext, np.ndarray]:
    """Roll out each context for `episodes` episodes and collect returns."""
    out: dict[MarsRoverContext, np.ndarray] = {}
    for c in contexts:
        env = env_factory(c)
        returns = np.zeros(episodes, dtype=float)
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                action = agent.act(obs, c)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns[ep] = total
        out[c] = returns
    return out


def optimality_gap(
    agent_returns: dict[MarsRoverContext, np.ndarray],
    optimal_returns: dict[MarsRoverContext, np.ndarray],
) -> float:
    """Mean per-context return gap E_c[G* - G_pi]."""
    diffs = [optimal_returns[c].mean() - agent_returns[c].mean() for c in agent_returns]
    return float(np.mean(diffs))


# -------------------------------------------------------------------- reporting


def _split_summary(
    returns_by_ctx: dict[MarsRoverContext, np.ndarray],
) -> tuple[float, float]:
    means = np.array([r.mean() for r in returns_by_ctx.values()])
    return float(means.mean()), float(means.std())


def plot_results(
    results: dict[str, dict[str, dict[MarsRoverContext, np.ndarray]]],
    save_path: pathlib.Path,
) -> None:
    """Bar chart: mean return per (split, agent) with error bars across contexts."""
    splits = list(results.keys())
    agents = list(next(iter(results.values())).keys())
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, agent in enumerate(agents):
        means = [_split_summary(results[s][agent])[0] for s in splits]
        stds = [_split_summary(results[s][agent])[1] for s in splits]
        x = np.arange(len(splits)) + (i - (len(agents) - 1) / 2) * width
        ax.bar(x, means, width=width, yerr=stds, capsize=3, label=agent)

    ax.set_xticks(np.arange(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Mean episode return")
    ax.set_title(
        "Contextual MarsRover - return per split (mean +/- std across contexts)"
    )
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def render_table(
    results: dict[str, dict[str, dict[MarsRoverContext, np.ndarray]]],
    gaps: dict[str, dict[str, float]],
) -> None:
    """Print a rich table with mean returns + optimality gaps per split / agent."""
    table = Table(title="Contextual MarsRover - mean episode return per split")
    table.add_column("Agent", style="bold")
    for split in results:
        table.add_column(f"{split}\nmean ± std")
        table.add_column(f"{split}\noptimality gap")
    for agent_name in next(iter(results.values())).keys():
        row = [agent_name]
        for split, by_agent in results.items():
            mu, sigma = _split_summary(by_agent[agent_name])
            row.append(f"{mu:.2f} ± {sigma:.2f}")
            gap = gaps[split].get(agent_name)
            row.append(f"{gap:+.2f}" if gap is not None else "—")
        table.add_row(*row)
    printr(table)


# ------------------------------------------------------------------------- main


def run_experiment(
    n_states: int = 7,
    horizon: int = 12,
    gamma: float = 0.95,
    seed: int = 0,
    eval_episodes: int = 20,
    save_dir: str | os.PathLike | None = None,
) -> dict[str, dict[str, dict[MarsRoverContext, np.ndarray]]]:
    """Run the L3 experiment end-to-end and return per-split / per-agent returns."""

    def env_factory(context: MarsRoverContext) -> ContextualMarsRover:
        return ContextualMarsRover(
            n_states=n_states,
            context=context,
            action_success_prob=1.0,
            horizon=horizon,
            seed=seed,
        )

    # Splits roughly following Section 6.4:
    # - train: 3 goals x 2 reward sizes
    # - val:   same goals, new reward size (interpolation in goal_reward only)
    # - test:  new goals altogether (extrapolation in goal_position)
    train_contexts = make_context_grid(
        goal_positions=[2, 3, 4], goal_rewards=[5.0, 10.0]
    )
    val_contexts = make_context_grid(goal_positions=[2, 3, 4], goal_rewards=[7.5])
    test_contexts = make_context_grid(goal_positions=[1, 5], goal_rewards=[15.0])

    printr("[bold]Training contexts[/bold]:", train_contexts)
    printr("[bold]Validation contexts[/bold]:", val_contexts)
    printr("[bold]Test contexts[/bold]:", test_contexts)

    hidden = train_hidden(train_contexts, env_factory, gamma=gamma, seed=seed)
    visible = train_visible(train_contexts, env_factory, gamma=gamma, seed=seed)

    splits = {
        "train": train_contexts,
        "val": val_contexts,
        "test": test_contexts,
    }

    # Specialized policies for everything we evaluate on, so we can read off
    # the optimality gap on each split.
    all_contexts = list({c for ctxs in splits.values() for c in ctxs})
    specialized = train_specialized(all_contexts, env_factory, gamma=gamma, seed=seed)

    results: dict[str, dict[str, dict[MarsRoverContext, np.ndarray]]] = {}
    gaps: dict[str, dict[str, float]] = {}
    for split, ctxs in splits.items():
        spec_returns = evaluate_per_context(
            specialized, ctxs, env_factory, eval_episodes, seed
        )
        hid_returns = evaluate_per_context(
            hidden, ctxs, env_factory, eval_episodes, seed
        )
        vis_returns = evaluate_per_context(
            visible, ctxs, env_factory, eval_episodes, seed
        )
        results[split] = {
            "specialized": spec_returns,
            "hidden": hid_returns,
            "visible": vis_returns,
        }
        gaps[split] = {
            "specialized": 0.0,
            "hidden": optimality_gap(hid_returns, spec_returns),
            "visible": optimality_gap(vis_returns, spec_returns),
        }

    render_table(results, gaps)

    printr("\n[bold]Hidden π (state -> action):[/bold]", hidden.pi.tolist())
    printr("[bold]Visible π per training context (state -> action):[/bold]")
    for c, p in zip(visible.train_contexts, visible.pis):
        printr(f"  {c} -> {p.tolist()}")

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_results(results, save_dir / "returns.png")
        for split, by_agent in results.items():
            for agent_name, ctx_returns in by_agent.items():
                rows = []
                for c, rs in ctx_returns.items():
                    rows.append(
                        np.column_stack(
                            [
                                np.full(len(rs), c.goal_position),
                                np.full(len(rs), c.goal_reward),
                                rs,
                            ]
                        )
                    )
                out = np.vstack(rows)
                np.savetxt(
                    save_dir / f"{split}_{agent_name}.csv",
                    out,
                    delimiter=",",
                    header="goal_position,goal_reward,episode_return",
                    comments="",
                )

    return results


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.resolve()
    run_experiment(save_dir=here / "results_l3")
