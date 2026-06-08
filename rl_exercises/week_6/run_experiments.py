from __future__ import annotations

from typing import Callable

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_exercises_mpl")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from rl_exercises.week_6.actor_critic import ActorCriticAgent  # noqa: E402
from rl_exercises.week_6.ppo import PPOAgent  # noqa: E402
from rl_exercises.week_6.sac import SACAgent, Transition  # noqa: E402
from rliable.library import get_interval_estimates  # noqa: E402
from rliable.plot_utils import plot_sample_efficiency_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _mean_curve(scores: np.ndarray) -> np.ndarray:
    return np.array([np.mean(scores[:, i]) for i in range(scores.shape[-1])])


def plot_curves(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    reps: int,
) -> None:
    algorithms = list(dict.fromkeys(df["algorithm"]))
    steps = np.array(sorted(df["step"].unique()))
    score_dict = {}
    for algorithm in algorithms:
        subset = df[df["algorithm"] == algorithm]
        seed_scores = []
        for seed in sorted(subset["seed"].unique()):
            seed_df = subset[subset["seed"] == seed].sort_values("step")
            seed_scores.append(seed_df["return_mean"].to_numpy())
        score_dict[algorithm] = np.vstack(seed_scores)

    scores, intervals = get_interval_estimates(score_dict, _mean_curve, reps=reps)
    plot_sample_efficiency_curve(
        steps,
        scores,
        intervals,
        algorithms=algorithms,
        xlabel="Environment steps",
        ylabel="Average return",
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def evaluate_discrete_agent(
    agent: ActorCriticAgent | PPOAgent, env_name: str, seed: int, episodes: int
) -> tuple[float, float]:
    eval_env = gym.make(env_name)
    returns = []
    for episode in range(episodes):
        state, _ = eval_env.reset(seed=seed * 1000 + episode)
        done = False
        total = 0.0
        while not done:
            if isinstance(agent, ActorCriticAgent):
                action, _ = agent.predict_action(state, evaluate=True)
            else:
                action, _, _, _ = agent.predict(state, evaluate=True)
            state, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total += reward
        returns.append(total)
    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def run_actor_critic_curve(
    env_name: str,
    baseline_type: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> list[dict[str, float | int | str]]:
    env = gym.make(env_name)
    agent = ActorCriticAgent(env, baseline_type=baseline_type, seed=seed)
    rows = []
    step_count = 0
    while step_count < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []
        while not done and step_count < total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append((state, action, float(reward), next_state, done, logp))
            state = next_state
            step_count += 1
            if step_count % eval_interval == 0 or step_count == total_steps:
                mean, std = evaluate_discrete_agent(
                    agent, env_name, seed, eval_episodes
                )
                rows.append(
                    {
                        "level": "l1",
                        "environment": env_name,
                        "algorithm": f"ac_{baseline_type}",
                        "seed": seed,
                        "step": step_count,
                        "return_mean": mean,
                        "return_std": std,
                    }
                )
        agent.update_agent(trajectory)
    env.close()
    return rows


def run_ppo_curve(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    algorithm: str,
    anneal_lr: bool,
    clip_vloss: bool,
) -> list[dict[str, float | int | str]]:
    env = gym.make(env_name)
    agent = PPOAgent(
        env,
        seed=seed,
        anneal_lr=anneal_lr,
        clip_vloss=clip_vloss,
        batch_size=64,
        epochs=4,
    )
    rows = []
    step_count = 0
    while step_count < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []
        while not done and step_count < total_steps:
            action, logp, ent, _ = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, float(reward), float(done), next_state)
            )
            state = next_state
            step_count += 1
            if step_count % eval_interval == 0 or step_count == total_steps:
                mean, std = evaluate_discrete_agent(
                    agent, env_name, seed, eval_episodes
                )
                rows.append(
                    {
                        "level": "l2",
                        "environment": env_name,
                        "algorithm": algorithm,
                        "seed": seed,
                        "step": step_count,
                        "return_mean": mean,
                        "return_std": std,
                    }
                )
        progress = max(0.0, 1.0 - step_count / float(total_steps))
        agent.update(trajectory, progress)
    env.close()
    return rows


def evaluate_continuous_agent(
    predict_action: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    seed: int,
    episodes: int,
) -> tuple[float, float]:
    eval_env = gym.make(env_name)
    returns = []
    for episode in range(episodes):
        state, _ = eval_env.reset(seed=seed * 1000 + episode)
        done = False
        total = 0.0
        while not done:
            action = predict_action(state)
            state, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total += reward
        returns.append(total)
    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def run_ppo_continuous_curve(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> list[dict[str, float | int | str]]:
    env = gym.make(env_name)
    agent = PPOAgent(env, seed=seed, batch_size=64, epochs=4)
    rows = []
    step_count = 0
    while step_count < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []
        while not done and step_count < total_steps:
            action, logp, ent, _ = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, float(reward), float(done), next_state)
            )
            state = next_state
            step_count += 1
            if step_count % eval_interval == 0 or step_count == total_steps:
                mean, std = evaluate_continuous_agent(
                    lambda obs: agent.predict(obs, evaluate=True)[0],
                    env_name,
                    seed,
                    eval_episodes,
                )
                rows.append(
                    {
                        "level": "l3",
                        "environment": env_name,
                        "algorithm": "ppo_continuous",
                        "seed": seed,
                        "step": step_count,
                        "return_mean": mean,
                        "return_std": std,
                    }
                )
        progress = max(0.0, 1.0 - step_count / float(total_steps))
        agent.update(trajectory, progress)
    env.close()
    return rows


def run_sac_curve(
    env_name: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> list[dict[str, float | int | str]]:
    env = gym.make(env_name)
    agent = SACAgent(
        env,
        seed=seed,
        batch_size=64,
        hidden_size=128,
        learning_starts=min(500, total_steps // 4),
        buffer_size=100_000,
    )
    rows = []
    state, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        if step <= agent.learning_starts:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.replay.add(Transition(state, action, float(reward), next_state, done))
        state = next_state
        if done:
            state, _ = env.reset()
        if step > agent.learning_starts:
            agent.update_agent()
        if step % eval_interval == 0 or step == total_steps:
            mean, std = evaluate_continuous_agent(
                lambda obs: agent.predict_action(obs, evaluate=True)[0],
                env_name,
                seed,
                eval_episodes,
            )
            rows.append(
                {
                    "level": "l3",
                    "environment": env_name,
                    "algorithm": "sac",
                    "seed": seed,
                    "step": step,
                    "return_mean": mean,
                    "return_std": std,
                }
            )
    env.close()
    return rows


def profile_settings(profile: str) -> dict[str, int]:
    if profile == "full":
        return {
            "seeds": 3,
            "l1_steps": 50_000,
            "l2_steps": 50_000,
            "l3_steps": 50_000,
            "eval_interval": 5_000,
            "eval_episodes": 5,
            "bootstrap_reps": 2_000,
        }
    if profile == "smoke":
        return {
            "seeds": 1,
            "l1_steps": 1_000,
            "l2_steps": 1_000,
            "l3_steps": 1_000,
            "eval_interval": 500,
            "eval_episodes": 1,
            "bootstrap_reps": 50,
        }
    return {
        "seeds": 2,
        "l1_steps": 4_000,
        "l2_steps": 4_000,
        "l3_steps": 4_000,
        "eval_interval": 1_000,
        "eval_episodes": 2,
        "bootstrap_reps": 200,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=["smoke", "quick", "full"], default="quick"
    )
    parser.add_argument("--skip-l1", action="store_true")
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--skip-l3", action="store_true")
    args = parser.parse_args()
    settings = profile_settings(args.profile)
    seeds = list(range(settings["seeds"]))
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.skip_l1:
        for env_name in ("CartPole-v1", "LunarLander-v3"):
            rows = []
            for baseline in ("none", "avg", "value", "gae"):
                for seed in seeds:
                    rows.extend(
                        run_actor_critic_curve(
                            env_name,
                            baseline,
                            seed,
                            settings["l1_steps"],
                            settings["eval_interval"],
                            settings["eval_episodes"],
                        )
                    )
            df = pd.DataFrame(rows)
            stem = env_name.lower().replace("-", "_")
            csv_path = RESULTS / "l1" / f"{stem}_actor_critic_baselines.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            plot_curves(
                df,
                f"Actor-Critic baselines on {env_name}",
                RESULTS / "l1" / f"{stem}_actor_critic_baselines.png",
                settings["bootstrap_reps"],
            )

    if not args.skip_l2:
        rows = []
        for seed in seeds:
            ac_rows = run_actor_critic_curve(
                "LunarLander-v3",
                "gae",
                seed,
                settings["l2_steps"],
                settings["eval_interval"],
                settings["eval_episodes"],
            )
            for row in ac_rows:
                row["level"] = "l2"
                row["algorithm"] = "actor_critic_gae"
            rows.extend(ac_rows)
            rows.extend(
                run_ppo_curve(
                    "LunarLander-v3",
                    seed,
                    settings["l2_steps"],
                    settings["eval_interval"],
                    settings["eval_episodes"],
                    "ppo_vanilla",
                    anneal_lr=False,
                    clip_vloss=False,
                )
            )
            rows.extend(
                run_ppo_curve(
                    "LunarLander-v3",
                    seed,
                    settings["l2_steps"],
                    settings["eval_interval"],
                    settings["eval_episodes"],
                    "ppo_enhanced",
                    anneal_lr=True,
                    clip_vloss=True,
                )
            )
        df = pd.DataFrame(rows)
        csv_path = RESULTS / "l2" / "lunarlander_ppo_comparison.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        plot_curves(
            df,
            "PPO comparison on LunarLander-v3",
            RESULTS / "l2" / "lunarlander_ppo_comparison.png",
            settings["bootstrap_reps"],
        )

    if not args.skip_l3:
        rows = []
        for seed in seeds:
            rows.extend(
                run_ppo_continuous_curve(
                    "Pendulum-v1",
                    seed,
                    settings["l3_steps"],
                    settings["eval_interval"],
                    settings["eval_episodes"],
                )
            )
            rows.extend(
                run_sac_curve(
                    "Pendulum-v1",
                    seed,
                    settings["l3_steps"],
                    settings["eval_interval"],
                    settings["eval_episodes"],
                )
            )
        df = pd.DataFrame(rows)
        csv_path = RESULTS / "l3" / "pendulum_ppo_sac_comparison.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        plot_curves(
            df,
            "PPO vs SAC on Pendulum-v1",
            RESULTS / "l3" / "pendulum_ppo_sac_comparison.png",
            settings["bootstrap_reps"],
        )


if __name__ == "__main__":
    main()
