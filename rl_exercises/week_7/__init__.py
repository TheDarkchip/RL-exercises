from typing import List

import torch
from gymnasium.envs.registration import register, registry
from rl_exercises.week_5.policy_gradient import REINFORCEAgent

if "LunarLander-v2" not in registry:
    register(
        id="LunarLander-v2",
        entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
        max_episode_steps=1000,
        reward_threshold=200,
    )


class REINFORCE(REINFORCEAgent):
    """Compatibility wrapper for the older week 7 policy-gradient tests."""

    def compute_returns(self, rewards: List[float]) -> List[float]:
        discounted_returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            discounted_returns.insert(0, running_return)
        return discounted_returns

    def update_agent(self, *args) -> float:
        if len(args) == 1:
            log_probs = [t[5]["log_prob"] for t in args[0]]
            rewards = [t[2] for t in args[0]]
            returns_t = torch.tensor(self.compute_returns(rewards), dtype=torch.float32)
            advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)
            loss = -(torch.stack(log_probs) * advantages).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return float(loss.item())
        if len(args) != 2:
            raise TypeError("update_agent expects a batch or (log_probs, rewards).")

        log_probs, rewards = args
        returns_t = torch.tensor(
            self.compute_returns(list(rewards)), dtype=torch.float32
        )
        advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)
        loss = -(torch.stack(list(log_probs)) * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def __getattr__(name: str):
    if name in {
        "DualHeadValueNetwork",
        "PredictorNetwork",
        "RewardForwardFilter",
        "TargetNetwork",
    }:
        from rl_exercises.week_7 import rnd_utils

        return getattr(rnd_utils, name)
    if name == "EDEDQNAgent":
        from rl_exercises.week_7.ensemble_dqn import EDEDQNAgent

        return EDEDQNAgent
    if name == "QuantileEnsembleQNetwork":
        from rl_exercises.week_7.ensemble_dqn import QuantileEnsembleQNetwork

        return QuantileEnsembleQNetwork
    if name == "NovelDPPOAgent":
        from rl_exercises.week_7.noveid_ppo import NovelDPPOAgent

        return NovelDPPOAgent
    if name == "RNDDQNAgent":
        from rl_exercises.week_7.rnd_dqn import RNDDQNAgent

        return RNDDQNAgent
    if name == "RNDPPOAgent":
        from rl_exercises.week_7.rnd_ppo import RNDPPOAgent

        return RNDPPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DualHeadValueNetwork",
    "EDEDQNAgent",
    "NovelDPPOAgent",
    "PredictorNetwork",
    "QuantileEnsembleQNetwork",
    "REINFORCE",
    "RNDDQNAgent",
    "RNDPPOAgent",
    "RewardForwardFilter",
    "TargetNetwork",
]
