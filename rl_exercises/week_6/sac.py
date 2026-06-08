from __future__ import annotations

from typing import Any, Deque, Tuple

import os
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent


def set_seed(env: gym.Env, seed: int = 0) -> None:
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.storage: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.storage, batch_size)
        states = torch.as_tensor(
            np.array([t.state for t in batch]), dtype=torch.float32
        )
        actions = torch.as_tensor(
            np.array([t.action for t in batch]), dtype=torch.float32
        )
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32
        )
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state.view(state.size(0), -1), action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


class GaussianPolicy(nn.Module):
    log_std_min = -20.0
    log_std_max = 2.0

    def __init__(self, state_dim: int, action_space: gym.spaces.Box, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, int(np.prod(action_space.shape)))
        self.log_std = nn.Linear(hidden_size, int(np.prod(action_space.shape)))
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_space.high - action_space.low) / 2.0).float(),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_space.high + action_space.low) / 2.0).float(),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = state.view(state.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return self.mean(x), log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        normal = torch.distributions.Normal(mean, log_std.exp())
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class SACAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        hidden_size: int = 256,
        learning_starts: int = 1_000,
        updates_per_step: int = 1,
        seed: int = 0,
    ) -> None:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError(
                "SACAgent requires a continuous gym.spaces.Box action space."
            )
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.updates_per_step = updates_per_step
        self.seed = seed

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.policy = GaussianPolicy(state_dim, env.action_space, hidden_size)
        self.q1 = SoftQNetwork(state_dim, action_dim, hidden_size)
        self.q2 = SoftQNetwork(state_dim, action_dim, hidden_size)
        self.q1_target = SoftQNetwork(state_dim, action_dim, hidden_size)
        self.q2_target = SoftQNetwork(state_dim, action_dim, hidden_size)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.replay = ReplayBuffer(buffer_size, seed=seed)

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.policy.deterministic(state_t)
            else:
                action, _ = self.policy.sample(state_t)
        return action.squeeze(0).cpu().numpy(), {}

    def update_agent(self, *_: Any, **__: Any) -> Tuple[float, float]:
        if len(self.replay) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )
        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(next_states)
            next_q = torch.min(
                self.q1_target(next_states, next_actions),
                self.q2_target(next_states, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (
                next_q - self.alpha * next_logp
            )

        q1_loss = F.mse_loss(self.q1(states, actions), target_q)
        q2_loss = F.mse_loss(self.q2(states, actions), target_q)
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, logp = self.policy.sample(states)
        q_new = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        policy_loss = (self.alpha * logp - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        return float(policy_loss.item()), float(q_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(
                source.parameters(), target.parameters()
            ):
                target_param.mul_(1.0 - self.tau).add_(source_param, alpha=self.tau)

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        state, _ = self.env.reset(seed=self.seed)
        for step in range(1, total_steps + 1):
            if step <= self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict_action(state)

            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.replay.add(Transition(state, action, float(reward), next_state, done))
            state = next_state

            if done:
                state, _ = self.env.reset()

            policy_loss = q_loss = 0.0
            if step > self.learning_starts:
                for _ in range(self.updates_per_step):
                    policy_loss, q_loss = self.update_agent()

            if step % eval_interval == 0:
                mean_r, std_r = self.evaluate(eval_env, eval_episodes)
                print(f"[Eval ] Step {step:6d} AvgReturn {mean_r:7.1f} ± {std_r:5.1f}")
                print(
                    f"[Train] Step {step:6d} Policy Loss {policy_loss:.3f} Q Loss {q_loss:.3f}"
                )
        print("Training complete.")

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        returns = []
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + episode)
            done = False
            total_r = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += reward
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])


@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    agent = SACAgent(
        env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        alpha=cfg.agent.alpha,
        batch_size=cfg.agent.batch_size,
        buffer_size=cfg.agent.buffer_size,
        hidden_size=cfg.agent.hidden_size,
        learning_starts=cfg.agent.learning_starts,
        updates_per_step=cfg.agent.updates_per_step,
        seed=cfg.seed,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
