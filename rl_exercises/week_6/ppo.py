# ppo.py
"""
On-policy Proximal Policy Optimization (PPO) with GAE, clipped surrogate objective,
value-loss coefficient, and entropy bonus, trained for a total number of environment steps.
"""

from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import os  # noqa: E402
import random  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from rl_exercises.agent import AbstractAgent  # noqa: E402
from rl_exercises.week_6.networks import (  # noqa: E402
    Policy,
    ValueNetwork,
)


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
    torch.cuda.manual_seed(seed)


class ContinuousPolicy(nn.Module):
    log_std_min = -20.0
    log_std_max = 2.0

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_space.high - action_space.low) / 2.0).float(),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_space.high + action_space.low) / 2.0).float(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.mean(x), self.log_std(x).clamp(self.log_std_min, self.log_std_max)

    def sample(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(x)
        normal = Normal(mean, log_std.exp())
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1), normal.entropy().sum(dim=-1)

    def log_prob_entropy(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(x)
        squashed = ((action - self.action_bias) / self.action_scale).clamp(
            -1.0 + 1e-6, 1.0 - 1e-6
        )
        z = torch.atanh(squashed)
        normal = Normal(mean, log_std.exp())
        log_prob = normal.log_prob(z) - torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        return log_prob.sum(dim=-1), normal.entropy().sum(dim=-1)

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class PPOAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        anneal_lr: bool = True,
        clip_vloss: bool = True,
        seed: int = 0,
        hidden_size: int = 128,
    ) -> None:
        set_seed(env, seed)
        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.anneal_lr = anneal_lr
        self.clip_vloss = clip_vloss

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        elif isinstance(env.action_space, gym.spaces.Box):
            self.policy = ContinuousPolicy(
                env.observation_space, env.action_space, hidden_size
            )
        else:
            raise TypeError("PPOAgent supports Discrete and Box action spaces.")
        self.value_fn = ValueNetwork(env.observation_space, hidden_size)

        # combined optimizer with separate lr for actor and critic
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ]
        )
        self.initial_lrs = [lr_actor, lr_critic]

    def predict(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int | np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(state).float()
        if self.is_discrete:
            probs = self.policy(t).squeeze(0)
            dist = Categorical(probs)
            if evaluate:
                action = int(torch.argmax(probs).item())
            else:
                action = dist.sample().item()
            return (
                action,
                dist.log_prob(torch.tensor(action)),
                dist.entropy(),
                self.value_fn(t),
            )

        if evaluate:
            action_t = self.policy.deterministic(t).squeeze(0)
            logp_t, entropy_t = self.policy.log_prob_entropy(t, action_t.unsqueeze(0))
        else:
            action_t, logp_t, entropy_t = self.policy.sample(t)
            action_t = action_t.squeeze(0)
        return (
            action_t.detach().cpu().numpy(),
            logp_t.squeeze(0),
            entropy_t.squeeze(0),
            self.value_fn(t),
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_t = torch.as_tensor(rewards, dtype=torch.float32)
        values = values.detach().flatten()
        next_values = next_values.detach().flatten()
        dones = dones.detach().float().flatten()

        advantages = torch.zeros_like(reward_t)
        gae = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            not_done = 1.0 - dones[t]
            delta = reward_t[t] + self.gamma * next_values[t] * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        return advantages.detach(), returns.detach()

    def update(
        self, trajectory: List[Any], progress_remaining: float = 1.0
    ) -> Tuple[float, float, float]:
        if self.anneal_lr:
            # Enhancement 1 from PPO implementation details: linearly anneal both
            # learning rates so late updates make smaller policy/value changes.
            for group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
                group["lr"] = initial_lr * progress_remaining

        # unpack trajectory
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        if self.is_discrete:
            actions = torch.tensor([t[1] for t in trajectory])
        else:
            actions = torch.as_tensor(np.array([t[1] for t in trajectory])).float()
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards = [t[4] for t in trajectory]
        dones = torch.tensor([t[5] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[6]).float() for t in trajectory])

        with torch.no_grad():
            values = self.value_fn(states)
            next_values = self.value_fn(next_states)
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        dataset = torch.utils.data.TensorDataset(
            states, actions, old_logps, advantages, returns, values.detach()
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret, b_old_val in loader:
                if self.is_discrete:
                    dist = Categorical(self.policy(b_states))
                    new_logp = dist.log_prob(b_actions)
                    entropy = dist.entropy()
                else:
                    new_logp, entropy = self.policy.log_prob_entropy(
                        b_states, b_actions
                    )
                ratio = (new_logp - b_oldlogp).exp()

                unclipped_objective = ratio * b_adv
                clipped_objective = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                )
                policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()

                values_pred = self.value_fn(b_states)
                if self.clip_vloss:
                    # Enhancement 2 from PPO2: clip value updates around old
                    # predictions for fidelity to PPO2. The blog notes mixed
                    # evidence for performance, so this stays configurable.
                    values_clipped = b_old_val + torch.clamp(
                        values_pred - b_old_val, -self.clip_eps, self.clip_eps
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            F.mse_loss(values_pred, b_ret, reduction="none"),
                            F.mse_loss(values_clipped, b_ret, reduction="none"),
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * F.mse_loss(values_pred, b_ret)

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        step_count = 0
        while step_count < total_steps:
            state, _ = self.env.reset(seed=self.seed)
            done = False
            trajectory: List[Any] = []

            while not done and step_count < total_steps:
                action, logp, ent, val = self.predict(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                trajectory.append(
                    (state, action, logp, ent, reward, float(done), next_state)
                )
                state = next_state
                step_count += 1

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(eval_env, num_episodes=eval_episodes)
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:5.1f} ± {std_r:4.1f}"
                    )

            # PPO update
            progress_remaining = max(0.0, 1.0 - step_count / float(total_steps))
            policy_loss, value_loss, entropy_loss = self.update(
                trajectory, progress_remaining
            )
            total_return = sum(t[4] for t in trajectory)
            print(
                f"[Train] Step {step_count:6d} Return {total_return:5.1f} Policy Loss {policy_loss:.3f} Value Loss {value_loss:.3f} Entropy Loss {entropy_loss:.3f}"
            )

        print("Training complete.")

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        returns = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed)
            done = False
            total_r = 0.0
            while not done:
                action, _, _, _ = self.predict(state, evaluate=True)
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += r
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))


@hydra.main(config_path="../configs/agent/", config_name="ppo", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = PPOAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        gae_lambda=cfg.agent.gae_lambda,
        clip_eps=cfg.agent.clip_eps,
        epochs=cfg.agent.epochs,
        batch_size=cfg.agent.batch_size,
        ent_coef=cfg.agent.ent_coef,
        vf_coef=cfg.agent.vf_coef,
        anneal_lr=cfg.agent.get("anneal_lr", True),
        clip_vloss=cfg.agent.get("clip_vloss", True),
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
