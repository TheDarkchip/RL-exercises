"""
Distributional ensemble DQN exploration inspired by EDE.

The paper uses an ensemble of QR-DQN heads to separate reducible epistemic
uncertainty from return-distribution noise. This compact implementation keeps
the same core idea for the low-dimensional Gym environments used in the course:
each ensemble member predicts quantiles for every action, actions are selected
with a UCB bonus from ensemble disagreement, and every member is trained with
the quantile Huber loss.
"""

from typing import Any, Dict, List, Tuple

from collections import OrderedDict

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.dqn import set_seed


class QuantileEnsembleQNetwork(nn.Module):
    """Independent QR-DQN heads used as an epistemic uncertainty ensemble."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_members: int = 3,
        n_quantiles: int = 51,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_members = n_members
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        output_dim = n_actions * n_quantiles
        self.members = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("fc1", nn.Linear(obs_dim, hidden_dim)),
                            ("relu1", nn.ReLU()),
                            ("fc2", nn.Linear(hidden_dim, hidden_dim)),
                            ("relu2", nn.ReLU()),
                            ("out", nn.Linear(hidden_dim, output_dim)),
                        ]
                    )
                )
                for _ in range(n_members)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantiles with shape (members, batch, actions, quantiles)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        outputs = [member(x) for member in self.members]
        quantiles = torch.stack(outputs, dim=0)
        return quantiles.view(
            self.n_members, x.size(0), self.n_actions, self.n_quantiles
        )

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values with shape (members, batch, actions)."""
        return self(x).mean(dim=-1)


class EDEDQNAgent:
    """
    DQN agent with EDE-style distributional ensemble exploration.

    Actions maximize mean Q plus `ucb_coef` times the ensemble standard deviation
    for that action. With `ucb_coef=0`, the policy becomes the ensemble mean
    greedy policy plus the optional epsilon-greedy fallback.
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 5000,
        target_update_freq: int = 1000,
        seed: int = 0,
        hidden_dim: int = 128,
        n_members: int = 3,
        n_quantiles: int = 51,
        ucb_coef: float = 1.0,
        huber_kappa: float = 1.0,
    ) -> None:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("EDEDQNAgent supports discrete action spaces only.")

        self.env = env
        set_seed(env, seed)
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.ucb_coef = ucb_coef
        self.huber_kappa = huber_kappa
        self.total_steps = 0

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = env.action_space.n
        self.q = QuantileEnsembleQNetwork(
            obs_dim, n_actions, n_members, n_quantiles, hidden_dim
        )
        self.target_q = QuantileEnsembleQNetwork(
            obs_dim, n_actions, n_members, n_quantiles, hidden_dim
        )
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        tau = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.registered_taus = tau.view(1, 1, 1, n_quantiles, 1)

    def epsilon(self) -> float:
        return float(
            self.epsilon_final
            + (self.epsilon_start - self.epsilon_final)
            * np.exp(-self.total_steps / self.epsilon_decay)
        )

    def predict_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon():
            return int(self.env.action_space.sample())

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            member_qs = self.q.q_values(state_t).squeeze(1)
            mean_q = member_qs.mean(dim=0)
            epistemic = member_qs.var(dim=0, unbiased=False).sqrt()
            scores = mean_q + self.ucb_coef * epistemic
        return int(scores.argmax().item())

    def epistemic_uncertainty(self, states: np.ndarray) -> np.ndarray:
        """Return per-action ensemble standard deviations for a state batch."""
        states_t = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            return self.q.q_values(states_t).var(dim=0, unbiased=False).sqrt().numpy()

    def _quantile_huber_loss(
        self, chosen_quantiles: torch.Tensor, target_quantiles: torch.Tensor
    ) -> torch.Tensor:
        td_error = target_quantiles.unsqueeze(-2) - chosen_quantiles.unsqueeze(-1)
        abs_error = td_error.abs()
        quadratic = torch.minimum(
            abs_error, torch.full_like(abs_error, self.huber_kappa)
        )
        linear = abs_error - quadratic
        huber = 0.5 * quadratic.pow(2) + self.huber_kappa * linear
        taus = self.registered_taus.to(chosen_quantiles.device)
        quantile_weight = (taus - (td_error.detach() < 0).float()).abs()
        return (quantile_weight * huber / self.huber_kappa).mean()

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)

        quantiles = self.q(states_t)
        action_index = actions_t.view(1, -1, 1, 1).expand(
            quantiles.size(0), -1, 1, quantiles.size(-1)
        )
        chosen_quantiles = quantiles.gather(2, action_index).squeeze(2)

        with torch.no_grad():
            next_member_qs = self.q.q_values(next_states_t)
            next_scores = (
                next_member_qs.mean(dim=0)
                + self.ucb_coef * next_member_qs.var(dim=0, unbiased=False).sqrt()
            )
            next_actions = next_scores.argmax(dim=1)

            target_quantiles_all = self.target_q(next_states_t)
            next_action_index = next_actions.view(1, -1, 1, 1).expand(
                target_quantiles_all.size(0), -1, 1, target_quantiles_all.size(-1)
            )
            next_quantiles = target_quantiles_all.gather(2, next_action_index).squeeze(
                2
            )
            target_quantiles = (
                rewards_t.view(1, -1, 1)
                + self.gamma * (1.0 - dones_t.view(1, -1, 1)) * next_quantiles
            )

        loss = self._quantile_huber_loss(chosen_quantiles, target_quantiles)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())
        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        state, _ = self.env.reset(seed=self.seed)
        ep_reward = 0.0
        recent_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            terminal = done or truncated
            self.buffer.add(state, action, reward, next_state, terminal, {})
            state = next_state
            ep_reward += reward

            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                self.update_agent(batch)

            if terminal:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, eps={self.epsilon():.3f}"
                    )

            if frame % eval_interval == 0 and recent_rewards:
                avg = np.mean(recent_rewards[-10:])
                print(f"[Eval ] Frame {frame}, recent AvgReward(10): {avg:.2f}")

        print("Training complete.")


@hydra.main(
    config_path="../configs/agent/", config_name="ensemble_dqn", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    agent = EDEDQNAgent(
        env,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        hidden_dim=cfg.ede.hidden_dim,
        n_members=cfg.ede.n_members,
        n_quantiles=cfg.ede.n_quantiles,
        ucb_coef=cfg.ede.ucb_coef,
        huber_kappa=cfg.ede.huber_kappa,
    )
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
