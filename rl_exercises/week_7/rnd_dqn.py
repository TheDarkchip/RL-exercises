"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.rnd_utils import PredictorNetwork, TargetNetwork  # noqa: F401
from torch import nn


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        rnd_hidden_size : int
            Hidden layer size for both RND networks.
        rnd_lr : float
            Learning rate for the RND predictor optimizer.
        rnd_update_freq : int
            Update the RND predictor every this many environment steps.
        rnd_n_layers : int
            Number of hidden layers in both RND networks.
        rnd_reward_weight : float
            Scale factor applied to the raw RND prediction error before adding it to the extrinsic reward.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        self.seed = seed
        set_seed(env, seed)

        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight

        obs_dim = env.observation_space.shape[0]
        output_dim = rnd_hidden_size

        self.target_network_rnd = TargetNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )
        self.predictor_network_rnd = PredictorNetwork(
            obs_dim, output_dim, hidden_dim=rnd_hidden_size, n_layers=rnd_n_layers
        )

        self.rnd_optimizer = optim.Adam(
            self.predictor_network_rnd.parameters(), lr=rnd_lr
        )

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        _, _, _, next_states, _, _ = zip(*training_batch)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)

        with torch.no_grad():
            target_embeddings = self.target_network_rnd(next_states)
        self.rnd_optimizer.zero_grad()
        predictor_embeddings = self.predictor_network_rnd(next_states)
        mse = nn.functional.mse_loss(predictor_embeddings, target_embeddings)

        mse.backward()
        self.rnd_optimizer.step()

        return mse.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            target_embedding = self.target_network_rnd(state_tensor)
            predictor_embedding = self.predictor_network_rnd(state_tensor)
            error = nn.functional.mse_loss(predictor_embedding, target_embedding)

        return float(self.rnd_reward_weight * error.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            reward += self.get_rnd_bonus(next_state)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if frame % self.rnd_update_freq == 0:
                    _ = self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")


@hydra.main(config_path="../configs/agent/", config_name="rnd_dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    agent = RNDDQNAgent(
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
        rnd_hidden_size=cfg.rnd.hidden_size,
        rnd_lr=cfg.rnd.learning_rate,
        rnd_update_freq=cfg.rnd.update_freq,
        rnd_n_layers=cfg.rnd.n_layers,
        rnd_reward_weight=cfg.rnd.reward_weight,
    )

    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
