# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
from typing import Any, Dict, List, Optional, Tuple
import functions as func
import time
import copy
import os
import wandb
import traceback
import pyrallis
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Normal
from dataclasses import dataclass
from tqdm import trange
from logger import init_logger, Logger

from attack import attack_dataset
from replay_buffer import ReplayBuffer
from networks import VectorizedLinear, Scalar

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Wandb logging
    use_wandb: int = 0
    group: str = "2023082100"
    # model params
    n_hidden: int = 2
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 2_000_000
    # env: str = "halfcheetah-medium-v2"
    env: str = "walker2d-medium-replay-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    seed: int = 4
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ######## others
    debug: int = 0  # 0 or 1
    threshold: float = 1.0
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "~/results/corruption"
    dataset_path: str = "/apdcephfs/share_1563664/ztjiaweixu/datasets"
    ###### corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 2023
    corruption_mode: str = "adversarial"  # none, random, adversarial
    corruption_obs: float = 1.0  # 0 or 1
    corruption_act: float = 0.0  # 0 or 1
    corruption_rew: float = 0.0  # 0 or 1
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_range: float = 0.5
    corruption_rate: float = 0.1
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
        if self.env == "halfcheetah-medium-v2":
            if self.corruption_obs:
                self.threshold = 1.0
            if self.corruption_act:
                self.threshold = 1.8
        elif self.env == "walker2d-medium-replay-v2":
            if self.corruption_obs:
                self.threshold = 3.0
            if self.corruption_act:
                self.threshold = 2.5
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        if self.env.startswith("antmaze"):
            self.num_epochs = 1000
            self.buffer_size = 1000000
            self.n_episodes = 100
            self.normalize_reward = False


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU())
        self.trunk = nn.Sequential(*model)

        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action

    def batch_act(self, state: np.ndarray, device: str):
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0]
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
        num_critics: int = 10,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_critics))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], 1, num_critics))
        self.critic = nn.Sequential(*model)

        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class EDAC:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        eta: float = 1.0,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.eta = eta

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_learning_rate
        )
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _critic_diversity_loss(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        num_critics = self.critic.num_critics
        # almost exact copy from the original implementation, only style changes:
        # https://github.com/snu-mllab/EDAC/blob/198d5708701b531fd97a918a33152e1914ea14d7/lifelong_rl/trainers/q_learning/sac.py#L192

        # [num_critics, batch_size, *_dim]
        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = (
            action.unsqueeze(0)
            .repeat_interleave(num_critics, dim=0)
            .requires_grad_(True)
        )
        # [num_critics, batch_size]
        q_ensemble = self.critic(state, action)

        q_action_grad = torch.autograd.grad(
            q_ensemble.sum(), action, retain_graph=True, create_graph=True
        )[0]
        q_action_grad = q_action_grad / (
            torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10
        )
        # [batch_size, num_critics, action_dim]
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = (
            torch.eye(num_critics, device=self.device)
            .unsqueeze(0)
            .repeat(q_action_grad.shape[0], 1, 1)
        )
        # removed einsum as it is usually slower than just torch.bmm
        # [batch_size, num_critics, num_critics]
        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        diversity_loss = self._critic_diversity_loss(state, action)

        loss = critic_loss + self.eta * diversity_loss

        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            func.soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


def train(config: TrainConfig, logger: Logger):
    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    func.set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)

    h5path = (
        config.dataset_path
        if config.dataset_path is None
        else os.path.expanduser(f"{config.dataset_path}/{config.env}.hdf5")
    )
    dataset = d4rl.qlearning_dataset(env, h5path=h5path)
    ##### corrupt
    if config.corruption_mode != "none":
        dataset, attack_indexs = attack_dataset(config, dataset, logger)

    dataset, state_mean, state_std = func.normalize_dataset(config, dataset)
    logger.info("state mean: ", state_mean)
    logger.info("state std: ", state_std)

    env = func.wrap_env(env, state_mean=state_mean, state_std=state_std)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(dataset)

    # Actor & Critic setup
    actor = Actor(
        state_dim, action_dim, config.hidden_dim, config.n_hidden, config.max_action
    ).to(config.device)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.n_hidden, config.num_critics
    ).to(config.device)
    logger.info(f"Actor Network: \n{str(actor)}")
    logger.info(f"Critic Network: \n{str(critic)}")

    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.actor_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = EDAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        logger.info(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    eval_log = func.eval(config, env, actor)
    logger.record("epoch", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)
    if config.use_wandb:
        wandb.log({"epoch": 0, **eval_log})

    best_reward = -np.inf
    total_updates = 0.0
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)
            total_updates += 1
        time_end = time.time()
        epoch_time = time_end - time_start

        # evaluation
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            eval_log = func.eval(config, env, actor)
            logger.record("epoch", epoch)
            logger.record("epoch_time", epoch_time)
            for k, v in eval_log.items():
                logger.record(k, v)
            for k, v in update_info.items():
                logger.record(f"update/{k}", v)
            logger.record("update/gradient_step", total_updates)

            if config.debug:
                # debug_log = func.detect_attacked_data(config, dataset, critic)
                # for k, v in debug_log.items():
                #     logger.record(f"debug/{k}", v)
                debug_log = func.debug_data_value(config, dataset, critic)
                for k, v in debug_log.items():
                    logger.record(f"debug/{k}", v)
            logger.dump(epoch)

            if config.use_wandb:
                update_log = {f"update/{k}": v for k, v in update_info.items()}
                wandb.log({"epoch": epoch, **update_log})
                wandb.log({"epoch": epoch, **eval_log})
                if config.debug:
                    debug_log = {f"debug/{k}": v for k, v in debug_log.items()}
                    wandb.log({"epoch": epoch, **debug_log})

            now_reward = eval_log["eval/reward_mean"]
            if now_reward > best_reward:
                best_reward = now_reward
                torch.save(
                    trainer.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_best.pth"),
                )
                logger.info(
                    f"Save policy on epoch {epoch} for best reward {best_reward}."
                )

    torch.save(
        trainer.state_dict(),
        os.path.join(logger.get_dir(), f"policy_final.pth"),
    )
    logger.info(f"Save policy on epoch {epoch}.")

    if config.use_wandb:
        wandb.finish()


@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    try:
        train(config, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")


if __name__ == "__main__":
    main()
