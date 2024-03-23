from typing import Dict

import os
import copy
import gym
import d4rl
import torch
import torch.nn as nn
import numpy as np


DATA_NAMSE = {
    "obs": "observations",
    "act": "actions",
    "rew": "rewards",
    "next_obs": "next_observations",
}

MODEL_PATH = {
    "IQL": "/apdcephfs/share_1563664/ztjiaweixu/cpt_sz/2023071300",
    "CQL": "/apdcephfs/share_1563664/ztjiaweixu/cpt_sz/2023090300",
    "MSG": "/apdcephfs/share_1563664/ztjiaweixu/cpt_sz/2023083100",
    "EDAC": "/apdcephfs/share_1563664/ztjiaweixu/cpt_sz/2023083000",
}


class mydict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = mydict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class Attack:
    def __init__(
        self,
        env_name: str,
        agent_name: str,
        dataset: Dict[str, np.ndarray],
        model_path: str,
        dataset_path: str,
        update_times: int = 100,
        step_size: float = 0.01,
        same_index: bool = False,
        froce_attack: bool = False,
        seed: int = 2023,
        device: str = "cpu",
    ):
        self.env_name = env_name
        self.agent_name = agent_name
        self.dataset = copy.deepcopy(dataset)
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.update_times = update_times
        self.step_size = step_size
        self.same_index = same_index
        self.froce_attack = froce_attack
        self.seed = seed
        self.device = device

        self._np_rng = np.random.RandomState(seed)
        self._th_rng = torch.Generator()
        self._th_rng.manual_seed(seed)

        self.attack_indexs = None
        self.original_indexs = None

        env = gym.make(env_name)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        env.close()

    def set_attack_config(
        self,
        corruption_name,
        corruption_tag,
        corruption_rate,
        corruption_range,
        corruption_random,
    ):
        self.corruption_tag = DATA_NAMSE[corruption_tag]
        self.corruption_rate = corruption_rate
        self.corruption_range = corruption_range
        self.corruption_random = corruption_random
        self.new_dataset_path = os.path.expanduser(
            os.path.join(self.dataset_path, "log_attack_data", self.env_name)
        )
        self.new_dataset_file = (
            f"random_{self.seed}{corruption_name}.pth"
            if self.corruption_random
            else f"{self.agent_name}_adversarial{corruption_name}.pth"
        )

        self.corrupt_func = getattr(self, f"corrupt_{corruption_tag}")
        self.loss_Q = getattr(self, f"loss_Q_for_{corruption_tag}")
        if self.attack_indexs is None or not self.same_index:
            self.attack_indexs, self.original_indexs = self.sample_indexs()

    def load_model(self):
        model_path = os.path.join(self.model_path, self.env_name)
        for root, dirs, files in os.walk(model_path):
            print(root)
            if f"{self.agent_name}_{self.env_name}_2023_" in root:
                if "policy.pth" in files:
                    model_path = os.path.join(root, "policy.pth")
                    break
                elif "policy_final.pth" in files:
                    model_path = os.path.join(root, "policy_final.pth")
                    break
        state_dict = torch.load(model_path, map_location=self.device)
        if self.agent_name == "IQL":
            from IQL import GaussianPolicy, TwinQ

            self.actor = (
                GaussianPolicy(
                    self.state_dim, self.action_dim, self.max_action, n_hidden=2
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                TwinQ(self.state_dim, self.action_dim, n_hidden=2)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["qf"])
        elif self.agent_name == "CQL":
            from CQL import TanhGaussianPolicy, CriticFunctions

            self.actor = (
                TanhGaussianPolicy(
                    self.state_dim,
                    self.action_dim,
                    max_action=self.max_action,
                    orthogonal_init=True,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                CriticFunctions(self.state_dim, self.action_dim, orthogonal_init=True)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.critic_1.load_state_dict(state_dict["critic1"])
            self.critic.critic_2.load_state_dict(state_dict["critic2"])
        elif self.agent_name == "EDAC":
            from EDAC import Actor, VectorizedCritic

            self.actor = (
                Actor(
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=256,
                    n_hidden=2,
                    max_action=self.max_action,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                VectorizedCritic(self.state_dim, self.action_dim, hidden_dim=256)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["critic"])
        elif self.agent_name == "MSG":
            from MSG import Actor, VectorizedCritic

            self.actor = (
                Actor(
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=256,
                    n_hidden=2,
                    max_action=self.max_action,
                )
                .to(self.device)
                .eval()
            )
            self.critic = (
                VectorizedCritic(self.state_dim, self.action_dim, hidden_dim=256)
                .to(self.device)
                .eval()
            )
            self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["critic"])
        else:
            raise NotImplementedError
        print(f"Load model from {model_path}")

    def optimize_para(self, para, std, obs, act=None):
        for _ in range(self.update_times):
            para = torch.nn.Parameter(para.clone(), requires_grad=True)
            optimizer = torch.optim.Adam(
                [para], lr=self.step_size * self.corruption_range
            )
            loss = self.loss_Q(para, obs, act, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para = torch.clamp(
                para, -self.corruption_range, self.corruption_range
            ).detach()
        return para * std

    def loss_Q_for_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_act(self, para, observation, action, std):
        noised_act = action + para * std
        qvalue = self.critic(observation, noised_act)
        return qvalue.mean()

    def loss_Q_for_next_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        action = self.actor.batch_act(noised_obs, self.device)
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_rew(self):
        # Just Placeholder
        raise NotImplementedError

    def sample_indexs(self):
        indexs = np.arange(len(self.dataset["rewards"]))
        random_num = self._np_rng.random(len(indexs))
        attacked = np.where(random_num < self.corruption_rate)[0]
        original = np.where(random_num >= self.corruption_rate)[0]
        return indexs[attacked], indexs[original]

    def sample_para(self, data, std):
        return (
            2
            * self.corruption_range
            * std
            * (torch.rand(data.shape, generator=self._th_rng).to(self.device) - 0.5)
        )

    def sample_data(self, data):
        random_data = self._np_rng.uniform(
            -self.corruption_range, self.corruption_range, size=data.shape
        )
        return random_data

    def corrupt_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_obs = original_obs + self.sample_data(original_obs) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)

            # adversarial attack obs
            attack_obs = np.zeros_like(original_obs)
            split = 10
            pointer = 0
            M = original_obs.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_act = original_act_torch[pointer : pointer + number]
                temp_obs = original_obs_torch[pointer : pointer + number]
                para = self.sample_para(temp_obs, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs, temp_act)
                noise = para.cpu().numpy()
                attack_obs[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_obs)
        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def corrupt_act(self, dataset):
        # load original act
        original_act = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_act = original_act + self.sample_data(original_act) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_obs = self.dataset["observations"][self.attack_indexs].copy()
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)

            # adversarial attack act
            attack_act = np.zeros_like(original_act)
            split = 10
            pointer = 0
            M = original_act.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_obs = original_obs_torch[pointer : pointer + number]
                temp_act = original_act_torch[pointer : pointer + number]
                para = self.sample_para(temp_act, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs, temp_act)
                noise = para.cpu().numpy()
                attack_act[pointer : pointer + number] = noise + temp_act.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")
        self.save_dataset(attack_act)
        dataset[self.corruption_tag][self.attack_indexs] = attack_act
        return dataset

    def corrupt_rew(self, dataset):
        # load original rew
        original_rew = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_rew = self._np_rng.uniform(
                -self.corruption_range, self.corruption_range, size=original_rew.shape
            )
            print(f"Random attack {self.corruption_tag}")
        else:
            attack_rew = original_rew.copy() * -self.corruption_range
            print(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_rew)
        dataset[self.corruption_tag][self.attack_indexs] = attack_rew
        return dataset

    def corrupt_next_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs) * std
            print(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)

            # adversarial attack obs
            attack_obs = np.zeros_like(original_obs)
            split = 10
            pointer = 0
            M = original_obs.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_obs = original_obs_torch[pointer : pointer + number]
                para = self.sample_para(temp_obs, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs)
                noise = para.cpu().numpy()
                attack_obs[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            print(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_obs)
        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def clear_gpu_cache(self):
        self.actor.to("cpu")
        self.critic.to("cpu")
        torch.cuda.empty_cache()

    def save_dataset(self, attack_datas):
        ### save data
        save_dict = {}
        save_dict["attack_indexs"] = self.attack_indexs
        save_dict["original_indexs"] = self.original_indexs
        save_dict[self.corruption_tag] = attack_datas
        if not os.path.exists(self.new_dataset_path):
            os.makedirs(self.new_dataset_path)
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        torch.save(save_dict, dataset_path)
        print(f"Save attack dataset in {dataset_path}")

    def get_original_data(self, indexs):
        dataset = {}
        dataset["observations"] = self.dataset["observations"][indexs]
        dataset["actions"] = self.dataset["actions"][indexs]
        dataset["rewards"] = self.dataset["rewards"][indexs]
        if "next_observations" in self.dataset.keys():
            dataset["next_observations"] = self.dataset["next_observations"][indexs]
        dataset["terminals"] = self.dataset["terminals"][indexs]
        return dataset

    def attack(self, dataset):
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        if os.path.exists(dataset_path) and not self.froce_attack:
            new_dataset = torch.load(dataset_path)
            print(f"Load new dataset from {dataset_path}")
            original_indexs, attack_indexs, attack_datas = (
                new_dataset["original_indexs"],
                new_dataset["attack_indexs"],
                new_dataset[self.corruption_tag],
            )
            ori_dataset = self.get_original_data(original_indexs)
            dataset[self.corruption_tag][attack_indexs] = attack_datas
            self.attack_indexs = attack_indexs
            return ori_dataset, dataset
        else:
            ori_dataset = self.get_original_data(self.original_indexs)
            att_dataset = self.corrupt_func(dataset)
            return ori_dataset, att_dataset


def attack_dataset(variant, dataset, logger):
    config = dictToObj(variant)
    attack_agent = Attack(
        env_name=config.env,
        agent_name=config.corruption_agent,
        dataset=dataset,
        model_path=MODEL_PATH[config.corruption_agent],
        dataset_path=config.dataset_path,
        same_index=config.same_index,
        froce_attack=config.froce_attack,
        seed=config.corruption_seed,
        device=config.device,
    )
    corruption_random = config.corruption_mode == "random"
    attack_params = {
        "corruption_rate": config.corruption_rate,
        # "corruption_range": config.corruption_range,
        "corruption_random": corruption_random,
    }
    name = ""
    attack_indexs = []
    if config.corruption_obs > 0:
        name += f"_obs_{config.corruption_obs}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_obs
        attack_agent.set_attack_config(name, "obs", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} observations")
    if config.corruption_act > 0:
        name += f"_act_{config.corruption_act}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_act
        attack_agent.set_attack_config(name, "act", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} actions")
    if config.corruption_rew > 0:
        name += f"_rew_{config.corruption_rew}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_rew
        attack_agent.set_attack_config(name, "rew", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} rewards")
    if config.corruption_next_obs > 0:
        name += f"_next_obs_{config.corruption_next_obs}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_next_obs
        attack_agent.set_attack_config(name, "next_obs", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} next_observations")

    attack_indexs = np.hstack(attack_indexs)
    return dataset, attack_indexs


if __name__ == "__main__":
    env_name = "walker2d-medium-replay-v2"
    agent_name = "CQL"
    model_path = "/data/ztjiaweixu/Code/Plot/results/cpt_sz/2023072300"
    dataset_path = "/data/ztjiaweixu/.d4rl/datasets"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = d4rl.qlearning_dataset(gym.make(env_name))
    attack_agent = Attack(
        env_name=env_name,
        agent_name=agent_name,
        dataset=dataset,
        model_path=model_path,
        dataset_path=dataset_path,
        same_index=True,
        froce_attack=True,
        device=device,
    )

    attack_agent.set_attack_config("obs", 0.2, 1.8, corruption_random=False)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("act", 0.2, 1.8, corruption_random=False)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("rew", 0.2, 1.8, corruption_random=False)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("next_obs", 0.2, 1.8, corruption_random=False)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("obs", 0.2, 1.8, corruption_random=True)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("act", 0.2, 1.8, corruption_random=True)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("rew", 0.2, 1.8, corruption_random=True)
    ori_dataset, att_dataset = attack_agent.attack(dataset)

    attack_agent.set_attack_config("next_obs", 0.2, 1.8, corruption_random=True)
    ori_dataset, att_dataset = attack_agent.attack(dataset)
