import os
import numpy as np
import torch
import sys
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

#from dppo.env.gym_utils.wrapper import wrapper_dict


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class ActionChunkWrapper(gym.Env):
	def __init__(self, env, cfg, max_episode_steps=200):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.ones(cfg.obs_dim),
			high=np.ones(cfg.obs_dim),
			dtype=np.float32
		)
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		acts_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, terminated_i, truncated_i, info_i = self.env.step(action[i])
			done_i = terminated_i or truncated_i
			obs_.append(obs_i)
			acts_.append(action[i])
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = info_[-1]
		info['obs_sequence'] = obs_
		info['action_sequence'] = acts_
		if self.count >= self.max_episode_steps:
			done = True
			truncated = True
		else:
			truncated = False
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, truncated, info

	def render(self):
		return self.env.render()
	
	def close(self):
		return
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim),
			dtype=np.float32
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		actions = actions.view(-1, self.action_horizon, self.action_dim)
		diffused_actions = self.base_policy(self.obs, actions)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		# Update base_policy history per environment
		num_envs = len(infos)
		
		# Initialize history if needed
		if len(self.base_policy.state_history) == 0:
			self.base_policy.state_history = [[] for _ in range(num_envs)]
			self.base_policy.action_history = [[] for _ in range(num_envs)]
		
		# Update each environment's history
		for env_idx, info in enumerate(infos):
			if 'obs_sequence' in info:
				# obs_sequence is [act_steps, obs_dim]
				# Extend the history with each individual observation
				for obs_i in info['obs_sequence']:
					self.base_policy.state_history[env_idx].append(obs_i)
					# Keep only last num_previous_states
					if len(self.base_policy.state_history[env_idx]) > self.base_policy.num_previous_states:
						self.base_policy.state_history[env_idx] = self.base_policy.state_history[env_idx][-self.base_policy.num_previous_states:]
			
			if 'action_sequence' in info:
				# action_sequence is [act_steps, action_dim]
				for action_i in info['action_sequence']:
					self.base_policy.action_history[env_idx].append(action_i)
					# Keep only last num_previous_actions
					if len(self.base_policy.action_history[env_idx]) > self.base_policy.num_previous_actions:
						self.base_policy.action_history[env_idx] = self.base_policy.action_history[env_idx][-self.base_policy.num_previous_actions:]
			
			# Reset history if episode is done
			if dones[env_idx]:
				self.base_policy.state_history[env_idx] = []
				self.base_policy.action_history[env_idx] = []
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()
	
