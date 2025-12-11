import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import sys
import os
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../diffusion_rl'))
from diffusion_policy_transformer import PolicyDiffusionTransformer
from train_diffusion_policy import TrainDiffusionPolicy

class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	

class PandaDiffusionPolicyWrapper:
    """Wrapper to make PolicyDiffusionTransformer compatible with DSRL"""
    def __init__(self, trainer, num_previous_states=10, num_previous_actions=9):
        """
        Args:
            trainer: Your TrainDiffusionPolicy instance (has model + diffusion_sample + normalization)
            num_previous_states: Number of previous states to condition on
            num_previous_actions: Number of previous actions to condition on
        """
        self.trainer = trainer
        self.device = trainer.device
        self.num_previous_states = num_previous_states
        self.num_previous_actions = num_previous_actions
        self.trainer.model.eval()
        self.use_time_embedding = True
        
        # State/action history tracking per environment
        self.state_history = []
        self.action_history = []
        
    def __call__(self, obs, initial_noise, return_numpy=True):
        """
        Args:
            obs: Current observation(s), shape (batch_size, obs_dim)
            initial_noise: shape (batch_size, act_steps, action_dim)
        """
        batch_size = obs.shape[0]
        act_steps = initial_noise.shape[1]
        
        # History is now managed by DiffusionPolicyEnvWrapper.step_wait()
        if len(self.state_history) == 0:
            # First call - initialize with current obs
            obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
            self.state_history = [[obs_np[i]] for i in range(batch_size)]
            self.action_history = [[] for _ in range(batch_size)]
        
        # Prepare batched inputs from existing history
        obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        previous_states_batch = []
        previous_actions_batch = []
        previous_states_mask_batch = []
        previous_actions_mask_batch = []
        episode_timesteps_batch = []
        
        for i in range(batch_size):
            # Get state sequence
            if len(self.state_history[i]) > 0:
                states = np.array(self.state_history[i][-self.num_previous_states:])
                state_seq_len = len(states)
            else:
                states = np.zeros((0, obs_np.shape[1]))
                state_seq_len = 0
            
            # Pad states if needed
            if state_seq_len < self.num_previous_states:
                padding = np.zeros((self.num_previous_states - state_seq_len, obs_np.shape[1]))
                states = np.concatenate([states, padding], axis=0)
                state_mask = np.concatenate([
                    np.zeros(state_seq_len),
                    np.ones(self.num_previous_states - state_seq_len)
                ])
            else:
                state_mask = np.zeros(self.num_previous_states)
            
            # Get action sequence
            if len(self.action_history[i]) > 0:
                actions = np.array(self.action_history[i][-self.num_previous_actions:])
                action_seq_len = len(actions)
            else:
                actions = np.zeros((0, initial_noise.shape[2]))
                action_seq_len = 0
            
            # Pad actions if needed
            if action_seq_len < self.num_previous_actions:
                padding = np.zeros((self.num_previous_actions - action_seq_len, initial_noise.shape[2]))
                actions = np.concatenate([actions, padding], axis=0)
                action_mask = np.concatenate([
                    np.zeros(action_seq_len),
                    np.ones(self.num_previous_actions - action_seq_len)
                ])
            else:
                action_mask = np.zeros(self.num_previous_actions)
            
            # Normalize states and actions
            states_normalized = self.trainer.normalize_states(states)
            if action_seq_len > 0:
                actions_normalized = self.trainer.normalize_actions(actions)
            else:
                actions_normalized = actions
            
            # Episode timesteps
            current_timestep = len(self.state_history[i]) - 1
            start_timestep = max(0, current_timestep - self.num_previous_states + 1)
            if self.use_time_embedding:
                episode_timesteps = np.arange(start_timestep, start_timestep + self.num_previous_states)
            else:
                episode_timesteps = np.zeros(self.num_previous_states)
            
            previous_states_batch.append(states_normalized)
            previous_actions_batch.append(actions_normalized)
            previous_states_mask_batch.append(state_mask)
            previous_actions_mask_batch.append(action_mask)
            episode_timesteps_batch.append(episode_timesteps)
        
        # Convert to tensors
        previous_states = torch.from_numpy(np.stack(previous_states_batch)).float().to(self.device)
        previous_actions = torch.from_numpy(np.stack(previous_actions_batch)).float().to(self.device)
        previous_states_mask = torch.from_numpy(np.stack(previous_states_mask_batch)).bool().to(self.device)
        previous_actions_mask = torch.from_numpy(np.stack(previous_actions_mask_batch)).bool().to(self.device)
        episode_timesteps = torch.from_numpy(np.stack(episode_timesteps_batch)).long().to(self.device)
        
        initial_noise_tensor = torch.from_numpy(initial_noise).float().to(self.device) if isinstance(initial_noise, np.ndarray) else initial_noise
        
        # Use TrainDiffusionPolicy's diffusion_sample method
        with torch.no_grad():
            diffused_actions = self.trainer.diffusion_sample(
                previous_states=previous_states,
                previous_actions=previous_actions,
                episode_timesteps=episode_timesteps,
                previous_states_padding_mask=previous_states_mask,
                previous_actions_padding_mask=previous_actions_mask,
                actions_padding_mask=None,
                max_action_len=act_steps,
                initial_noise=initial_noise_tensor
            )
        
        # Unnormalize actions
        diffused_actions_np = diffused_actions.cpu().numpy()
        diffused_actions_unnorm = self.trainer.unnormalize_actions(diffused_actions_np)
        
        # Flatten actions for SB3 format: (batch_size, act_steps * action_dim)
        diffused_actions_flat = diffused_actions_unnorm.reshape(batch_size, -1)
        
        if return_numpy:
            return diffused_actions_flat
        else:
            return torch.from_numpy(diffused_actions_flat).to(self.device)
    
    def reset(self):
        """Reset state/action history (call when environment resets)"""
        self.state_history = []
        self.action_history = []
		
def load_base_policy(cfg):
    device = cfg.model.device
    
    # Load checkpoint
    checkpoint = torch.load(cfg.base_policy_path, map_location=device, weights_only=False)
    
    # Load model
    model = PolicyDiffusionTransformer(
        num_transformer_layers=6,
        hidden_size=128,
        n_transformer_heads=1,
        state_dim=cfg.obs_dim,
        act_dim=cfg.action_dim,
        max_episode_length=900,
        device=device,
    )
    
    # Load model weights from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Backward compatibility: old checkpoints only have model state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create trainer with dummy data (only for interface, normalization stats loaded from checkpoint)
    dummy_states = np.zeros((1, 1, cfg.obs_dim))
    dummy_actions = np.zeros((1, 1, cfg.action_dim))
    
    trainer = TrainDiffusionPolicy(
        env=None,
        model=model,
        optimizer=None,
        states_array=dummy_states,
        actions_array=dummy_actions,
        device=device,
		multi_goal=True,
		load_checkpoint=True
    )
    
    # Load normalization stats from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        trainer.s_min = checkpoint['s_min']
        trainer.s_max = checkpoint['s_max']
        trainer.a_min = checkpoint['a_min']
        trainer.a_max = checkpoint['a_max']
        trainer.low_var_dims_states = checkpoint['low_var_dims_states']
        trainer.low_var_means_states = checkpoint['low_var_means_states']
        trainer.low_var_dims_actions = checkpoint['low_var_dims_actions']
        trainer.low_var_means_actions = checkpoint['low_var_means_actions']
    
    base_policy = PandaDiffusionPolicyWrapper(
        trainer=trainer,
        num_previous_states=10,
        num_previous_actions=9
    )
    
    return base_policy


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					wandb.log({
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in tqdm(range(num_steps), desc="Collecting rollouts"):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
			action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()