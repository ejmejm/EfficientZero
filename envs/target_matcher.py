import gym
import numpy as np

# Create the environment using the gym interface
class TargetMatcher(gym.Env):
    def __init__(
            self, n_targets=10, n_distractors=0, n_buttons=5,
            min_changes=0, max_changes=5, max_timesteps=100):
        self.n_targets = n_targets
        self.n_distractors = n_distractors
        self.n_state_vals = n_targets + n_distractors
        self.n_buttons = n_buttons
        self.min_changes = min_changes
        self.max_changes = min(max_changes, self.n_state_vals)
        self.min_val = -1
        self.max_val = 1
        self.state = None # [target & distractor mix + targets + target positions + button combos]
        self.max_timesteps = max_timesteps

        self.action_space = gym.spaces.Discrete(n_buttons + 1)
        self.observation_space = gym.spaces.Box(
            low=self.min_val, high=self.max_val,
            shape=(2*n_targets + (1+n_buttons)*self.n_state_vals,))
    
    def _get_state(self):
        return np.concatenate([self.state_vals, self.targets,
            self.target_positions, self.button_vals.flatten()])

    def reset(self):
        self.timestep = 0

        val_range = self.max_val - self.min_val
        # State vals and targets
        self.state_vals = np.random.rand(self.n_state_vals) * val_range + self.min_val
        self.targets = np.random.rand(self.n_targets) * val_range + self.min_val

        # Target positions
        self.target_idxs = np.arange(self.n_state_vals)
        np.random.shuffle(self.target_idxs)
        self.target_idxs = np.array([i for i, x in enumerate(self.target_idxs) if x < self.n_targets])
        np.random.shuffle(self.target_idxs)
        self.target_positions = self.target_idxs / len(self.state_vals)

        # Button vals
        self.button_vals = (np.random.rand(self.n_buttons, self.n_state_vals) - 0.5) * val_range / 2
        self.button_vals /= 2
        change_counts = np.random.randint(self.min_changes, self.max_changes+1, self.n_buttons)
        for i in range(self.n_buttons):
            mask = np.zeros_like(self.button_vals[i])
            mask[np.random.choice(len(mask), change_counts[i], replace=False)] = 1
            self.button_vals[i] *= mask

        self.state = self._get_state()
        return self.state

    def step(self, action):
        if self.timestep >= self.max_timesteps:
            raise Exception('The episode is done, need to reset!')

        self.timestep += 1
        if action == 0:
            reward = 0
        else:
            prev_dist = np.sum(np.abs(self.state_vals[self.target_idxs] - self.targets))
            # Update state
            self.state_vals += self.button_vals[action-1]
            self.state_vals = np.clip(self.state_vals, self.min_val, self.max_val)
            new_dist = np.sum(np.abs(self.state_vals[self.target_idxs] - self.targets))
            reward = prev_dist - new_dist
            self.state = self._get_state()
        done = self.timestep >= self.max_timesteps

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print('State values:', [f'{x:.2f}' for x in self.state_vals])
        print('Targets:', [f'{x:.2f} ({i})' for x, i in zip(self.targets, self.target_idxs)])
        print('Buttons:', [[f'{x:.2f}' for x in b] for b in self.button_vals])

class SimplifyTMObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # State => [ordered non-distractor vals + targets + button target vals]
        self.observation_space = gym.spaces.Box(
            low=env.min_val, high=env.max_val,
            shape=(2*env.n_targets + env.n_buttons*env.n_targets,))

    def observation(self, _):
        self.curr_vals = self.env.state_vals[self.env.target_idxs]
        self.targets = self.env.targets
        self.buttons = np.array([b[self.env.target_idxs] for b in self.env.button_vals])
        return np.concatenate([self.curr_vals, self.targets, self.buttons.flatten()])

    def render(self, mode='human'):
        print('State values:', [f'{x:.2f}' for x in self.curr_vals])
        print('Targets:', [f'{x:.2f}' for x in self.targets])
        print('Buttons:', [[f'{x:.2f}' for x in b] for b in self.buttons])


if __name__ == '__main__':
    def make_env():
        env = TargetMatcher(n_targets=5, n_distractors=5, n_buttons=3, max_timesteps=50)
        # env = TargetMatcher(n_targets=5, n_distractors=0, n_buttons=3, max_timesteps=50)
        # env = SimplifyTMObs(env)
        return env

    # obs = env.reset()
    # env.render()
    # reward = env.step(1)[1]
    # print('Reward:', reward)
    # env.render()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    # Parallel environments
    env = make_vec_env(make_env, n_envs=8)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(5e7))