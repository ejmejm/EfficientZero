import numpy as np
import torch

from core.config import BaseConfig
from core.utils import make_minigrid, WarpFrame
from core.dataset import Transforms
from envs.target_matcher import TargetMatcher, SimplifyTMObs
from ..general.env_wrapper import AtariWrapper
from ..general.dense_model import EfficientZeroNet


class TargetMatcherConfig(BaseConfig):
    def __init__(self):
        super(TargetMatcherConfig, self).__init__(
            training_steps=100_000,
            last_steps=2000,
            test_interval=1000,
            log_interval=200,
            vis_interval=500,
            test_episodes=50,
            checkpoint_interval=1000,
            target_model_interval=200,
            save_ckpt_interval=1000,
            max_moves=50,
            test_max_moves=50,
            history_length=50,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=10,
            batch_size=128,
            td_steps=5,
            num_actors=2,
            # network initialization/ & normalization
            episode_life=False,
            init_zero=True,
            clip_reward=True,
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=30_000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=5,
            total_transitions=5_000_000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=1,
            stacked_observations=1,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            lstm_hidden_size=64,
            lstm_horizon_len=5,
            # siamese
            proj_hid=128,
            proj_out=128,
            pred_hid=64,
            pred_out=128,)

        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip

        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.image_channel = 1
        self.use_augmentation = False
        # representation network
        self.repr_size = 128
        self.repr_layer_sizes = [256, 256, 256]
        # dynamics network
        self.dynamics_layer_sizes = [128, 128, 128]
        self.fc_reward_layers = [64, 64]
        # prediction network
        self.prediction_layer_sizes = [128, 128]
        self.value_linear = 64
        self.policy_linear = 64
        self.fc_value_layers = [64, 64]
        self.fc_policy_layers = [64, 64]

        self.bn_mt = 0.1

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps + self.last_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps + self.last_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        self.obs_shape = game.observation_space.shape
        self.action_space_size = game.action_space.n

    def get_uniform_network(self):
        return EfficientZeroNet(
            np.prod(self.obs_shape),
            self.action_space_size,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            repr_size=self.repr_size,
            repr_layer_sizes=self.repr_layer_sizes,
            dynamics_layer_sizes=self.dynamics_layer_sizes,
            prediction_layer_sizes=self.prediction_layer_sizes,
            fc_reward_layers=self.fc_reward_layers,
            value_linear=self.value_linear,
            policy_linear=self.policy_linear,
            fc_value_layers=self.fc_value_layers,
            fc_policy_layers=self.fc_policy_layers,
            bn_mt=self.bn_mt,
            lstm_hidden_size=self.lstm_hidden_size,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        if test:
            env = TargetMatcher(n_targets=4, n_distractors=0, n_buttons=3, max_timesteps=self.test_max_moves)
        else:
            env = TargetMatcher(n_targets=4, n_distractors=0, n_buttons=3, max_timesteps=self.max_moves)
        env = SimplifyTMObs(env)
        # if seed is not None:
        #     env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        env.env = env
        return env
        # return AtariWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        raise NotImplementedError

    def transform(self, images):
        raise NotImplementedError


game_config = TargetMatcherConfig()