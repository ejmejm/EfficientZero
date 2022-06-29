# import os
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
# from env_wrapper import AtariWrapper
# from model import EfficientZeroNet

import torch

from core.config import BaseConfig
from core.utils import make_minigrid, WarpFrame
from core.dataset import Transforms
from ..general.env_wrapper import AtariWrapper
from ..general.model import EfficientZeroNet


class MinigridDebugConfig(BaseConfig):
    def __init__(self):
        super(MinigridDebugConfig, self).__init__(
            training_steps=30_000,
            last_steps=2000,
            test_interval=500,
            log_interval=200,
            vis_interval=1000,
            test_episodes=5,
            checkpoint_interval=1000,
            target_model_interval=200,
            save_ckpt_interval=1000,
            max_moves=200,
            test_max_moves=200,
            history_length=200,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=10,
            batch_size=256,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=True,
            # storage efficient
            cvt_string=True,
            image_based=True,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=1000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=1, # 4,
            total_transitions=100_000,
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

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
        self.repr_shape = (8, 8)
        self.repr_channels = 1
        self.discretize_type = None
        # self.vq_params = {'codebook_size': self.repr_channels}

        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)

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
        # gray scale
        if self.gray_scale:
            self.image_channel = 1

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.repr_shape,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm,
            discretize_type=self.discretize_type,
            repr_channels=self.repr_channels,
            vq_params=self.vq_params,)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        if test:
            env = make_minigrid(self.env_name, skip=self.frame_skip,
                max_episode_steps=self.test_max_moves, partial_obs=False)
        else:
            env = make_minigrid(self.env_name, skip=self.frame_skip,
                max_episode_steps=self.max_moves, partial_obs=False)

        orig_size = env.observation_space.shape
        self.obs_shape = (self.image_channel * self.stacked_observations, orig_size[0], orig_size[1])
        env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return AtariWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)


game_config = MinigridDebugConfig()
# game_config.set_game('MiniGrid-MultiRoom-N2-S4-v0')
# env = game_config.new_game()