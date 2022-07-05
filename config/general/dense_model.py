# import os
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch

import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from core.model import BaseNet, renormalize


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)

# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        out_size=128,
        layer_sizes=[256, 256, 256],
        momentum=0.1
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.layers = mlp(observation_size, layer_sizes, out_size, momentum=momentum)

    def forward(self, x):
        return self.layers(x)

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean

# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        full_support_size,
        base_layers=[128, 128, 128],
        fc_reward_layers=[64, 64],
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size

        self.base_layers = mlp(input_size, base_layers[:-1], base_layers[-1])
        self.lstm = nn.LSTM(input_size=base_layers[-1], hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        x = self.base_layers(x)
        state = x

        value_prefix, reward_hidden = self.lstm(x.unsqueeze(0), reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = [0]
        for name, param in self.base_layers.named_parameters():
            dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
        reward_mean = np.abs(temp_weights).mean()
        return temp_weights, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        action_space_size,
        full_support_size,
        base_layers=[128, 128],
        value_linear=64,
        policy_linear=64,
        fc_value_layers=[64, 64],
        fc_policy_layers=[64, 64],
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()
        self.base_layers = mlp(input_size, base_layers[:-1], base_layers[-1])
        
        self.value_linear = nn.Linear(base_layers[-1], value_linear)
        self.policy_linear = nn.Linear(base_layers[-1], policy_linear)
        self.bn_value = nn.BatchNorm1d(value_linear, momentum=momentum)
        self.bn_policy = nn.BatchNorm1d(policy_linear, momentum=momentum)

        self.fc_value = mlp(value_linear, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(policy_linear, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        x = self.base_layers(x)

        value = self.value_linear(x)
        value = self.bn_value(value)
        value = nn.functional.relu(value)
        value = self.fc_value(value)

        policy = self.policy_linear(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)
        policy = self.fc_policy(policy)

        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_size,
        action_space_size,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        repr_size=128,
        repr_layer_sizes=[256, 256, 256],
        dynamics_layer_sizes=[128, 128, 128],
        prediction_layer_sizes=[128, 128],
        fc_reward_layers=[64, 64],
        value_linear=64,
        policy_linear=64,
        fc_value_layers=[64, 64],
        fc_policy_layers=[64, 64],
        bn_mt=0.1,
        lstm_hidden_size=64,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False,
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_reward: int
            channels of reward head
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm
        self.action_space_size = action_space_size

        self.representation_network = RepresentationNetwork(
            observation_size,
            repr_size,
            repr_layer_sizes,
            momentum=bn_mt,
        )

        self.dynamics_network = DynamicsNetwork(
            repr_size + action_space_size,
            reward_support_size,
            base_layers=dynamics_layer_sizes,
            fc_reward_layers=fc_reward_layers,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,)

        self.prediction_network = PredictionNetwork(
            repr_size,
            action_space_size,
            value_support_size,
            base_layers=prediction_layer_sizes,
            value_linear=value_linear,
            policy_linear=policy_linear,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=fc_policy_layers,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        self.projection_in_dim = repr_size
        self.projection = nn.Sequential(
            nn.Linear(self.projection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        action_one_hot = F.one_hot(action.squeeze(), num_classes=self.action_space_size).float()
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.projection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

#------------------------------------------------------------------------------------

# Testing
if __name__ == '__main__':
    rn = RepresentationNetwork(
            observation_shape = (3, 84, 84),
            num_blocks = 1,
            num_channels = 64,
            out_shape = (6, 6),
            downsample = True,
            momentum = 0.1,
            discretize_type = 'gumbel')
    print(rn)
    print('# Params:', sum(p.numel() for p in rn.parameters()))
    out = rn(torch.randn(10, 3, 84, 84))
    print(out.shape)
    print(out[0])

#------------------------------------------------------------------------------------