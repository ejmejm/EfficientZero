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


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

# Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = n_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(n_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def decode(self, encoding_indices):
      batch_size, n_latents = encoding_indices.shape
      encoding_indices = encoding_indices.reshape(-1, 1)
      encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
      encodings.scatter_(1, encoding_indices, 1)
      # Quantize and unflatten
      quantized = torch.matmul(encodings, self._embedding.weight).view(batch_size, n_latents, self._embedding_dim)
      return quantized.permute(0, 2, 1)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

def discretize(x, discretize_type, **kwargs):
    loss = 0
    if discretize_type == 'gumbel':
        x = F.gumbel_softmax(x, hard=True, dim=1)
    elif discretize_type == 'stochastic':
        in_shape = x.shape
        n_channels = in_shape[1]
        probs = F.softmax(x, dim=1)
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, n_channels)
        samples = torch.multinomial(flat_probs, 1).squeeze()
        one_hot_samples = F.one_hot(samples, num_classes=n_channels).float()
        one_hot_samples = one_hot_samples.reshape(in_shape[0], in_shape[2], in_shape[3], in_shape[1])
        one_hot_samples = one_hot_samples.permute(0, 3, 1, 2)
        x = probs + (one_hot_samples - probs).detach() # Straight-though gradients
        x = x.contiguous()
    elif discretize_type == 'vq':
        quantizer = kwargs['vq_model']
        loss, quantized, _, _ = quantizer(x)
        x = quantized
    elif discretize_type == 'vq_one_hot':
        quantizer = kwargs['vq_model']
        loss, _, _, encodings = quantizer(x)
        encodings = encodings.reshape(*x.shape)
        x = encodings
    else:
        raise ValueError('Unknown discretize type: {}'.format(discretize_type))
    return x, loss

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2, downsample=self.conv2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = nn.AdaptiveAvgPool2d(out_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        out_shape,
        downsample,
        momentum=0.1,
        discretize_type=None,
        repr_channels=None,
        vq_model=None,
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
        if discretize_type and discretize_type.startswith('vq'):
            assert vq_model is not None, 'vq_model is None but discretize_type is vq'
        self.discretize_type = discretize_type
        self.vq_model = vq_model
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
                out_shape,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )
        repr_channels = repr_channels or num_channels
        self.repr_conv = conv3x3(num_channels, repr_channels)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        x = self.repr_conv(x)
        disc_loss = 0
        if self.discretize_type is not None:
            x, disc_loss = discretize(x, self.discretize_type, vq_model=self.vq_model)
        return x, disc_loss

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
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
        discretize_type=None,
        repr_channels=None,
        vq_model=None,
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
        if discretize_type and discretize_type.startswith('vq'):
            assert vq_model is not None, 'vq_model is None but discretize_type is vq'
        self.discretize_type = discretize_type
        self.vq_model = vq_model

        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        repr_channels = repr_channels or num_channels
        self.conv1x1_repr = nn.Conv2d(repr_channels, num_channels, 1)
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.repr_conv = conv3x3(num_channels - 1, repr_channels - 1)

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-1,:,:]
        x = self.conv1x1_repr(x)
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x
        state = self.repr_conv(state)
        disc_loss = 0
        if self.discretize_type is not None:
            state, disc_loss = discretize(state, self.discretize_type, vq_model=self.vq_model)

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)

        x = x.reshape(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix, disc_loss

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        momentum=0.1,
        init_zero=False,
        repr_channels=None,
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
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )
        repr_channels = repr_channels or num_channels
        self.conv1x1_repr = nn.Conv2d(repr_channels, num_channels, 1)
        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        x = self.conv1x1_repr(x)
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = nn.functional.relu(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)

        value = value.reshape(-1, self.block_output_size_value)
        policy = policy.reshape(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        num_channels,
        out_shape,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        downsample,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False,
        discretize_type=None,
        repr_channels=None,
        vq_params={},
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
        block_output_size_reward = (np.prod([reduced_channels_reward, *out_shape])
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2]))

        block_output_size_value = (np.prod([reduced_channels_value, *out_shape])
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2]))

        block_output_size_policy = (np.prod([reduced_channels_policy, *out_shape])
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2]))

        repr_channels = repr_channels or num_channels
        self.vq_model = None
        self.disc_cum_loss = 0
        self.disc_loss_count = 0
        if discretize_type and discretize_type.startswith('vq'):
            codebook_size = vq_params.get('codebook_size', 64)
            if discretize_type == 'vq_one_hot':
                assert codebook_size == repr_channels, 'codebook_size must be equal to repr_channels ' + \
                    f'({repr_channels}) for vq_one_hot'

            self.vq_model = VectorQuantizerEMA(
                codebook_size,
                repr_channels,
                commitment_cost = vq_params.get('commitment_cost', 0.25),
                decay = vq_params.get('decay', 0.99))

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_blocks,
            num_channels,
            out_shape,
            downsample,
            momentum=bn_mt,
            discretize_type=discretize_type,
            repr_channels=repr_channels,
            vq_model=self.vq_model,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels + 1,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
            discretize_type=discretize_type,
            repr_channels=repr_channels + 1,
            vq_model=self.vq_model,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            block_output_size_value,
            block_output_size_policy,
            momentum=bn_mt,
            init_zero=self.init_zero,
            repr_channels=repr_channels,
        )

        # projection
        in_dim = np.prod([repr_channels, *out_shape])
        self.projection_in_dim = in_dim
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
        encoded_state, disc_loss = self.representation_network(observation)
        if self.training:
            self.disc_cum_loss += disc_loss
            self.disc_loss_count += 1
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix, disc_loss = self.dynamics_network(x, reward_hidden)

        if self.training:
            self.disc_cum_loss += disc_loss
            self.disc_loss_count += 1

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def reset_disc_losses(self):
        mean_disc_loss = self.disc_cum_loss / self.disc_loss_count
        self.disc_cum_loss = 0
        self.disc_loss_count = 0
        return mean_disc_loss

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