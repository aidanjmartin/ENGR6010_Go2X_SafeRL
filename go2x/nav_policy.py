"""Navigation actor-critic for RSL-RL 3.1.2: a small convolutional encoder for the stacked top-down image
plus an MLP. RSL-RL 3.1.2 ships only MLP policies (`rsl_rl/modules/actor_critic.py`), so this subclass keeps its
distribution, normalization and PPO interface and replaces the actor and critic networks.

Observation layout (flat, from the navigation environment): the image (stack x 4 x S x S) first, then the
previous normalized velocity target (3).
"""
import torch
from rsl_rl.modules import ActorCritic
from torch import nn


class ImageMLP(nn.Module):
    def __init__(self, image_shape, extra_dim, out_dim, channels=(16, 32, 32), cnn_out=128, mlp=(128, 64)):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.image_numel = int(torch.tensor(image_shape).prod())
        layers, c_in = [], image_shape[0]
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.ELU()]
            c_in = c
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *image_shape)).shape[-1]
        self.proj = nn.Sequential(nn.Linear(n, cnn_out), nn.ELU())
        dims, d_in = [], cnn_out + extra_dim
        for h in mlp:
            dims += [nn.Linear(d_in, h), nn.ELU()]
            d_in = h
        self.head = nn.Sequential(*dims, nn.Linear(d_in, out_dim))

    def forward(self, x):
        img = x[:, : self.image_numel].reshape(-1, *self.image_shape)
        extra = x[:, self.image_numel:]
        return self.head(torch.cat([self.proj(self.cnn(img)), extra], -1))


class CNNActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, image_shape=(8, 64, 64), cnn_channels=(16, 32, 32), cnn_out=128,
                 mlp=(128, 64), **kwargs):
        kwargs.pop("actor_hidden_dims", None)
        kwargs.pop("critic_hidden_dims", None)
        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=[8], critic_hidden_dims=[8], **kwargs)
        n_actor = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        n_critic = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        numel = int(torch.tensor(image_shape).prod())
        self.actor = ImageMLP(image_shape, n_actor - numel, num_actions, cnn_channels, cnn_out, mlp)
        self.critic = ImageMLP(image_shape, n_critic - numel, 1, cnn_channels, cnn_out, mlp)
        print(f"CNN actor: {self.actor}")


def register_with_rsl_rl():
    """RSL-RL's OnPolicyRunner builds the policy with eval(class_name) in its own module namespace."""
    import rsl_rl.runners.on_policy_runner as opr

    opr.CNNActorCritic = CNNActorCritic
