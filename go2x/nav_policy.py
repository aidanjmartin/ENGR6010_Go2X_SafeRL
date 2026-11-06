"""Navigation actor-critic for RSL-RL 3.1.2: a small convolutional encoder for the stacked top-down image
plus an MLP. RSL-RL 3.1.2 ships only MLP policies (`rsl_rl/modules/actor_critic.py`), so this subclass keeps its
distribution, normalization and PPO interface and replaces the actor and critic networks.

Observation layout (flat, from the navigation environment): the image (stack x 4 x S x S) first, then the
previous normalized velocity target (3).
"""
import torch
from rsl_rl.modules import ActorCritic
from torch import nn  # noqa: F401

from go2x.nav_net import ImageMLP


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
