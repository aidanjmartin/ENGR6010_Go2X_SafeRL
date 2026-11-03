"""The CNN actor-critic plugs into RSL-RL 3.1.2's PPO interface."""
import torch
from tensordict import TensorDict

from go2x.nav_policy import CNNActorCritic, ImageMLP


def test_image_mlp_uses_image_and_extra():
    torch.manual_seed(0)
    net = ImageMLP((8, 64, 64), 3, 2)
    x = torch.randn(5, 8 * 64 * 64 + 3)
    y = net(x)
    assert y.shape == (5, 2)
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(net(x2), y)          # the previous action reaches the head
    x3 = x.clone()
    x3[:, :100] += 1.0
    assert not torch.allclose(net(x3), y)          # the image reaches the head


def test_actor_critic_interface():
    B = 4
    obs = TensorDict({"policy": torch.rand(B, 8 * 64 * 64 + 3)}, batch_size=[B])
    ac = CNNActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 3, init_noise_std=0.5)
    a = ac.act(obs)
    assert a.shape == (B, 3)
    assert ac.get_actions_log_prob(a).shape == (B,)
    assert ac.evaluate(obs).shape == (B, 1)
    assert ac.act_inference(obs).shape == (B, 3)
    assert torch.allclose(ac.action_std, torch.full((B, 3), 0.5))
    sd = ac.state_dict()
    ac2 = CNNActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, 3)
    ac2.load_state_dict(sd)
    assert torch.allclose(ac2.act_inference(obs), ac.act_inference(obs))


def test_critic_reads_its_own_group():
    """Job 050 attempt 2: the critic gets the policy input plus 6 critic-only values; the actor ignores them."""
    B = 4
    pol = torch.rand(B, 8 * 64 * 64 + 3)
    obs = TensorDict({"policy": pol, "critic": torch.rand(B, 6)}, batch_size=[B])
    ac = CNNActorCritic(obs, {"policy": ["policy"], "critic": ["policy", "critic"]}, 3)
    assert ac.evaluate(obs).shape == (B, 1)
    obs2 = obs.clone()
    obs2["critic"][:, -1] += 1.0
    assert torch.allclose(ac.act_inference(obs2), ac.act_inference(obs))
    assert not torch.allclose(ac.evaluate(obs2), ac.evaluate(obs))
