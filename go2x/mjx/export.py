"""Brax PPO walking policy -> TorchScript with the Isaac walker's interface (torch and numpy only at load time).

Brax's policy (brax 0.14.2, training/networks.py make_policy_network with tanh_normal): the 'state' observation
normalized as (x - mean) / std with the running statistics, then an MLP (Dense layers hidden_0 .. hidden_n,
swish between them, none after the last), whose last layer gives 2 x 12 values (mean, scale). The deterministic
action is tanh(mean). The exported module returns tanh(mean) * action_scale / interface_scale, so a runner that
sets target = default + 0.25 * output (go2x/sim2sim.py joint_targets) applies default + action_scale * tanh(mean).
"""
import numpy as np
import torch
from torch import nn


class BraxPolicy(nn.Module):
    def __init__(self, mean, std, layers, out_scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(np.array(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.array(std), dtype=torch.float32))
        self.layers = nn.ModuleList()
        for W, b in layers:
            lin = nn.Linear(W.shape[0], W.shape[1])
            with torch.no_grad():
                lin.weight.copy_(torch.tensor(np.array(W).T, dtype=torch.float32))
                lin.bias.copy_(torch.tensor(np.array(b), dtype=torch.float32))
            self.layers.append(lin)
        self.n_act = layers[-1][0].shape[1] // 2
        self.out_scale = float(out_scale)

    def forward(self, obs):
        x = (obs - self.mean) / self.std
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = x * torch.sigmoid(x)       # swish
        return torch.tanh(x[:, : self.n_act]) * self.out_scale


def _get(tree, key):
    return tree[key] if isinstance(tree, dict) else getattr(tree, key)


def brax_policy_module(params, cfg, obs_key="state"):
    normalizer, policy = params[0], params[1]
    mean, std = _get(normalizer, "mean"), _get(normalizer, "std")
    if isinstance(mean, dict):
        mean, std = mean[obs_key], std[obs_key]
    p = policy["params"] if "params" in policy else policy
    names = sorted(p, key=lambda k: int(k.split("_")[-1]))
    layers = [(np.asarray(p[k]["kernel"]), np.asarray(p[k]["bias"])) for k in names]
    a = cfg["actuator"]
    return BraxPolicy(np.asarray(mean), np.asarray(std), layers, a["action_scale"] / a["interface_scale"]).eval()


def export_torchscript(params, hidden, cfg, path):
    m = brax_policy_module(params, cfg)
    assert [l.out_features for l in m.layers[:-1]] == list(hidden), "layer sizes differ from the config"
    for q in m.parameters():
        q.requires_grad_(False)
    ts = torch.jit.script(m)
    ts.save(str(path))
    return ts
