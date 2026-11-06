"""The navigation network body (a small convolutional encoder for the stacked top-down image, then an MLP).

Kept apart from go2x/nav_policy.py, which also needs RSL-RL, so the robot runner (go2x/deploy) loads the
exported actor with torch alone.
"""
import torch
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
