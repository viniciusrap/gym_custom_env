"""
cpp_policy.py
-------------
Custom feature extractor for the CPP environment.

What it does
------------
The 3x3 local view comes in as integer values in {0, 1, 2}. We:
  1. Convert to one-hot (so the network does NOT treat 0/1/2 as ordered scalars).
  2. Run a small CNN to extract spatial features.
  3. Concatenate with an MLP-encoded version of the scalar inputs (agent_pos,
     coverage).

Why we do not just flatten the local view
-----------------------------------------
The default `MultiInputPolicy` of Stable-Baselines3 flattens every Dict
component and concatenates them. That destroys the spatial topology of the
3x3 neighborhood. A 3-channel CNN is overkill for a 3x3 window, but it
preserves the structure correctly and adds essentially zero compute.

Frame stacking compatibility
----------------------------
We use `VecFrameStack(n_stack=4)`. With a (3, 3) integer matrix observation
SB3 stacks along the last axis -> the local_view tensor that arrives here has
shape (batch, 3, 3 * n_stack).  We unstack, one-hot each frame, and concatenate
all frames as channels of the CNN input (so out CNN sees `3 * n_stack` channels).

The agent_pos and coverage scalars are also frame-stacked (shape doubles); we
keep all stacked copies — they encode trajectory information for free.

Why this enables transfer learning
----------------------------------
Every input component has a fixed shape regardless of the grid size, so the
SAME network weights are valid for 5x5, 10x10 and 20x20 environments. The
PPO model can be loaded from a 5x5 checkpoint into a 10x10 environment and
continue training without any architectural change.
"""

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


NUM_CATEGORIES = 3   # {0=free, 1=obstacle, 2=visited}


class CPPFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_features: int = 64,
        scalar_features: int = 32,
    ):
        super().__init__(
            observation_space, features_dim=cnn_features + scalar_features
        )

        # --- detect frame stacking from the observed shape ---
        # local_view shape coming in:
        #   without stacking: (3, 3)
        #   with    stacking: (3, 3 * n_stack)
        local_shape = observation_space["local_view"].shape
        assert len(local_shape) == 2 and local_shape[0] == 3, (
            f"unexpected local_view shape {local_shape}"
        )
        # Number of stacked frames implied by the observation shape.
        if local_shape[1] % 3 != 0:
            raise ValueError(
                f"local_view second-axis ({local_shape[1]}) must be a multiple "
                f"of 3 (frame width). Got shape {local_shape}."
            )
        self.n_stack = local_shape[1] // 3
        in_channels = NUM_CATEGORIES * self.n_stack   # one-hot channels per frame * frames

        # --- CNN branch on a 3x3 spatial window with `in_channels` channels ---
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 3x3 -> 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),           # 3x3 -> 1x1
            nn.ReLU(inplace=True),
            nn.Flatten(),                                          # -> (batch, 64)
        )
        self.cnn_head = nn.Sequential(
            nn.Linear(64, cnn_features),
            nn.ReLU(inplace=True),
        )

        # --- scalar branch ---
        scalar_dim = (
            observation_space["agent_pos"].shape[0]
            + observation_space["coverage"].shape[0]
        )  # already includes any frame-stacking duplication
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, scalar_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # local_view: (batch, 3, 3 * n_stack) of integers in {0, 1, 2}
        local = observations["local_view"].long()
        batch = local.shape[0]
        # Reshape to (batch, 3, n_stack, 3) -> (batch, n_stack, 3, 3)
        local = local.view(batch, 3, self.n_stack, 3).permute(0, 2, 1, 3).contiguous()
        # one-hot -> (batch, n_stack, 3, 3, NUM_CAT)
        one_hot = F.one_hot(local, num_classes=NUM_CATEGORIES).float()
        # Move classes axis to be channels and merge n_stack with NUM_CAT
        # -> (batch, n_stack * NUM_CAT, 3, 3)
        one_hot = one_hot.permute(0, 1, 4, 2, 3).contiguous()
        one_hot = one_hot.view(batch, self.n_stack * NUM_CATEGORIES, 3, 3)

        cnn_out = self.cnn(one_hot)
        cnn_out = self.cnn_head(cnn_out)

        scalars = torch.cat(
            [observations["agent_pos"], observations["coverage"]], dim=1
        )
        scalar_out = self.scalar_mlp(scalars)

        return torch.cat([cnn_out, scalar_out], dim=1)
