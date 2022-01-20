from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
import clip


class DenseClip(nn.Module):

    def __init__(self, name: str, device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 jit: bool = False, download_root: str = None):
        super(DenseClip, self).__init__()
        self.clip_model, self.preprocess = clip.load(name, device, jit, download_root)
        self.visual = self.clip_model.visual
        self.conv1 = nn.Conv2d(self.visual.attnpool.v_proj.in_features,
                               self.visual.attnpool.v_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)
        self.conv2 = nn.Conv2d(self.visual.attnpool.c_proj.in_features,
                               self.visual.attnpool.c_proj.out_features,
                               kernel_size=(1, 1)).to(device).to(self.dtype)

        conv1_weight_shape = (*self.visual.attnpool.v_proj.weight.shape, 1, 1)
        conv2_weight_shape = (*self.visual.attnpool.c_proj.weight.shape, 1, 1)
        self.conv1.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.v_proj.weight.reshape(conv1_weight_shape),
                        bias=self.visual.attnpool.v_proj.bias))
        self.conv2.load_state_dict(
            OrderedDict(weight=self.visual.attnpool.c_proj.weight.reshape(conv2_weight_shape),
                        bias=self.visual.attnpool.c_proj.bias))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _stem(self, x):
        for conv, bn in [(self.visual.conv1, self.visual.bn1),
                         (self.visual.conv2, self.visual.bn2),
                         (self.visual.conv3, self.visual.bn3)]:
            x = self.visual.relu(bn(conv(x)))

        x = self.visual.avgpool(x)
        return x

    def encode_image(self, image):
        image = image.type(self.dtype)
        feature = self._stem(image)
        feature = self.visual.layer1(feature)
        feature = self.visual.layer2(feature)
        feature = self.visual.layer3(feature)
        feature = self.visual.layer4(feature)

        # removed attnpool
        feature = self.conv1(feature)
        feature = self.conv2(feature)

        return feature

    def forward(self, x):
        pass

    @staticmethod
    def available_models():
        return clip.available_models()
