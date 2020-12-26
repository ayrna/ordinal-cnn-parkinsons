"""Network Architectures

All the network architectures are defined in this file as PyTorch nn.Module's, as well
as the loss function used for the ordinal methodology.

- ConvNet: Convolutional part, common part of both nominal and ordinal architectures
- NominalDenseNet, OrdinalDenseNet: hidden fully connected parts of the nominal and ordinal
    architectures, respectively.
- NominalNet, OrdinalNet: Combination of convolutional and hidden fully connected parts of
    the nominal and ordinal architectures (ConvNet + NominalDenseNet, ConvNet + OrdinalDenseNet)
- ordinal_distance_loss: ordinal loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from typing import List, Tuple


def conv_output_shape(input_shape, kernel_size, stride, padding=0, dilation=1):
    """
    Compute output shape of convolution
    """
    return np.floor(((input_shape + 2*padding - dilation * (kernel_size-1) - 1) / stride) + 1)


class BrainNet(nn.Module):
    convnet: nn.Module
    densenet: nn.Module

    def predict(self, x):
        self.eval()
        x = self.convnet(x)
        return self.densenet.predict(x)


class ConvNet(nn.Module):

    def __init__(self, image_shape: Tuple[int, ...], n_channels: List[int], kernel_size: int, stride: int):
        super(ConvNet, self).__init__()

        self.n_channels = n_channels
        inout = zip(self.n_channels[:-1], self.n_channels[1:])
        self.convs = list()
        shape = np.array(image_shape)
        for in_channels, out_channels in inout:
            conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride)
            shape = conv_output_shape(shape, kernel_size, stride)

            self.convs.append(conv_layer)
        self.batch_norms = [nn.BatchNorm3d(nc, eps=1e-3, momentum=0.01) for nc in self.n_channels[1:]]

        self.convs = nn.ModuleList(self.convs)
        self.batch_norms = nn.ModuleList(self.batch_norms)

        self.conv_output_size = self.n_channels[-1] * np.prod(shape).astype(int)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.leaky_relu(bn(conv(x)))

        x = x.view(-1, self.conv_output_size)
        return x


class NominalDenseNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, dropout_rate: float):
        super(NominalDenseNet, self).__init__()
        self.dense_hidden = nn.Linear(input_size, hidden_size)
        self.dense_hidden_dropout = nn.Dropout(dropout_rate)
        self.dense_output = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = F.leaky_relu(self.dense_hidden(x))
        x = self.dense_hidden_dropout(x)
        x = self.dense_output(x)
        return x

    def predict(self, x):
        self.eval()
        outputs = self.forward(x).detach().cpu().numpy()
        labels = outputs.argmax(axis=1)
        return labels


class NominalNet(BrainNet):
    def __init__(self, image_shape: Tuple[int, ...], n_channels: List[int], kernel_size: int, stride: int,
                 hidden_size: int, n_classes: int, dropout_rate: float):
        super(NominalNet, self).__init__()
        self.convnet = ConvNet(image_shape, n_channels, kernel_size, stride)
        self.densenet = NominalDenseNet(self.convnet.conv_output_size, hidden_size,
                                        n_classes, dropout_rate)

    def forward(self, x):
        x = self.convnet(x)
        x = self.densenet(x)
        return x


class OrdinalDenseNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, dropout_rate: float):
        super(OrdinalDenseNet, self).__init__()

        hidden_size_per_unit = np.round(hidden_size / (n_classes - 1)).astype(int)
        self.dense_hidden = nn.ModuleList(
            [nn.Linear(input_size, hidden_size_per_unit) for _ in range(n_classes - 1)]
        )
        self.dense_hidden_dropout = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(n_classes - 1)]
        )
        self.dense_output = nn.ModuleList(
            [nn.Linear(hidden_size_per_unit, 1) for _ in range(n_classes - 1)]
        )

        # Reference vectors for each class, for predictions
        self.target_class = np.ones((n_classes, n_classes - 1), dtype=np.float32)
        self.target_class[np.triu_indices(n_classes, 0, n_classes - 1)] = 0.0

    def forward(self, x):
        xs = [drop(F.leaky_relu(hidden(x))) for hidden, drop in zip(self.dense_hidden, self.dense_hidden_dropout)]
        xs = [torch.sigmoid(output(xc))[:, 0] for output, xc in zip(self.dense_output, xs)]
        return xs

    def predict(self, x):
        self.eval()
        x = self.forward(x)
        outputs = torch.cat([o.unsqueeze(dim=1) for o in x], dim=1).detach().cpu().numpy()
        distances = cdist(outputs, self.target_class, metric='euclidean')
        labels = distances.argmin(axis=1)
        return labels


class OrdinalNet(BrainNet):
    def __init__(self, image_shape: Tuple[int, ...], n_channels: List[int], kernel_size: int, stride: int,
                 hidden_size: int, n_classes: int, dropout_rate: float):
        super(OrdinalNet, self).__init__()
        self.convnet = ConvNet(image_shape, n_channels, kernel_size, stride)
        self.densenet = OrdinalDenseNet(self.convnet.conv_output_size, hidden_size,
                                        n_classes, dropout_rate)

    def forward(self, x):
        x = self.convnet(x)
        x = self.densenet(x)
        return x


def ordinal_distance_loss(n_classes, device):
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = torch.tensor(target_class, device=device)
    mse = nn.MSELoss(reduction='sum')

    def _ordinal_distance_loss(net_output, target):
        net_output = torch.stack(net_output, dim=1)
        target = target_class[target]
        return mse(net_output, target)

    return _ordinal_distance_loss
