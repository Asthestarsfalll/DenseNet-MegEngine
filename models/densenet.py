from collections import OrderedDict
from typing import Any, Tuple

import megengine as mge
from megengine import hub
import megengine.functional as F
import megengine.module as M
from megengine import Tensor


class DenseLayer(M.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float = 0.0,
    ):
        super(DenseLayer, self).__init__()
        hidden_dim = bn_size * growth_rate
        self.norm1 = M.BatchNorm2d(num_input_features)
        self.relu1 = M.ReLU()
        self.conv1 = M.Conv2d(
            num_input_features,
            hidden_dim,
            kernel_size=1,
            stride=1,
            bias=False)
        self.norm2 = M.BatchNorm2d(hidden_dim)
        self.relu2 = M.ReLU()
        self.conv2 = M.Conv2d(
            hidden_dim,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.drop = M.Dropout(drop_rate)

    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        concated_features = F.concat(inputs, axis=1)
        bottleneck_output = self.conv1(
            self.relu1(self.norm1(concated_features)))

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        return self.drop(new_features)


class DenseBlock(M.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            setattr(self, f'denselayer{(i + 1)}', layer)

    def forward(self, init_features):
        features = [init_features]
        for i in range(self.num_layers):
            layer = getattr(self, f'denselayer{(i + 1)}')
            new_features = layer(features)
            features.append(new_features)
        return F.concat(features, axis=1)


class Transition(M.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        layer_dict = OrderedDict([
            ('norm', M.BatchNorm2d(num_input_features)),
            ('relu', M.ReLU()),
            ('conv', M.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False)),
            ('pool', M.AvgPool2d(kernel_size=2, stride=2))
        ])
        super(Transition, self).__init__(layer_dict)


class DenseNet(M.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ):

        super(DenseNet, self).__init__()

        # First convolution
        features = [
            ('conv0', M.Conv2d(
                in_channels=3,
                out_channels=num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)),
            ('norm0', M.BatchNorm2d(num_init_features)),
            ('relu0', M.ReLU()),
            ('pool0', M.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            features.append((f'denseblock{i + 1}', block))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                features.append((f'transition{i + 1}', trans))
                num_features = num_features // 2

        # Final batch norm
        features.append(('norm5', M.BatchNorm2d(num_features)))
        # Linear layer
        self.classifier = M.Linear(num_features, num_classes)

        # make features
        self.features = M.Sequential(OrderedDict(features))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, M.Conv2d):
            M.init.msra_normal_(m.weight)
        elif isinstance(m, M.BatchNorm2d):
            M.init.ones_(m.weight)
            M.init.zeros_(m.bias)
        elif isinstance(m, M.Linear):
            M.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, )
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = F.flatten(out, 1)
        out = self.classifier(out)
        return out


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    **kwargs: Any
):
    return DenseNet(growth_rate, block_config, num_init_features, **kwargs)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/105/files/ccd24ece-92bb-4ffb-986f-21904ceb286e"
)
def densenet121(**kwargs: Any):
    return DenseNet(32, (6, 12, 24, 16), 64, **kwargs)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/105/files/a8051ccd-57b8-4f37-9104-3b98f1445fb8"
)
def densenet161(**kwargs: Any):
    return DenseNet(48, (6, 12, 36, 24), 96, **kwargs)


def densenet169(**kwargs: Any):
    return DenseNet(32, (6, 12, 32, 32), 64, **kwargs)


def densenet201(**kwargs: Any):
    return DenseNet(32, (6, 12, 48, 32), 64, **kwargs)


if __name__ == '__main__':
    model = densenet121()
    inp = mge.random.normal(size=(2, 3, 224, 224))
    out = model(inp)
    print(out.shape)
