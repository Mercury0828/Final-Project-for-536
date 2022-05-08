
from trojanvision.models import _ImageModel, ImageModel
import trojanvision.models.torchvision.resnet as resnet
import trojanvision.models.torchvision.efficientnet as efficientnet

from trojanvision.environ import env

import torch
import torch.nn as nn
# TODO: torchvision.ops.Conv2dNormActivation
from torchvision.ops.misc import ConvNormActivation
import torch.nn.functional as F
from collections import OrderedDict


class _CNN(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1_bn_relu', ConvNormActivation(
                in_channels=16, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv2_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv3_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv4_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
        ]))
        # self.features = nn.Sequential(OrderedDict([
        #     ('conv1_bn_relu', ConvNormActivation(
        #         in_channels=16, out_channels=32,
        #         kernel_size=4, stride=2, padding=1)),
        #     ('conv2_bn_relu', ConvNormActivation(
        #         in_channels=32, out_channels=32,
        #         kernel_size=4, stride=2, padding=1)),
        #     ('conv3_bn_relu', ConvNormActivation(
        #         in_channels=32, out_channels=32,
        #         kernel_size=4, stride=2, padding=1)),
        #     ('conv4_bn', ConvNormActivation(
        #         in_channels=32, out_channels=32,
        #         kernel_size=4, stride=2, padding=1,
        #         activation_layer=None)),
        # ]))
        self.pool = nn.Identity()
        self.classifier = self.define_classifier(
            num_features=[32*4*4, 512],
            num_classes=self.num_classes,
            dropout=0.5)


class CNN(ImageModel):
    available_models = ['cnn']

    def __init__(self, name: str = 'cnn', model=_CNN, **kwargs):
        super().__init__(name=name, model=model, **kwargs)


class _ResNet(resnet._ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fc: nn.Linear = self.classifier[0]
        self.classifier = self.define_classifier(
            num_features=[fc.in_features, 512],
            num_classes=8 + 9 + 21,
            dropout=0.5)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 224, 224)
        x = self.get_final_fm(x, **kwargs)  # (N, 512)
        x = self.classifier(x)  # (N, 8 + 9 + 21)
        return x[:, :8]  # (N, 8)

    def get_full_logits(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (N, 16, 224, 224)
        x = self.get_final_fm(x, **kwargs)  # (N, 512)
        x = self.classifier(x)  # (N, 8 + 9 + 21)
        pred = x[:, :8]  # (N, 8)
        meta_target_pred = x[:, 8: 17]  # (N, 9)
        meta_struct_pred = x[:, 17:]  # (N, 9)
        return pred, meta_target_pred, meta_struct_pred


class ResNet(resnet.ResNet):
    def __init__(self, model=_ResNet, **kwargs):
        super().__init__(model=model, **kwargs)


class _LSTM(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1_bn_relu', ConvNormActivation(
                in_channels=1, out_channels=16,
                kernel_size=3, stride=2, padding=0)),
            ('conv2_bn_relu', ConvNormActivation(
                in_channels=16, out_channels=16,
                kernel_size=3, stride=2, padding=0)),
            ('conv3_bn_relu', ConvNormActivation(
                in_channels=16, out_channels=16,
                kernel_size=3, stride=2, padding=0)),
            ('conv4_bn_relu', ConvNormActivation(
                in_channels=16, out_channels=16,
                kernel_size=3, stride=2, padding=0)),
        ]))
        self.pool = nn.Identity()
        self.flatten = nn.Identity()
        self.lstm = nn.LSTM(input_size=16*4*4, hidden_size=96,
                            num_layers=1, batch_first=True)
        self.classifier = self.define_classifier(
            num_features=[96, 96],
            num_classes=self.num_classes,
            dropout=0.5)

    def get_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 80, 80)
        return super().get_fm(x.flatten(0, 1).unsqueeze(1), **kwargs).view(
            x.size(0), 16, 16*4*4)
        # (N, 16, 256)

    def get_final_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.get_fm(x, **kwargs)
        hidden, _ = self.lstm(x)
        return hidden[:, -1]


class LSTM(ImageModel):
    available_models = ['lstm']

    def __init__(self, name: str = 'lstm', model=_LSTM, **kwargs):
        super().__init__(name=name, model=model,
                         modify_first_layer_channel=False, **kwargs)


class _WReN(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1_bn_relu', ConvNormActivation(
                in_channels=1, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv2_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv3_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
            ('conv4_bn_relu', ConvNormActivation(
                in_channels=32, out_channels=32,
                kernel_size=3, stride=2, padding=0)),
        ]))
        self.pool = nn.Identity()
        self.proj = nn.Linear(512, 256)
        self.relation = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(512, 512)),
            ('relu2', nn.ReLU(True)),
            ('fc3', nn.Linear(512, 512)),
            ('relu3', nn.ReLU(True)),
            ('fc4', nn.Linear(512, 256)),
            ('relu4', nn.ReLU(True)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(256, 256)),
            ('relu2', nn.ReLU(True)),
            ('dropout', nn.Dropout(0.5)),
            ('fc3', nn.Linear(256, 10)),
        ]))

    def group_panel_embeddings_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        # (N, 16, 256)
        context_embeddings = embeddings[:, :8, :]  # (N, 8, 256)
        context_embeddings_2 = context_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)  # (N, 8, 8, 256)
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1)    # (N, 8, 8, 256)
        context_embeddings_pairs = torch.cat((context_embeddings, context_embeddings_2), dim=3
                                             ).flatten(1, 2).unsqueeze(1).expand(-1, 8, -1, -1)  # (N, 8, 64, 512)

        choice_embeddings = embeddings[:, 8:, :]   # (N, 8, 256)
        choice_embeddings = choice_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)      # (N, 8, 8, 256)
        choice_context_order = torch.cat((context_embeddings, choice_embeddings), dim=3)      # (N, 8, 8, 512)
        choice_context_reverse = torch.cat((choice_embeddings, context_embeddings), dim=3)    # (N, 8, 8, 512)
        embedding_pairs = [context_embeddings_pairs, choice_context_order, choice_context_reverse]
        return torch.cat(embedding_pairs, dim=2)  # (N, 8, 80, 512)

    def get_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 80, 80)
        return super().get_fm(x.flatten(0, 1).unsqueeze(1), **kwargs)  # (1600, 32, 4, 4)

    def get_final_fm(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 80, 80)
        batch_size = x.size(0)
        out: torch.Tensor = self.get_fm(x, **kwargs)    # (1600, 32, 4, 4)
        out = self.flatten(out)  # (1600, 512)
        out = self.proj(out)     # (1600, 256)
        out = out.view(batch_size, 16, 256)  # (N, 16, 256)
        out = self.group_panel_embeddings_batch(out)  # (N, 8, 80, 512)
        out = out.flatten(0, -2)  # (N * 8 * 80, 512)
        out = self.relation(out)  # (N * 8 * 80, 256)
        out = out.view(batch_size * 8, 80, 256).sum(dim=1)  # (N * 8, 256)
        return out

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 80, 80)
        x = self.get_final_fm(x, **kwargs)  # (N * 8, 256)
        x = self.classifier(x)  # (N * 8, 10)
        return x[:, -1].view(-1, 8)  # (N, 8)

    def get_full_logits(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # (N, 16, 80, 80)
        x = self.get_final_fm(x, **kwargs)  # (N * 8, 256)
        x = self.classifier(x)  # (N * 8, 10)
        x = x.view(-1, 8, 10)  # (N, 8, 10)
        pred = x[:, :, -1]
        meta_pred = x[:, :, :-1].sum(dim=1)  # (N, 9)
        return pred, meta_pred


class WReN(ImageModel):
    available_models = ['wren']

    def __init__(self, name: str = 'wren', model=_WReN, meta_beta: float = 10.0, **kwargs):
        super().__init__(name=name, model=model,
                         modify_first_layer_channel=False, **kwargs)
        self._model: _WReN
        self.meta_beta = meta_beta

    def get_full_logits(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self._model.get_full_logits(x, **kwargs)

    def loss(self, _input: dict[str, torch.Tensor] = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, reduction: str = 'mean', **kwargs) -> torch.Tensor:
        _output, meta_output = self.get_full_logits(_input['image'], **kwargs)
        meta_target_loss = torch.zeros(1, device=_output.device)
        for idx in range(meta_output.size(1)):
            meta_target_loss += F.binary_cross_entropy(meta_output[:, idx].sigmoid(), _input['meta_target'][:, idx])
        return self.criterion(_output, _label)+self.meta_beta*meta_target_loss

    def __call__(self, _input: dict[str, torch.Tensor], amp: bool = False, **kwargs) -> torch.Tensor:
        return super().__call__(_input['image'], amp=amp, **kwargs)

    @staticmethod
    def get_data(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return ({'image': data[0].to(env['device'], non_blocking=True),
                 'meta_target': data[2].to(env['device'])},
                data[1].to(env['device'], dtype=torch.long, non_blocking=True))


class _EfficientNet(efficientnet._EfficientNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fc: nn.Linear = self.classifier[-1]
        self.classifier = self.define_classifier(
            num_features=[fc.in_features, 512],
            num_classes=8 + 9 + 21,
            dropout=0.5)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (N, 16, 224, 224)
        x = self.get_final_fm(x, **kwargs)  # (N, 512)
        x = self.classifier(x)  # (N, 8 + 9 + 21)
        return x[:, :8]  # (N, 8)


class EfficientNet(efficientnet.EfficientNet):
    def __init__(self, model=_EfficientNet, **kwargs):
        super().__init__(model=model, **kwargs)
