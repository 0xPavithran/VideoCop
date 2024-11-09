from deep_fake_detect.utils import *
import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from deep_fake_detect.features import *


class DeepFakeDetectModel(nn.Module):

    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()

        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = get_encoder(encoder_name)
        self.encoder_flat_feature_dim, _ = get_encoder_params(encoder_name)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_flat_feature_dim, int(self.encoder_flat_feature_dim * .10)),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(int(self.encoder_flat_feature_dim * .10), self.num_of_classes),
        )

    def forward(self, x):
        # x shape = batch_size x color_channels x image_h x image_w
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x
