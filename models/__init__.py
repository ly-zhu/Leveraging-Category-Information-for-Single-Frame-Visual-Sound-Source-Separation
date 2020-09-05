import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from ._deeplab import _deeplabV3Plus_mobilenet
from .audio_net import AudioVisual7layerUNet 
from .vision_net import ResnetDilated
from .criterion import BCELoss, LogL1Loss, L1Loss, L2Loss, CELoss, BCELoss_noWeight, MSELoss
from .avol import InnerProd_AVOL
import numpy as np
import torch.nn.init as init
import math

def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.001)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))

    def weights_init_1(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0, 1/np.sqrt(y))

    # builder for Sound
    def build_sound(self, arch='unet7', input_channel=1, output_channel=1, fc_dim=64, weights=''):
        # 2D models
        if arch == 'unet7':
            net_sound = AudioVisual7layerUNet(fc_dim = fc_dim, ngf = 64, input_nc=input_channel)
        elif arch == 'deeplabV3Plus_mobilenetv2':
            net_sound = _deeplabV3Plus_mobilenet(input_channel=input_channel, name='deeplabv3plus', backbone_name='mobilenetv2', num_classes=fc_dim, output_stride=16, pretrained_backbone=False)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound


    # builder for Appearance
    def build_frame(self, arch='resnet18dilated', fc_dim=64, pool_type='avgpool', feature_type='fc', os_channel=512,
                    weights=''):
        pretrained=True
        if arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type, feature_type=feature_type)
        elif arch == 'resnet18dilated_50':
            original_resnet = torchvision.models.resnet50(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type, os_channel=2048)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net


    # builder for attention
    def build_avol(self, arch='AVOL',  fc_dim=64, weights=''):
        if arch == 'AVOL':
            net = InnerProd_AVOL()
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init_1)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights))
        return net


    # builder for criterion
    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        elif arch == 'ce':
            net = CELoss()
        elif arch == 'bce_noWeight':
            net = BCELoss_noWeight()
        elif arch == 'logl1':
            net = LogL1Loss()
        elif arch == 'mse':
            net = MSELoss()
        else:
            raise Exception('Architecture undefined!')
        return net
