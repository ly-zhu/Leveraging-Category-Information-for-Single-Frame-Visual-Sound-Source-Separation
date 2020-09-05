import torch
import torch.nn as nn
import torch.nn.functional as F

def unet_conv(input_nc, output_nc, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=False)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)

    if innermost:
        return nn.Sequential(*[downrelu, downconv])
    elif outermost:
        return nn.Sequential(*[downconv])
    else:
        return nn.Sequential(*[downrelu, downconv, downnorm])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
    if outermost:
        upconv = nn.Conv2d(
                input_nc, output_nc, kernel_size=3, padding=1)
        return nn.Sequential(*[uprelu, upsample, upconv])#, nn.Sigmoid()])
    else:
        upconv = nn.Conv2d(
                input_nc, output_nc, kernel_size=3,
                padding=1, bias=False)

        return nn.Sequential(*[uprelu, upsample, upconv, upnorm])


class AudioVisual7layerUNet(nn.Module):
    def __init__(self, fc_dim=64, ngf=64, input_nc=1, output_nc=1):
        super(AudioVisual7layerUNet, self).__init__()

        #initialize layers
        self.bn0 = nn.BatchNorm2d(input_nc)

        # encoder down
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, innermost=False, outermost=True)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8, innermost=True, outermost=False)

        # decoder --branch 1, output and the end: k channels without norm
        self.audionet_upconvlayer1 = unet_upconv(ngf * 8, ngf * 8, outermost=False)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, fc_dim, outermost=True) #outerost layer use a sigmoid to bound the mask

    def forward(self, x):
        x = self.bn0(x)
        # encoder
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        # decoder --branch 1
        audio_feature_7 = self.audionet_upconvlayer1(audio_conv7feature)
        audio_feature_6 = self.audionet_upconvlayer2(torch.cat((audio_feature_7, audio_conv6feature), dim=1))
        audio_feature_5 = self.audionet_upconvlayer3(torch.cat((audio_feature_6, audio_conv5feature), dim=1))
        audio_feature_4 = self.audionet_upconvlayer4(torch.cat((audio_feature_5, audio_conv4feature), dim=1))
        audio_feature_3 = self.audionet_upconvlayer5(torch.cat((audio_feature_4, audio_conv3feature), dim=1))
        audio_feature_2 = self.audionet_upconvlayer6(torch.cat((audio_feature_3, audio_conv2feature), dim=1))
        audio_feature_1 = self.audionet_upconvlayer7(torch.cat((audio_feature_2, audio_conv1feature), dim=1))
        
        return audio_feature_1
