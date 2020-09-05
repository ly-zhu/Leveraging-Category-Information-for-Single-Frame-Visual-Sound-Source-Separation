import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerProd_AVOL(nn.Module):
    def __init__(self):
        super(InnerProd_AVOL, self).__init__()

        self.cnn1 = nn.Conv2d(1, 1, 1, stride=1)

    def forward(self, feats_img, feat_sound):
        (B, C, T, HI, WI) = feats_img.size()
        feats_img = feats_img.permute(0, 2, 1, 3, 4).contiguous()
        feats_img = feats_img.view(B*T, C, HI, WI)

        (B, C) = feat_sound.size()
        feat_sound = feat_sound.view(B, 1, C)
        feats_img = feats_img.view(B, C, HI*WI)
        z = torch.bmm(feat_sound, feats_img) \
            .view(B, 1, HI, WI)
        z = self.cnn1(z)
        return z
