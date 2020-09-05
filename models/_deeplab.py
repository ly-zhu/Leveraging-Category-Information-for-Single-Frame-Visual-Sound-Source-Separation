import torch
from torch import nn
from torch.nn import functional as F
from .backbone import mobilenetv2
from collections import OrderedDict


class _DeepLabV3Plus_MV2_Model(nn.Module):
    def __init__(self, backbone, classifier):
        super(_DeepLabV3Plus_MV2_Model, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        #x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class _Encoder_Model(nn.Module):
    def __init__(self, backbone, classifier):
        super(_Encoder_Model, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features, pool=pool)
        
        if pool:
            _, C = out.size()[0:2]
            out = out.view(B, C)
        else:
            (_, C, H, W) = out.size()
            out = out.view(B, T, C, H, W)
            out = out.permute(0, 2, 1, 3, 4)

        #x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return out

    def forward_multiframe_feat_emb(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B*T, C, H, W)

        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features, pool=False)

        (_, C, H, W) = out.size()
        out = out.view(B, T, C, H, W)
        out = out.permute(0, 2, 1, 3, 4)

        if not pool:
            return out
            #_, C = out.size()[0:2]
            #out = out.view(B, C)
        else:
            #if self.pool_type == 'avgpool':
            #    output_feature = F.adaptive_avg_pool2d(output_feature, 1)
            #elif self.pool_type == 'maxpool':
            #    output_feature = F.adaptive_max_pool2d(output_feature, 1)
            
            output_pool = F.adaptive_max_pool3d(out, 1)
            _, C = output_pool.size()[0:2]
            output_pool = output_pool.view(B, C)
        #x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return out, output_pool


def _deeplabV3Plus_mobilenet(input_channel, name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(input_channel0=input_channel, pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3(backbone, classifier)
    if name=='encoder':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = Embed_ASPP(inplanes, low_level_planes, num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = Encoder(backbone, classifier)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
        model = DeepLabV3(backbone, classifier)
    #backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #model = DeepLabV3(backbone, classifier)
    return model


class DeepLabV3(_DeepLabV3Plus_MV2_Model):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class Encoder(_Encoder_Model):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        '''
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        '''
        self.decoder1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, num_classes, 1)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )


        #self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        #print('low_level_feature: ', low_level_feature.shape)
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = self.decoder1( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        output_feature = F.interpolate(output_feature, size=[low_level_feature.shape[2]*4, low_level_feature.shape[3]*4], mode='bilinear', align_corners=False)
        output_feature = self.decoder2(output_feature)
        #return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        return output_feature
    

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Embed_ASPP(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], pool_type='maxpool'):
        super(Embed_ASPP, self).__init__()
        self.pool_type = pool_type

        '''
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        '''

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, num_classes, 3, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, num_classes, 1)
        )
        
        self._init_weight()

    def forward(self, feature, pool=True):
        #low_level_feature = self.project( feature['low_level'] )
        #print('low_level_feature: ', low_level_feature.shape)
        output_feature = self.aspp(feature['out'])
        #print('feature aspp shape: ', output_feature.shape)
        output_feature = self.classifier(output_feature)
        #print('feature classifier shape: ', output_feature.shape)
        if not pool:
            return output_feature
        
        # for evaluation (sound source separation)
        if self.pool_type == 'avgpool':
            output_feature = F.adaptive_avg_pool2d(output_feature, 1)
        elif self.pool_type == 'maxpool':
            output_feature = F.adaptive_max_pool2d(output_feature, 1)
        #print('after pooling: ', output_feature.shape)
        #output_feature = output_feature.view(B, C)
        
        #return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
        return output_feature


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """


    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
