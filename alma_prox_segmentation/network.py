import torch
from torch import nn

import torchvision
import torch.nn.functional as F

import resnet as models
import numpy as np
import copy

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, BatchNorm):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm(out_channels), nn.ReLU()))
        for atrous_rate in atrous_rates:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False), BatchNorm(out_channels), nn.ReLU()))
        modules.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), BatchNorm(out_channels), nn.ReLU()))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        res.append(F.interpolate(self.convs[-1](x), x.shape[2:], mode='bilinear', align_corners=True))
        return torch.cat(res, dim=1)


class DeepLabV3(nn.Module):
    def __init__(self, layers=50, atrous_rates=(6, 12, 18), dropout=0.1, classes=2, zoom_factor=8, use_aspp=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=False):
        super(DeepLabV3, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        in_channels, out_channels = 2048, 256
        if use_aspp:
            self.aspp = ASPP(in_channels, out_channels, atrous_rates, BatchNorm)
            fea_dim = out_channels * (len(atrous_rates) + 2)

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1)
        )

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(out_channels, classes, kernel_size=1)
            )

    def forward(self, x, y=None, indicate=0):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = torchvision.transforms.functional.normalize(x, self.mean, self.std)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        if self.use_aspp:
            x = self.aspp(x)
        x = self.cls(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x   

class DeepLabV3_DDCAT(nn.Module):
    def __init__(self, layers=50, atrous_rates=(6, 12, 18), dropout=0.1, classes=2, zoom_factor=8, use_aspp=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=False):
        super(DeepLabV3_DDCAT, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        self.criterion_mask = nn.CrossEntropyLoss()

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        in_channels, out_channels = 2048, 256
        if use_aspp:
            self.aspp = ASPP(in_channels, out_channels, atrous_rates, BatchNorm)
            fea_dim = out_channels * (len(atrous_rates) + 2)

        self.cls1 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1)
        )
        ################################################################
        self.mask1 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, 2, kernel_size=1)
        )
        ################################################################
        self.cls2 = nn.Sequential(
            nn.Conv2d(fea_dim, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, classes, kernel_size=1)
        )
        ################################################################

        if self.training:
            self.aux_cls1 = nn.Sequential(
                nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(out_channels, classes, kernel_size=1)
            )

    def forward(self, x, y_target=None, indicate_map=None, indicate=0):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = torchvision.transforms.functional.normalize(x, self.mean, self.std)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_tmp = self.layer3(x)
        x_tmp2 = self.layer4(x_tmp)

        if self.use_aspp:
            x = self.aspp(x_tmp2)

        result_normal = self.cls1(x)
        result_adver = self.cls2(x)
        mask1 = self.mask1(x)

        if self.zoom_factor != 1:
            result_normal = F.interpolate(result_normal, size=(h, w), mode='bilinear', align_corners=True)
            result_adver = F.interpolate(result_adver, size=(h, w), mode='bilinear', align_corners=True)
            mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=True)

        final_result = result_normal
        return final_result
        
class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None, indicate=0):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = torchvision.transforms.functional.normalize(x, self.mean, self.std)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

class PSPNet_DDCAT(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(PSPNet_DDCAT, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm
        self.num_class = classes
        self.criterion_mask = nn.CrossEntropyLoss()

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
            fea_dim *= 2

        ################################################################
        self.cls1 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        ################################################################
        self.mask1 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 2, kernel_size=1),
        )
        ################################################################
        self.cls2 = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        ################################################################
        if self.training:
            self.aux_cls1 = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y_target=None, indicate_map=None, indicate=0):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = torchvision.transforms.functional.normalize(x, self.mean, self.std)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_tmp2 = self.layer4(x_tmp)
        if self.use_ppm:
            x_ppm = self.ppm(x_tmp2)

        result_normal = self.cls1(x_ppm)
        result_adver = self.cls2(x_ppm)
        mask1 = self.mask1(x_ppm)

        if self.zoom_factor != 1:
            result_normal = F.interpolate(result_normal, size=(h, w), mode='bilinear', align_corners=True)
            result_adver = F.interpolate(result_adver, size=(h, w), mode='bilinear', align_corners=True)
            mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=True)

        return result_normal