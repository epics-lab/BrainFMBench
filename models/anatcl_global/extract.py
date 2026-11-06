"""AnatCL (global descriptor) for BrainFMBench: ResNet-18 encoder, mean over the 5 fold models, 512 dimensions.

Input preparation follows the extraction used for the MedAGI 2026 paper and the AnatCL README:
CAT12 VBM grey matter map (mwp1), resized to 121 x 128 x 121 if needed, no further normalization.
Weights: weights_global_fold{0..4}.pth, AnatCL checkpoints from https://github.com/EIDOSLAB/AnatCL
The ResNet and SupConResNet code below is copied from anatcl/models/resnet3d.py,
Copyright (c) 2021 EIDOSlab, MIT License.
"""
import os
import glob

import numpy as np
import nibabel as nib

DESCRIPTOR = "global"
BATCH = 32
TARGET = (121, 128, 121)

# ---- anatcl/models/resnet3d.py ----
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Standard 3D-ResNet architecture with big initial 7x7x7 kernel.
    It can be turned in mode "classifier", outputting a vector of size <n_classes> or
    "encoder", outputting a latent vector of size 512 (independent of input size).
    Note: only a last FC layer is added on top of the "encoder" backbone.
    """
    def __init__(self, block, layers, in_channels=1,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, initial_kernel_size=7):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.name = "resnet"
        self.inputs = None
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        initial_stride = 2 if initial_kernel_size==7 else 1
        padding = (initial_kernel_size-initial_stride+1)//2
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=initial_kernel_size, stride=initial_stride, padding=padding, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        channels = [64, 128, 256, 512]

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        return torch.flatten(x5, 1)

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],  **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, use_head=True):
        super().__init__()
        model_fun, dim_in = model_dict[name]

        self.encoder = model_fun()
        self.dim_in = dim_in
        self.feat_dim = feat_dim
        self.use_head = use_head

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        if not self.use_head:
            return feat

        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def features(self, x):
        return self.forward(x)
# ---- end of copied code ----


def _setup():
    # Same settings as the MedAGI 2026 extraction: deterministic, single threaded, CPU
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _prepare(path):
    img = nib.load(path).get_fdata().astype(np.float32)
    if img.shape != TARGET:
        img = F.interpolate(torch.from_numpy(img[None, None]), size=TARGET,
                            mode="trilinear", align_corners=False).squeeze().numpy()
    return torch.from_numpy(img.copy()).float()[None]


def _alias_pickled_modules():
    # The AnatCL checkpoints pickle classes from the training code (models.estimators.AgeEstimator).
    # Register placeholder modules so torch.load can unpickle them, as the paper extraction did.
    import sys
    import types
    if "models.estimators" in sys.modules:
        return
    pkg = types.ModuleType("models")
    pkg.__path__ = []
    resnet = types.ModuleType("models.resnet3d")
    resnet.__dict__.update({k: v for k, v in globals().items() if isinstance(v, type)})
    est = types.ModuleType("models.estimators")
    est.AgeEstimator = type("AgeEstimator", (), {})
    pkg.resnet3d, pkg.estimators = resnet, est
    sys.modules.setdefault("models", pkg)
    sys.modules.setdefault("models.resnet3d", resnet)
    sys.modules.setdefault("models.estimators", est)


def _load(path):
    _alias_pickled_modules()
    enc = SupConResNet("resnet18", feat_dim=128, use_head=False)
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return enc.eval()


def extract(input_dir, output_csv, weights_dir):
    _setup()
    paths = sorted(glob.glob(os.path.join(input_dir, "*", "mwp1.nii*")))
    sids = [os.path.basename(os.path.dirname(p)) for p in paths]
    print(f"[anatcl_{DESCRIPTOR}] {len(paths)} subjects in {input_dir}", flush=True)
    folds = []
    with torch.no_grad():
        for k in range(5):
            enc = _load(os.path.join(weights_dir, f"weights_{DESCRIPTOR}_fold{k}.pth"))
            feats = [enc(torch.stack([_prepare(p) for p in paths[i:i + BATCH]])).numpy()
                     for i in range(0, len(paths), BATCH)]
            folds.append(np.vstack(feats))
            print(f"[anatcl_{DESCRIPTOR}] fold {k} done", flush=True)
    feats = np.mean(folds, axis=0)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("subject_id," + ",".join(f"f{i}" for i in range(feats.shape[1])) + "\n")
        for sid, row in zip(sids, feats):
            f.write(sid + "," + ",".join(str(np.float32(v)) for v in row) + "\n")
    print(f"[anatcl_{DESCRIPTOR}] wrote {feats.shape[0]} x {feats.shape[1]} -> {output_csv}", flush=True)
