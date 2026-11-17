"""UPerNet (Xiao et al., ECCV 2018) on a timm encoder, with a 32-channel bottleneck before the classifier
so that the cached per-image feature map is part of the trained model.

Outputs are at 1/4 of the input resolution (the native resolution of the fused FPN map): class logits
(num_classes channels) and the bottleneck features (feat_dim channels). `sliding_window` stitches full
images from overlapping crops.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PPM(nn.Module):
    """Pyramid pooling on the coarsest feature map (PSPNet)."""

    def __init__(self, cin: int, cout: int, bins: Sequence[int] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b), conv_bn_relu(cin, cout, 1)) for b in bins])
        self.fuse = conv_bn_relu(cin + len(bins) * cout, cout, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x] + [F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False) for s in self.stages]
        return self.fuse(torch.cat(outs, 1))


class UPerNet(nn.Module):
    def __init__(self, encoder_name: str = "convnext_tiny", num_classes: int = 11, channels: int = 256,
                 feat_dim: int = 32, pretrained: bool = True, encoder: Optional[nn.Module] = None,
                 encoder_channels: Optional[Sequence[int]] = None, dropout: float = 0.1):
        super().__init__()
        if encoder is None:
            import timm
            encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
            encoder_channels = encoder.feature_info.channels()
        assert encoder_channels is not None and len(encoder_channels) == 4, "need 4 feature levels (strides 4, 8, 16, 32)"
        self.encoder = encoder
        self.ppm = PPM(encoder_channels[-1], channels)
        self.lateral = nn.ModuleList([conv_bn_relu(c, channels, 1) for c in encoder_channels[:-1]])
        self.fpn = nn.ModuleList([conv_bn_relu(channels, channels, 3) for _ in encoder_channels[:-1]])
        self.fuse = conv_bn_relu(4 * channels, channels, 3)
        self.bottleneck = nn.Sequential(nn.Dropout2d(dropout), conv_bn_relu(channels, feat_dim, 1))
        self.classifier = nn.Conv2d(feat_dim, num_classes, 1)
        self.num_classes, self.feat_dim = num_classes, feat_dim

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        feats: List[torch.Tensor] = list(self.encoder(x))          # strides 4, 8, 16, 32
        laterals = [lat(f) for lat, f in zip(self.lateral, feats[:-1])] + [self.ppm(feats[-1])]
        for i in range(len(laterals) - 1, 0, -1):                     # top-down pathway
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:],
                                                              mode="bilinear", align_corners=False)
        outs = [fpn(l) for fpn, l in zip(self.fpn, laterals[:-1])] + [laterals[-1]]
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs[1:]]
        feat = self.bottleneck(self.fuse(torch.cat(outs, 1)))        # (B, feat_dim, H/4, W/4)
        logits = self.classifier(feat)
        return (logits, feat) if return_feat else logits

    def param_groups(self, lr: float, encoder_mult: float = 0.1, weight_decay: float = 0.01):
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc}
        dec = [p for p in self.parameters() if p.requires_grad and id(p) not in enc_ids]
        return [{"params": enc, "lr": lr * encoder_mult, "weight_decay": weight_decay},
                {"params": dec, "lr": lr, "weight_decay": weight_decay}]


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def window_starts(length: int, crop: int, stride: int) -> List[int]:
    """Start offsets covering [0, length) with crops of size `crop`, last one flush with the end."""
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


@torch.no_grad()
def sliding_window(model: UPerNet, image: torch.Tensor, crop: int = 1024, stride: int = 768,
                   batch: int = 4, amp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-image inference. image: (3, H, W) normalised tensor on the model's device.
    Returns logits (C, H/4, W/4) and features (feat_dim, H/4, W/4) averaged over overlapping windows."""
    model.eval()
    _, H, W = image.shape
    pad_h, pad_w = max(0, crop - H), max(0, crop - W)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h))
    _, Hp, Wp = image.shape
    s = 4
    logits = torch.zeros(model.num_classes, Hp // s, Wp // s, device=image.device, dtype=torch.float32)
    feat = torch.zeros(model.feat_dim, Hp // s, Wp // s, device=image.device, dtype=torch.float32)
    count = torch.zeros(1, Hp // s, Wp // s, device=image.device, dtype=torch.float32)
    boxes = [(y, x) for y in window_starts(Hp, crop, stride) for x in window_starts(Wp, crop, stride)]
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        x = torch.stack([image[:, y:y + crop, x0:x0 + crop] for y, x0 in chunk])
        with torch.autocast("cuda", enabled=amp and image.is_cuda):
            lg, ft = model(x, return_feat=True)
        for (y, x0), l, f in zip(chunk, lg.float(), ft.float()):
            ys, xs = y // s, x0 // s
            logits[:, ys:ys + crop // s, xs:xs + crop // s] += l
            feat[:, ys:ys + crop // s, xs:xs + crop // s] += f
            count[:, ys:ys + crop // s, xs:xs + crop // s] += 1
    logits /= count
    feat /= count
    return logits[:, :H // s, :W // s], feat[:, :H // s, :W // s]
