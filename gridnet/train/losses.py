"""Losses: weighted cross-entropy + Lovász-softmax (Berman et al., CVPR 2018) on flat point logits."""
import torch
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    """probas: (P, C) softmax probabilities, labels: (P,) with ignore already removed."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class SegLoss(torch.nn.Module):
    """ce_weight * weighted CE + lovasz_weight * Lovász-softmax, ignoring label 255."""

    def __init__(self, class_weights=None, ce_weight: float = 1.0, lovasz_weight: float = 1.0, ignore: int = 255):
        super().__init__()
        self.register_buffer("w", None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32))
        self.ce_weight, self.lovasz_weight, self.ignore = ce_weight, lovasz_weight, ignore

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        loss = self.ce_weight * F.cross_entropy(logits, target, weight=self.w, ignore_index=self.ignore)
        if self.lovasz_weight > 0:
            keep = target != self.ignore
            loss = loss + self.lovasz_weight * lovasz_softmax_flat(F.softmax(logits[keep], dim=1), target[keep])
        return loss
