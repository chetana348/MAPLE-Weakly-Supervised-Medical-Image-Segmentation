import torch
import torch.nn.functional as F
import math

def partial_ce_loss(logits, labels, confidence_mask):
    """
    logits: (B, 1, H, W)
    labels: (B, H, W) in {0,1}
    confidence_mask: (B, H, W) in {0,1}
    """
    logits = logits.squeeze(1)

    ce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )

    masked_ce = ce * confidence_mask
    return masked_ce.sum() / (confidence_mask.sum() + 1e-6)

def geodesic_consistency_loss(prob, geodesic_map, lam=1.0):
    """
    prob: (B, 1, H, W) sigmoid output
    geodesic_map: (B, H, W) normalized [0,1]
    """
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])

    gx = geodesic_map[:, :, 1:] + geodesic_map[:, :, :-1]
    gy = geodesic_map[:, 1:, :] + geodesic_map[:, :-1, :]

    loss = (dx * gx.unsqueeze(1)).mean() + (dy * gy.unsqueeze(1)).mean()
    return lam * loss


def mc_dropout_uncertainty(model, x, T=6):
    model.train()  # important
    preds = []

    with torch.no_grad():
        for _ in range(T):
            preds.append(torch.sigmoid(model(x)))

    preds = torch.stack(preds, dim=0)
    mean = preds.mean(0)
    var = preds.var(0)

    return mean, var


def build_confidence_mask(pseudo, uncertainty, tau_p=0.9, tau_u=0.02):
    """
    pseudo: (B, H, W)
    uncertainty: (B, 1, H, W)
    """
    confident_fg = (pseudo == 1)
    confident_bg = (pseudo == 0)

    high_conf = uncertainty.squeeze(1) < tau_u

    mask = (confident_fg | confident_bg) & high_conf
    return mask.float()


def dice_coefficient(pred, target, eps=1e-6):
    """
    pred: (B, 1, H, W) sigmoid output
    target: (B, H, W) binary
    """
    pred = (pred > 0.5).float()
    target = target.unsqueeze(1).float()  # (B, 1, H, W)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()
    
def consistency_loss(student_logits, teacher_logits):
    student_prob = torch.sigmoid(student_logits)
    teacher_prob = torch.sigmoid(teacher_logits).detach()
    return torch.mean((student_prob - teacher_prob) ** 2)


def sdf_loss(pred, target, band=5):
    boundary = (target.abs() < band).float()
    return torch.mean(boundary * torch.abs(pred - target))
    

def heatmap_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def boundary_weighted_ce(logits, target, band=5):
    logits = logits.squeeze(1)
    prob = torch.sigmoid(logits)
    target = target.float()

    # approximate boundary
    from torch.nn.functional import max_pool2d
    dil = max_pool2d(target, 3, stride=1, padding=1)
    ero = -max_pool2d(-target, 3, stride=1, padding=1)
    boundary = (dil - ero).abs()

    weight = 1.0 + band * boundary

    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    return (ce * weight).mean()

def compactness_loss(prob):
    area = prob.sum(dim=(2,3)) + 1e-6
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    perimeter = dx.sum(dim=(2,3)) + dy.sum(dim=(2,3)) + 1e-6
    return torch.mean((perimeter ** 2) / area)

def hole_penalty(prob):
    filled = torch.nn.functional.max_pool2d(
        prob, kernel_size=15, stride=1, padding=7
    )
    return torch.mean((filled - prob).clamp(min=0))
    
    
def consistency_loss(student_prob, teacher_prob, uncertainty):
    weight = torch.exp(-uncertainty)
    return ((student_prob - teacher_prob) ** 2 * weight).mean()


def slice_consistency_loss(prob):
    loss = 0.0
    for offset in [-1, 1]:
        shifted = torch.roll(prob, shifts=offset, dims=2)
        loss += F.mse_loss(prob, shifted)
    return loss / 2


def shape_prior_loss(prob):
    area = prob.mean()
    area_loss = torch.abs(area - 0.02)

    tv = (
        torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean() +
        torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    )

    return area_loss + tv
