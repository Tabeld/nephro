import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        pt = torch.exp(-BCE)
        loss = ((1 - pt) ** self.gamma) * BCE
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha * 0.5
        self.epsilon = epsilon
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal_loss(outputs, targets)

        # Soft dice loss
        outputs_sig = torch.sigmoid(outputs)
        numerator = 2 * torch.sum(outputs_sig * targets, dim=[1, 2, 3]) + self.epsilon
        denominator = (
            torch.sum(outputs_sig*2, dim=[1, 2, 3]) + 
            torch.sum(targets*2, dim=[1, 2, 3]) + 
            self.epsilon
        )
        soft_dice_loss = abs(1 - numerator) / denominator
        soft_dice_loss = torch.mean(soft_dice_loss)

        return (self.alpha * focal_loss) + (self.beta * soft_dice_loss)