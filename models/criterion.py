import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight)
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err

    def forward_noWeight_Loss(self, preds, targets):
        if isinstance(preds, list):
            N = len(preds)
            errs = [self._forward(preds[n], targets[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))
        elif isinstance(preds, torch.Tensor):
            err = self._forward(preds, targets)
        return err


class LogL1Loss(BaseLoss):
    def __init__(self):
        super(LogL1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        #return torch.mean(weight * torch.abs(pred - target))
        return torch.log(torch.abs(pred - target) + 0.5).mean()


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        #return torch.mean(weight * torch.abs(pred - target))
        return torch.mean(torch.abs(pred - target))

class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        #return torch.mean(weight * torch.pow(pred - target, 2))
        return torch.mean(torch.pow(pred - target, 2))


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)
 
class BCELoss_noWeight(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target):
        return F.binary_cross_entropy(pred, target)


class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target):
        #return nn.CrossEntropyLoss()#pred, target)
        return F.cross_entropy(pred, target)

class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, preds, targets, weight):
        return F.mse_loss(preds, targets)

