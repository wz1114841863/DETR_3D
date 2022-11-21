import math
import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module
class RLELoss(nn.Module):
    ''' 
    RLE Regression Loss
    '''
    def __init__(self, loss_weight=1, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.loss_weight = loss_weight

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, nf_loss, preds, sigmas, targets, weights, avg_factor=1):
        nf_loss = nf_loss * weights[:, :, :1]
        residual = True
        if residual:
            Q_logprob = self.logQ(targets, preds, sigmas) * weights
            loss = nf_loss + Q_logprob

        if self.size_average and weights.sum() > 0:
            return loss.sum() / len(loss) * self.loss_weight
        else:
            return loss.sum()
        
@LOSSES.register_module
class RLEDepthLoss(nn.Module):
    ''' 
    RLE Regression Depth Loss
    '''
    def __init__(self, loss_weight=1, size_average=True):
        super(RLEDepthLoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)
        self.loss_weight = loss_weight

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, nf_loss, preds, sigmas, targets, weights, avg_factor=1):
        nf_loss = nf_loss * weights
        residual = True
        if residual:
            Q_logprob = self.logQ(targets, preds, sigmas) * weights
            loss = nf_loss + Q_logprob

        if self.size_average and weights.sum() > 0:
            return loss.sum() / len(loss) * self.loss_weight
        else:
            return loss.sum()