import torch.nn as nn
from monai.losses import DiceCELoss


class LossFlood(nn.Module):
    def __init__(self):
        super().__init__()        
        self.diceCE = DiceCELoss(include_background=False,
                                 sigmoid=True,
                                 reduction='mean',
                                 lambda_dice=0.85,
                                 lambda_ce=0.15,
                                 batch=True)
        
    def _loss(self, p, y):
        return self.diceCE(p, y)                    
    
    def forward(self, p, y):
        return self._loss(p, y)