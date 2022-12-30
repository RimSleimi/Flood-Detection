import torch
from torchmetrics import Metric

class MetricsFlood(Metric):
    def __init__(self, n_class, full_state_update=False):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("IoU", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("precision", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("recall", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("omission_error", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("comission_error", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")

                
    def update(self, p, y, loss):
        self.steps += 1
        self.dice += self.compute_dice(p, y)
        self.IoU += self.compute_IoU(p, y)
        self.precision += self.compute_precision(p, y)
        self.recall += self.compute_recall(p, y)
        self.omission_error = self.compute_omission(p, y)
        self.comission_error = self.compute_comission(p, y)
        self.loss += loss
                
        
    def compute(self):  
        mean_dice = 100 * self.dice / self.steps
        mean_IoU = 100 * self.IoU / self.steps
        mean_omission = 100 * self.omission_error / self.steps
        mean_precision = 100 * self.precision / self.steps
        mean_recall = 100 * self.recall / self.steps
        mean_comission = 100 * self.comission_error / self.steps
        mean_loss = self.loss / self.steps
        return mean_dice, mean_IoU, mean_precision, mean_recall, mean_omission, mean_comission, mean_loss
    
    def compute_dice(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()  

        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (2 * tp + fn + fp).to(torch.float)
            score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores
    
    
    def compute_IoU(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()  
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue      
                
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (tp + fn + fp).to(torch.float)
            score_cls = tp.to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores
    
    def compute_precision(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()  
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue      
                
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (tp + fp).to(torch.float)
            score_cls = tp.to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores
    
    def compute_recall(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()  
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue      
                
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (tp + fn).to(torch.float)
            score_cls = tp.to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores
    
    def compute_omission(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()        
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue
            tp, fn, _ = self.get_stats(p_i, y_i, 1)
            denom = (fn + tp).to(torch.float)
            score_cls = fn.to(torch.float)/ denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores

    def compute_comission(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()        
        
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue
            tp, _, fp = self.get_stats(p_i, y_i, 1)
            denom = (fp + tp).to(torch.float)
            score_cls = fp.to(torch.float)/ denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores

    @staticmethod
    def get_stats(p, y, class_idx):
        tp = torch.logical_and(p == class_idx, y == class_idx).sum()
        fn = torch.logical_and(p != class_idx, y == class_idx).sum()
        fp = torch.logical_and(p == class_idx, y != class_idx).sum()
        return tp, fn, fp