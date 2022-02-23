import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import CONFIG,Optim
from metrics import get_metrics_collections_base


class LitSystem(pl.LightningModule):
    def __init__(self,
                 
                  lr,
                  optim:str="SGD",
                  is_regresor:bool=True
                  ):
        
        super().__init__()

        self.train_metrics_base=get_metrics_collections_base(prefix="train",is_regressor=is_regresor)
        self.valid_metrics_base=get_metrics_collections_base(prefix="valid",is_regressor=is_regresor)
        self.test_metrics_base=get_metrics_collections_base(prefix="test",is_regressor=is_regresor)
        # log hyperparameters
        self.save_hyperparameters()    
        self.lr=lr
        if isinstance(optim,str):
            self.optim=Optim[optim.lower()]
           
    
    def on_epoch_start(self):
        # torch.cuda.empty_cache()
        pass
    
    def configure_optimizers(self):
        if self.optim==Optim.adam:
            optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim==Optim.sgd:
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(15,100,10),gamma=0.95)
        scheduler=WarmupCosineSchedule(optimizer,warmup_steps=int(50*0.1),t_total=50)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                  max_lr=self.lr, steps_per_epoch=self.steps_per_epoch,
        #                                 epochs=self.epochs, pct_start=0.2, cycle_momentum=False, div_factor=20)
        return [optimizer], [scheduler]

    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix:str):
 
        on_step=False
        on_epoch=True 
        
        for metric,value in data_dict.items():
            if metric != "preds":
                
                self.log("_".join([prefix,metric]),value,
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        logger=True
                )
                
    def add_prefix_into_dict_only_loss(self,data_dict:dict,prefix:str=""):
        data_dict_aux={}
        for k,v in data_dict.items():            
            data_dict_aux["_".join([prefix,k])]=v
            
        return data_dict_aux
    
import math

from torch.optim.lr_scheduler import LambdaLR
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))