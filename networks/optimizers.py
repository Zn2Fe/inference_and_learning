from time import sleep
import torch
import torch.optim as optim

import torch.nn as nn

import networks.custom_layers as cl

from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Tuple, Dict, Optional, Callable
#region optimizers
def get_optimizer(pd_dict:dict,model:nn.Module)->optim.Optimizer:
    p = pd_dict["optimizer_params"]
    if pd_dict["optimizer"] == "SGD":
            return optim.SGD(model.parameters(),lr=p["lr"],momentum=p["momentum"],weight_decay=p["weight_decay"])
    if pd_dict["optimizer"] == "B-lasso":
            if not isinstance(model,cl.interfaceModel):
                raise ValueError("B-lasso optimizer only works with B-lasso models")
            return B_lasso([
                {'params': model.conv_like.parameters(), 'l1_coeff':p["l1_coeff"],"BB_l1":p["BB"]*p["l1_coeff"]},
                {'params': model.FC.parameters(), 'l1_coeff':p["l1_coeff_FC"],"BB_l1":p["BB"]*p["l1_coeff_FC"]}
                ],lr=p["lr"],l1_coeff=p["l1_coeff"],BB_l1=p["BB"]*p["l1_coeff"])
    
    raise ValueError("Optimizer not found")

class B_lasso(Optimizer):
    def __init__(self, params, lr = None,l1_coeff=0.0, BB_l1 = 0):
        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, l1_coeff=l1_coeff, BB_l1=BB_l1)
        super(B_lasso, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(B_lasso, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True
            
            
            
            params : List[Tensor] = params_with_grad
            d_p_list : List[Tensor] = d_p_list
            _single_tensor_B_lasso(params,d_p_list,group['lr'],group['l1_coeff'],group['BB_l1'],has_sparse_grad)
                
        return loss

def _single_tensor_B_lasso(params: List[Tensor],grads: List[Tensor],lr:float,l1_coeff:float,BB_l1:float,has_sparse_grad:bool = False):
    if len(params) == 0:
        return
    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)
    grads = torch._foreach_add(grads,[param.sign() for param in params] , alpha = l1_coeff)
    
    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
        temp = []
        for temps in torch._foreach_sub(params,BB_l1):
            temp.append(temps.heaviside(temps.new_tensor([0])))
        torch._foreach_mul_(params,temp)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
            params[i].mul_(params[i].sub(BB_l1).sign().add(1).div(2))
    
#endregion

#region scheduler
def get_scheduler(pd_dict:dict,optimizer:optim.Optimizer):
    params = pd_dict["scheduler_params"]
    if pd_dict["scheduler"] == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = params["T_0"],
                T_mult=params["T_mult"],
                last_epoch=params["last_epoch"])
    raise ValueError("Scheduler not found")
#endregion