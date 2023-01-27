
# Author: Nicolas DEVAUX
import torch
import torch.optim as optim
from torch import Tensor
from torch.optim.optimizer import Optimizer,required,_use_grad_for_differentiable
import torch.nn as nn

import networks.custom_layers as cl

from typing import List, Tuple, Dict, Optional, Callable

#region getters

def get_scheduler(des:dict,optimizer:optim.Optimizer):
    if des["name"] == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = des["T_0"],
                T_mult=des["T_mult"],
                last_epoch=des["last_epoch"])
    raise ValueError("Scheduler not found")


def get_optimizer(des:dict,model:nn.Module)->optim.Optimizer:
    if des["name"] == "SGD":
            return optim.SGD(model.parameters(),lr=des["lr"],momentum=des["momentum"],weight_decay=des["weight_decay"])
    if des["name"] == "B-lasso":
            if not isinstance(model,cl.interfaceModel):
                raise ValueError("B-lasso optimizer only works with B-lasso models")
            return B_LASSO([
                {'params': model.conv_like.parameters(), 'l1':des["l1_coeff"],"B":des["B"]},
                {'params': model.FC.parameters(), 'l1':des["l1_coeff_FC"],"B":des["B"]}
                ],lr=des["lr"],l1=des["l1_coeff"],B=des["B"],foreach=True)
    
    raise ValueError("Optimizer not found")
#endregion

#region scheduler

#endregion

#region optimizers

# Algorithm according to : https://arxiv.org/abs/2007.13657
# Implementation inspired by torch.optim.SGD, see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html 
class B_LASSO(Optimizer):
    r"""Implements B-lasso optimization algorithm.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        l1 (float): l1 regularization coefficient
        B (float): B-lasso coefficient
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
    
    """
    def __init__(self, params, lr=required, l1=0.0, B = 0.0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, l1=l1, B=B, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(B_LASSO, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(B_LASSO, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
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
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            b_lasso(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                l1 = group['l1'],
                B = group['B'],
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
    
def b_lasso(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        l1: float,
        B: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs B_lasso algorithm computation.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False
    
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_b_lasso
    else:
        func = _single_tensor_b_lasso

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         l1=l1,
         B=B,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_b_lasso(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       l1: float,
                       B: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):
    for i, param in enumerate(params):
        d_p :Tensor = d_p_list[i] if not maximize else -d_p_list[i]
        # lasso specific
        d_p = d_p.add(d_p.sign(),alpha = l1)
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
        # B_lasso specific
        param.mul_(param.abs().sub(B*l1).heaviside(param.new_tensor([0])))

def _multi_tensor_b_lasso(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      l1: float,
                      B: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):
    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    # lasso specific:
    grads = torch._foreach_add(grads, [param.sign() for param in params], alpha=l1)
    
    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
        # B_lasso specific:
        torch._foreach_mul_(params,[param.heaviside(param.new_tensor([0])) for param in torch._foreach_sub(torch._foreach_abs(params),B*l1)])
        
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)

    
#endregion
