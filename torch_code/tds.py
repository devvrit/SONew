import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

def ldl2tridiag(Lsub,D):
    n = D.shape[0]
    Xd = torch.zeros_like(D)
    Xd[1:] = D[1:]+Lsub*Lsub*D[:n-1]
    Xd[0] = D[0]
    Xe = Lsub*D[:n-1]
    return Xd, Xe

def tridiag(Sd, Se):
    n = Sd.shape[0]
    psi = Se/Sd[1:]
    condCov = torch.zeros_like(Sd)
    condCov[:n-1] = Sd[:n-1]-psi*Se
    condCov[n-1] = Sd[n-1]
    # Edge Removal:
    psi[condCov[:-1]<=0] = 0.0
    condCov[:n-1] = Sd[:n-1]-psi*Se
    D = 1/condCov
    Lsub = -psi
    Xd, Xe = ldl2tridiag(Lsub, D)
    return Xd, Xe

def tridiagMult(Xd,Xe,vecv):
    #find tridiag(Xd,Xe).vecv
    a = Xd*vecv
    a[1:] = a[1:]+Xe*vecv[:-1]
    a[:-1] = a[:-1]+Xe*vecv[1:]
    return a

class TDS(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        super(TDS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            sub_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros(p.numel(), device=p.device)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros(p.numel(), device = p.device)
                        # Exponential moving average of subdiagonal of gg^T
                        state['sub_exp_avg_sq'] = torch.zeros(p.numel(), device=p.device)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    sub_exp_avg_sqs.append(state['sub_exp_avg_sq'])
                    state_steps.append(state['step'])
            
            _single_tensor_adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 sub_exp_avg_sqs,
                 state_steps,
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'])

        return loss


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        sub_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float):
    # gradList = []
    for i, param in enumerate(params):

        grad = grads[i] 
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        sub_exp_avg_sq = sub_exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad.view(-1), alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad.view(-1), grad.view(-1).conj(), value=1 - beta2)
        sub_exp_avg_sq[:-1].mul_(beta2).addcmul_(grad.view(-1)[:-1], grad.conj().view(-1)[1:], value=1 - beta2)
        
        #Uncommment the below if it isn't commented
        # sub_exp_avg_sq = sub_exp_avg_sq.view(param.shape)
        # sub_exp_avg_sq = sub_exp_avg_sq.view(-1)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        denomDiag = (exp_avg_sq/ bias_correction2).add_(eps).view(-1)
        denomsubDiag = (sub_exp_avg_sq/ bias_correction2).view(-1)

        inverse = tridiag(denomDiag, denomsubDiag[:-1])
        invdenomDiag, invdenomsubDiag = inverse
        momt = exp_avg.view(-1)/bias_correction1
        update = tridiagMult(invdenomDiag,invdenomsubDiag,momt)
        # Adam graft
        adam_update = momt/(torch.sqrt(exp_avg_sq)+1e-8)
        update = update*(torch.linalg.norm(adam_update)/torch.linalg.norm(update))
        update = update.view(param.shape)
        # Weight Decay
        if weight_decay != 0:
            update = update.add(param, alpha=weight_decay)
        param.add_(update, alpha=-lr)