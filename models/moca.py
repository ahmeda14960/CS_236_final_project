import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit


import numpy as np
import time
import math
import datetime
from copy import deepcopy

base_config = {
    'learnable_noise': False,
    'sigma_eps': [0.1]*4,
    'return_mean_particle': True,
    'batch_size': 10,
    'data_horizon': 40,
    'multistep_training': False,
    'learning_rate': 1e-3,
    'hazard': 0.05,
    'learnable_hazard': False,
    'hidden_dim': 128,
    'phi_dim': 128,
    'max_run_length': 200,
    'log_params': True,
    'log_nlls': True
}

@jit.script
def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

class Moca(nn.Module):
    """
    Lightweight version of MOCA Adapted for this project to compute 
    run-length beliefs for implicit skill segmentation. Only used to compute
    change points
    """

    def __init__(self, device, config={}):
        super().__init__()

        if config is {}:
            print('Warning: Moca does not inherit config of base meta learning model')
        
        self.config = deepcopy(base_config)
        self.config.update(config)

        self.device = device
        self.x_dim = self.config['x_dim']
        self.u_dim = self.config['u_dim']
        
        # hazard rate:
        hazard_logit = np.log( self.config['hazard'] / (1 - self.config['hazard'] ) )
        # don't learn hazard rate according to James
        self.hazard_logit = nn.Parameter(torch.from_numpy(np.array([hazard_logit])), requires_grad=self.config['learnable_hazard'])
        # TODO(james): when hazard is learnable, this gives a problem when calling backward
        
        
        self.mrl = self.config['max_run_length']

        # initial log_prx:
        # assuming this is log prob of run length?
        # I think this is saying that the initial probability is 1 for a run
        # length of 0, and 0 for all other run lengths
        self.init_prgx1 = nn.Parameter(torch.ones([1,1]), requires_grad=False)
        self.init_prgx2 = nn.Parameter(torch.zeros([1,self.mrl-1]), requires_grad=False)
        #self.init_log_prgx = torch.log(torch.cat((init_prgx1, init_prgx2),1)).to(self.device)

        # guessing that once again, for each possible value of the run_length
        # for run_length value 0, we init log p( task switch) with the hazard rate.
        # for all other run length values,  we init with log 1 - p(task switch)
        # because for those run lengths to be valid we had to have not switched
        hazard_term1 = self.log_hazard.view(1,1)
        hazard_term2 = self.log_1m_hazard.view(1,1).repeat(1,self.mrl-1)
        self.log_hazard_term = torch.cat((hazard_term1,hazard_term2),1).to(self.device)

    def log_p_r_given_x(self,log_prx):
        """
            computes log p(r|x)
            inputs: log_prx: (batch_size, t+1), log p(r, x) for each r in 0, ..., t
                    log_prx: (batch_size, t+1), log p(r | x) for each r in 0, ..., t
        """
        return nn.functional.log_softmax(log_prx,dim=1)

    @property 
    def init_log_prgx(self):
        return torch.log(torch.cat((self.init_prgx1, self.init_prgx2),1)).to(self.device)

    @property
    def z(self):
        return torch.zeros((1,1)).float().to(self.device)

    @property
    def log_hazard(self):
        """
        log p( task_switch )
        """
        return torch.log(torch.sigmoid(self.hazard_logit)).float()

    @property
    def log_1m_hazard(self):
        """
        log (1 - p(task_switch))
        """
        return torch.log(1-torch.sigmoid(self.hazard_logit)).float()

    @property
    def hazard(self):
        return torch.sigmoid(self.hazard_logit)

    def prior_params(self):
        # prior_params = self.meta_learning_alg.prior_params()
        
        # params = []
        # for p in prior_params:
        #     #transform p to be correctly shaped
        #     p_rs = p[None,None,:,:].repeat(1,self.mrl,1,1)
        #     #append
        #     params.append(p_rs)

        log_prgx = self.init_log_prgx # p(r, all data so far)
        
        return log_prgx
    
    def changepoint_filtering(self, log_pygx, log_prgx):
        # currently ignoring changepoint supervision
        batch_size = log_pygx.shape[0]
        # compute log p(y|x,hist) for all possible run lengths (shape: (batch_size, t+1))
        # and return updated params incorporating new point
        # log_pi_t = log p(y|x,r,hist) for all r = [0,...,i]
        # log probs for alpaca
        # seems like a normal alpaca update (great)
        #log_pygx, updated_posterior_params = self.meta_learning_alg.log_predictive_prob(phi.unsqueeze(-2), y.unsqueeze(-2), posterior_params, update_params=True)

        # compute posterior beliefs given data
        # log p(y | x) + log b(r|x) = log( p(y | x) b(r|x))
        brgy = log_pygx + log_prgx
        # renormalize
        # so this is log b(r | x, y)
        normalized_brgy = torch.log_softmax(brgy, 1)# - logsumexp(brgy, dim=1).view(-1,1) # log p(r_{i+1} | x_{0:i}, y_{0:i})
        
        # do temporal push forward; can do a matrix operation as probs but have to do nonlinear with logs
        # shouldn't we use log of 1 - hazard term somewhere?
        # tensor of zeros (batch_size, 1)
        zeros = self.z.repeat(batch_size, 1)
        # tensor of (batch_size, ...) but it ignores the last two elements. of the
        second_term = normalized_brgy[:,:-2]
        # no idea what's going on here... seems like it ignores first two terms?
        # odd indexing, then we log sum exp over run length dim, but then we add a dummy
        # dimension at the end.
        third_term = logsumexp(normalized_brgy[:,-2:], dim=1)[:,None]
        # concat all tensors on run_length dim and add log hazard term for push forward.
        pushed_fwd_pr = torch.cat((zeros, second_term, third_term), 1) + self.log_hazard_term
        
        # updating params
        # prior_params = self.meta_learning_alg.prior_params()
        # new_params = []
        # why? why not use use posterior params?
        # for p,pp in zip(updated_posterior_params, prior_params):
        #     # currently doing this as a cat; is there a more memory efficient way to do this?
        #     pn = torch.cat((pp[None,None,:,:].repeat(batch_size,1,1,1), p[:,:-1,:,:]),1)
        #     new_params.append(pn)
        
        return pushed_fwd_pr