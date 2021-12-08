import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


import numpy as np
import time
import math
import datetime
from copy import deepcopy

from .torch_dynamics import TorchAdaptiveDynamics
from torch import jit

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
    MOCA Wraps an underlying MetaLearning algorithm to allow training on timeseries of
    sequential examples with discrete task switches that are unlabeled. Adapted for
    this project to compute run-length beliefs for implicit skill segmentation
    """

    def __init__(self, meta_learning_alg, config={}):
        super().__init__()

        if config is {}:
            print('Warning: Moca does not inherit config of base meta learning model')
        
        self.config = deepcopy(base_config)
        self.config.update(config)


        self.x_dim = self.config['x_dim']
        self.u_dim = self.config['u_dim']

        self.meta_learning_alg = meta_learning_alg
        
        # hazard rate:
        hazard_logit = np.log( self.config['hazard'] / (1 - self.config['hazard'] ) )
        # don't learn hazard rate according to James
        self.hazard_logit = nn.Parameter(torch.from_numpy(np.array([hazard_logit])), requires_grad=False)
        # TODO(james): when hazard is learnable, this gives a problem when calling backward
        
        
        self.mrl = self.config['max_run_length']

        # initial log_prx:
        # assuming this is log prob of run length?
        # I think this is saying that the initial probability is 1 for a run
        # length of 0, and 0 for all other run lengths
        init_prgx1 = nn.Parameter(torch.ones([1,1]), requires_grad=False)
        init_prgx2 = nn.Parameter(torch.zeros([1,self.mrl-1]), requires_grad=False)
        self.init_log_prgx = torch.log(torch.cat((init_prgx1, init_prgx2),1))

        # guessing that once again, for each possible value of the run_length
        # for run_length value 0, we init log p( task switch) with the hazard rate.
        # for all other run length values,  we init with log 1 - p(task switch)
        # because for those run lengths to be valid we had to have not switched
        hazard_term1 = self.log_hazard.view(1,1)
        hazard_term2 = self.log_1m_hazard.view(1,1).repeat(1,self.mrl-1)
        self.log_hazard_term = torch.cat((hazard_term1,hazard_term2),1)

        self.encoder = self.meta_learning_alg.encoder
        
    def nll(self, log_pi, log_prgx):
        """
            log_pi: shape(batch_size x t x ...)       log p(new data | x, r=i, data so far) for all i = 0, ..., t
            log_prgx: shape (batch_size x t x ...)    log p(r=i | data so far) for all i = 0, ..., t
        """
        # jointly optimizes prob of UPM and run length model.
        if len(log_pi.shape) == 3:
            return -torch.logsumexp(log_pi + log_prgx.unsqueeze(-1), dim=1)

        return -torch.logsumexp(log_pi + log_prgx, dim=1)

    def log_p_r_given_x(self,log_prx):
        """
            computes log p(r|x)
            inputs: log_prx: (batch_size, t+1), log p(r, x) for each r in 0, ..., t
                    log_prx: (batch_size, t+1), log p(r | x) for each r in 0, ..., t
        """
        return nn.functional.log_softmax(log_prx,dim=1)

    @property
    def z(self):
        return torch.zeros((1,1)).float()

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
        prior_params = self.meta_learning_alg.prior_params()
        
        params = []
        for p in prior_params:
            #transform p to be correctly shaped
            p_rs = p[None,None,:,:].repeat(1,self.mrl,1,1)
            #append
            params.append(p_rs)

        log_prgx = self.init_log_prgx # p(r, all data so far)
        
        return params, log_prgx
    
    def changepoint_filtering(self, phi, y, posterior_params, log_prgx):
        # currently ignoring changepoint supervision
        batch_size = y.shape[0]
        # compute log p(y|x,hist) for all possible run lengths (shape: (batch_size, t+1))
        # and return updated params incorporating new point
        # log_pi_t = log p(y|x,r,hist) for all r = [0,...,i]
        # log probs for alpaca
        # seems like a normal alpaca update (great)
        log_pygx, updated_posterior_params = self.meta_learning_alg.log_predictive_prob(phi.unsqueeze(-2), y.unsqueeze(-2), posterior_params, update_params=True)

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
        prior_params = self.meta_learning_alg.prior_params()
        new_params = []
        # why? why not use use posterior params?
        for p,pp in zip(updated_posterior_params, prior_params):
            # currently doing this as a cat; is there a more memory efficient way to do this?
            pn = torch.cat((pp[None,None,:,:].repeat(batch_size,1,1,1), p[:,:-1,:,:]),1)
            new_params.append(pn)
        
        return log_pygx, pushed_fwd_pr, new_params


    def log_predictive_prob(self, phi, y, posterior_params, log_prgx, update_params=False):
        batch_size = y.shape[0]
        
        # recursively update params; compute associated terms (used for loss comp)
        log_pygx, updated_log_prgx, updated_params = self.changepoint_filtering(phi,y,posterior_params,log_prgx)

        # compute posterior predictive         
        nll = self.nll(log_pygx, updated_log_prgx)
        
        
        if not update_params:
            return logp
                        
        return nll, updated_log_prgx, updated_params
        
    
    def forward(self,x_mat,y_mat):
        """
        Takes in x,y batches; loops over horizon to recursively compute posteriors
        Inputs:
        - x_mat; shape = batch size x horizon x x_dim
        - y_mat; shape = batch size x horizon x y_dim
        """
        raise NotImplementedError
        
class MocaDynamics(TorchAdaptiveDynamics):
    def __init__(self, model, f_nom=None):
        """
        model: Moca object
        """
        super().__init__()
        self.f_nom = f_nom
        if f_nom is None:
            self.f_nom = lambda x,u: x
            
        self.model = model
        self.reset()

        self.ob_dim = self.model.x_dim
        self.u_dim = self.model.u_dim
        
        
    def reset(self):
        # TODO(james): reset belief
        # reset prior params
        self.params, self.log_belief = self.model.prior_params()
        
    # Just returns samples from alpaca
    def sample_posterior(self,Q,Linv,num_samples):
        # TODO(james)
        sig = self.model.SigEps
        #currently doing matrix normal sampling

        mu = (Linv @ Q)[None,...]

        # do sampling
        X = torch.randn((num_samples,self.phi_dim, self.y_dim))

        reg_eps = 0.00001
        sig_chol = torch.cholesky(Linv + reg_eps*torch.eye(Linv.shape[0]))[None,...]

        K = (mu + sig_chol @ X * sig)

        # TODO(james) add in return mean particle as in adaptiveDynamicsTorch
        
        return K
    


    def get_model_torch(self, model_type=TorchAdaptiveDynamics.ModelType.POSTERIOR_PREDICTIVE, **kwargs):
        """
        returns function mapping x,u to mu, sig of next_state
        """

        if model_type == self.ModelType.POSTERIOR_PREDICTIVE:
            # the posterior predictive is a mixture of Gaussians, not clear what format this should be returned in
            # not currently implementing, if needed can decide on a spec -- james
            raise NotImplementedError

        elif model_type == self.ModelType.MAP:
            # TODO(james): test this function
            
            # can't maintain differentiability of this op
            # for now, first grab most likely model, then grab mean of that model
            # this computation of MAP for a GMM is inaccurate---should optimize the pdf
            # better approx computation would be looking at product of belief and each gaussian pdf

            # get max belief element
            max_belief_idx = torch.argmax(self.logbelief)

            # output model mean corresponding to max belief element
            map_params = [p[max_belief_idx,:,:] for p in self.params]

            # with map params, compute posterior predictive
            Q, Linv = map_params
            Kbar = Linv @ Q
            sig = self.model.SigEps
            K = Kbar
            def f(x,u):
                z = torch.concat([x,u], dim=-1)
                phi = self.model.encoder(z)
                mu = ( K.transpose(-1,-2) @ phi.unsqueeze(-1) ).squeeze(-1)
                return mu, sig

        elif model_type == self.ModelType.SAMPLE:            
            # no torch choose so have to do ourselves
            # should we even be doing this in torch?
            r = torch.rand((1,1))
            
            belief = torch.exp(self.logbelief)
            for idx in range(self.model.mrl):
                if torch.sum(belief[:idx]) > r:
                    break
                    
            # idx index of sampled particle
            params = [p[idx,:,:] for p in self.params]
            
            # sample from these params
            Q, Linv = params
            
            def f(x,u,num_samples=1):
                Ks = self.sample_posterior(Q, Linv, num_samples)
                
                print(Ks.shape)
                print(phi.shape)
                
                samples = Ks.transpose(-1,-2) @ phi.unsqueeze(-1)
                
                return samples
                
        return f
        
    def incorporate_transition_torch_(self, x, u, xp):
        """
        updates self.params and self.log_belief after conditioning on transition (x,u,xp)
        """
        
        updated_params, updated_logbelief = self.incorporate_transition(self.params, self.log_belief, x, u, xp)
        
        self.params = updated_params
        self.log_belief = updated_logbelief
        
        return self.params, self.log_belief
        
    def incorporate_transition_torch(self, params, logbelief, x, u, xp):
        """
        returns posterior params after updating params with transition x, u, xp
        """
        
        # compute phi, y
        z = torch.cat([x,u], dim=-1)
        phi = self.model.encoder(z)
        y = xp - self.f_nom(x,u)
        
        _, updated_logbelief, updated_params = self.model.changepoint_filtering(phi, y, params, logbelief)
        return updated_params, updated_logbelief
    

def train(model, dataloader, num_train_updates, config, val_dataloader=None, verbose=False):
    model.reset()
    path = 'moca'
    writer = SummaryWriter('./runs/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))

    step = 0

    # set up optimizer
    optimizer = optim.Adam(model.model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=1-config['lr_decay_rate'])

    horizon = config['data_horizon']
    max_context = config['max_context']

    validation_freq = 20
    val_iters = 5

    print_freq = validation_freq if verbose else num_train_updates+1

    val_loss = 0.
    
    for idx, sample in enumerate(dataloader):
        if idx > num_train_updates:
            break
       
        x = sample['x'].float()[0,...]
        u = sample['u'].float()[0,...]
        xp = sample['xp'].float()[0,...]

        y = xp - model.f_nom(x,u)
    
        optimizer.zero_grad()
        
        # get prior statistics
        stats, log_prgx = model.model.prior_params()
    
        local_stats = stats
        local_log_prgx = log_prgx
        z = torch.cat([x,u], dim=-1)            
        phi = model.model.encoder(z)
        
        # loop over data:
        for j in range(max_context):
            phi_ = phi[:,j,:]
            y_ = y[:,j,:]

            # get posterior likelihood
            nll, local_log_prgx, local_stats = model.model.log_predictive_prob(phi_, y_, local_stats, local_log_prgx, update_params=True)
            if j == 0:
                total_nll = nll
            else:
                total_nll += nll   
        if max_context < horizon:
            raise ValueError('max_context must be greater than horizon in MOCA')
        print(total_nll)
        loss = torch.mean(total_nll/horizon)

        # backprop and optimize steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        
        writer.add_scalar('Loss/Train', loss.item(), step)
        writer.add_scalar('learning_rate', scheduler.get_lr()[0], step)
            
        if idx % print_freq == 0:
            print('Iteration: %d' % idx)
            print('Training Loss: %f' % loss.detach().numpy())
            if val_dataloader is not None: print('Validation NLL: %f' % val_loss)
            print('---------------')

        model.reset()

