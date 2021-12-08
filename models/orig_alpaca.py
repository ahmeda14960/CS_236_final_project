import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn.functional as F

from copy import deepcopy

def get_encoder(config):
    activation = nn.Tanh()
    hid_dim = config['model.hidden_dim']
    phi_dim = config['model.phi_dim']
    x_dim = config['model.x_dim']
    
    if config['env'] == 'kitchen':
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, phi_dim)
        )
    else:
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            activation,
            nn.Linear(hid_dim, hid_dim),
            activation,
            nn.Linear(hid_dim, hid_dim),
            activation,
            nn.Linear(hid_dim, phi_dim)
        )
    return encoder

base_config = {
    'env': 'lqr',
    # model parameters
    'model.x_dim': None,
    'model.u_dim': None,
    'model.z_dim': None,
    'model.hidden_dim': 128,
    'model.phi_dim': 128,
    'model.sigma_eps': [0.1]*4,
    'model.use_input_feats': False,
    
    # training parameters
    'training.learnable_noise': True,
    'training.data_horizon': 100,
    'training.learning_rate': 1e-3,
    'training.tqdm': True,
    'training.learning_rate_decay': False,
    'training.lr_decay_rate': 1e-3,
    'training.grad_clip_value':1,
}


class AlpacaImitation(nn.Module):
    """
    Not set up to run as an independent model within camelid
    For an independent standalone model, use adaptiveDynamicsTorch
    """
    def __init__(self, config={}, cuda=0, model_path=None):
        super().__init__()

        self.cuda = cuda
        self.config = deepcopy(base_config)
        if model_path is not None:
            data = torch.load(model_path)
            config = data["config"]        
            
        self.config.update(config)

        self.x_dim = self.config['model.x_dim']
        self.u_dim = self.config['model.u_dim']

        self.phi_dim = 5 if self.config['model.use_input_feats'] else self.config['model.phi_dim'] 

        self.sigma_eps = self.config['model.sigma_eps']*self.u_dim
        self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)).float(), requires_grad=self.config['training.learnable_noise'])

        self.Q = nn.Parameter(torch.randn(self.u_dim, 1, self.phi_dim)*4/(np.sqrt(self.phi_dim)+ np.sqrt(self.u_dim)))
        self.L_asym = nn.Parameter(torch.randn(self.u_dim, self.phi_dim, self.phi_dim)/self.phi_dim**2)
        self.L_base = nn.Parameter(torch.linspace(-5,0, self.phi_dim).repeat(self.u_dim,1))

        # u_dim here is dimensionality k mentioned in gaussian pdf
        self.normal_nll_const = self.u_dim*np.log(2*np.pi)
    
        # ensure NN is on GPU
       

        self.backbone = get_encoder(self.config).to(torch.device('cuda:{}'.format(self.cuda)))
        if model_path is not None:
            print("loading state dict")
            self.load_state_dict(data['state_dict'])
        
    @property
    def L(self):
        return self.L_asym @ self.L_asym.transpose(-2,-1) + torch.diag_embed( torch.exp( self.L_base ) )

    @property
    def logdetSigEps(self):
        return torch.sum(self.logSigEps)

    @property
    def invSigEps(self):
        return torch.diag(torch.exp(-self.logSigEps))

    @property
    def invSigEpsVec(self):
        return torch.exp(-self.logSigEps)

    @property
    def SigEpsVec(self):
        return torch.exp(self.logSigEps)

    @property
    def SigEps(self):
        return torch.diag(torch.exp(self.logSigEps)).to(torch.device('cuda:{}'.format(self.cuda)))

    def encoder(self, x):
        # shapes last output of phi to be 1 (to batch across ydim) x phidim x 1
        if self.config['model.use_input_feats']:
            ones = torch.ones_like(x)[...,0:1]
            phi = torch.cat((x,ones),dim=-1) 
        else:
            phi = self.backbone(x)

        return phi[...,None].unsqueeze(-3)        
    
    def prior_params(self):
        return (self.Q.unsqueeze(-4).repeat(self.config['model.z_dim'],1,1,1),
                self.L.unsqueeze(-4).repeat(self.config['model.z_dim'],1,1,1))

    def recursive_update(self, phi, u, z, params):
        """
            inputs: phi: shape (..., 1, phi_dim, 1)
                    u:   shape (..., u_dim )
                    z:   shape (..., z_dim)   (one-hot)
                    params: tuple of Q, L
                        Q: shape (..., z_dim, u_dim, 1, phi_dim)
                        L: shape (..., z_dim, u_dim, phi_dim, phi_dim)
        """
        Q, L = params

        # Indexing with None adds a new dimension
        z = z[..., None, None, None] # new z tensor is (batch_size, 1, 1, 1)
        z = z.to(torch.device('cuda:{}'.format(self.cuda)))
        u = u.to(torch.device('cuda:{}'.format(self.cuda)))
        # zeros out entries in z_dim axis that don't belong to current skill
        # if z = [1, 0, 0, 0] then L_onehot[1:3] is all zeros
        L_onehot = z * L  # (..., z_dim, u_dim, phi_dim, phi_dim)
        u_hat = u.unsqueeze(-1).unsqueeze(-1).unsqueeze(-4)  # expand dims to (..., 1, u_dim, 1, 1)
        # Once again zero out all action vectors that don't correspond to current skill
        # u_onehot is effectively four two dimensional vectors for action input
        u_onehot = z * u_hat  # (..., z_dim, u_dim, 1, 1)
        # all entries not on current skill are zero
        # effective size is (..., u_dim, phi_dim)
        Lphi = L_onehot @ phi # (..., z_dim, u_dim, phi_dim, 1)
        phi_T = torch.transpose(phi,-1,-2) # (..., 1, 1, phi_dim)
        # effective size is (..., u_dim)
        phi_L_phi = phi_T @ Lphi # (..., z_dim, u_dim, 1, 1)
        # all entries not on current skill are zero
        # effective size is (..., u_dim, phi_dim, phi_dim)
        Lphi_t_Lphi = Lphi @ Lphi.transpose(-1,-2) # (..., z_dim, u_dim, phi_dim, phi_dim)
        L_update = 1./(1 + phi_L_phi) * Lphi_t_Lphi
        L = L - L_update # (..., z_dim, u_dim, phi_dim, phi_dim)
        # TODO(Ahmed) this should be (..., z_dim, u_dim, phi_dim, 1) but is (..., z_dim, u_dim, 1, phi_dim)
        Q_update = u_onehot @ phi_T
        Q = Q + Q_update

        return (Q, L)

    def log_predictive_prob(self, phi, u, z, posterior_params, to_cuda, eval_pcoc=False, update_params=False):
        """
            input:  phi: shape (..., 1, phi_dim)
                    u: shape (..., u_dim)     (note: y ~= K.T phi, not xp)
                    z: shape (..., z_dim)
                    posterior_params: tuple of Q, L:
                        Q: shape (..., u_dim, 1, phi_dim)
                        L: shape (..., u_dim, phi_dim, phi_dim)
                    update_params: bool, whether to perform recursive update on
                                   posterior params and return updated params
                    eval_pcoc: bool, indicates evaluating pcoc for EM, and therefore
                    not weighing by the ground truth skills
            output: logp: log p(y | x, posterior_parms)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """

        # compute full predictions for each skill dim; weight predictions by skill prob
        Q, L = posterior_params
        K = Q @ L #(..., z_dim, u_dim, 1, phi_dim) # K is (y_dim, phi_dim)
        sigfactor = 1 + (torch.transpose(phi,-1,-2) @ L @ phi).squeeze(-1).squeeze(-1) # (..., z_dim,  u_dim)
        
        err = u.unsqueeze(-2)  - (K @ phi).squeeze(-1).squeeze(-1) # (..., z_dim,  u_dim)
        invsigVec = to_cuda(self.invSigEpsVec)#.unsqueeze(0).unsqueeze(-1)
        # print('invissvec', invsigVec.size())
        # print('sigfactor', sigfactor.size())
        invsig = invsigVec / sigfactor # shape (..., z_dim, u_dim)
        nll_quadform = err**2 * invsig
        nll_logdet = - torch.log(invsig)
        logp = -0.5*(self.normal_nll_const + nll_quadform + nll_logdet).squeeze(-1).squeeze(-1)

        # averaging step --- re-evaluate when moving to soft labels #TODO(ahmed)
        prod = logp if eval_pcoc else logp * z.unsqueeze(-1)
        if update_params:
            updated_params = self.recursive_update(phi,u,z,posterior_params)
            return prod, updated_params

        
        return torch.sum(prod, dim=-4)


    def forward(self, x, posterior_params):
        """
            input: x
            output: mu, and sig tensors
        """
                         
        phi = self.encoder(x)        
        Q, L = posterior_params
        K = Q @ L # (..., z_dim, u_dim, 1, phi_dim)
        # TODO(james): fix batching here
        # equation (8) in Alpaca, phi_L_phi from recursive update
        sigfactor = 1 + ((torch.transpose(phi,-1,-2) @ L @ phi)).squeeze(-1).squeeze(-1) # (..., z_dim, u_dim)
        mu = ( K @ phi).squeeze(-1) # (..., z_dim, u_dim, 1)
        sig = self.SigEps * sigfactor.unsqueeze(-1)
        return mu, sig

class AlpacaImitationStateful():
    """
    Wrapper class that maps torch alpaca with training functions and
    online prediction and adaptation functions.
    """
    def __init__(self, model, cuda=-1, exp_name=None):
        """
        Inputs:
        model: alpacaTorch object
        f_nom: function mapping tensors x -> to tensor y
        
        Sets up SummaryWriter to log to for Tensorboard visualization.
        """
        super().__init__()
        self.model = model
        self.reset()

        # used for annealing during training
        self.train_step = 0
        
        if exp_name:
            path = 'X{}X_alpacaimitation'.format(exp_name)
        else:
            path = 'alpacaimitation'
        self.writer = SummaryWriter('/iris/u/ahmedah/runs_alp/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))

        # set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.model.config['training.learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=1 - self.model.config['training.lr_decay_rate'] )

        self.cuda = cuda
        
    def reset(self):
        self.params = self.model.prior_params()
        
    def to_cuda(self,x):
        if self.cuda<0:
            return x.cpu()
        else:
            return x.cuda(self.cuda)
    
    def incorporate_transition_torch_(self, x, u, z):
        """
        updates self.params after conditioning on transition (x,u,xp)
        """
        self.params = self.incorporate_transition_torch(self.params, x, u, z)
        return self.params

    def incorporate_transition_torch(self, params, x, u, z):
        """
        returns posterior params after updating params with transition x, u, xp
        """
        phi = self.model.encoder(self.to_cuda(x))
        Q, L = params
        params = (self.to_cuda(Q), self.to_cuda(L))
        # Make one-hot tensor of shape (..., z_dim) with num_classes == z_dim
        z = F.one_hot(z,num_classes=self.model.config['model.z_dim'])
        return self.model.recursive_update(phi, u, z, params)

    # TRAINING FUNCTIONS
    def evaluate(self, sample, add_noise=False):
        """
        uses model to evaluate a sample from the dataloader
        conditions on some number of data points before evaluating the rest with the posterior
        mean over time horizon (dim 1), mean over batch (dim 0). returns a scalar

        add_noise: bool, whether to add a scaled down isotropic gaussian noise every other timestep.
        Most useful for Franka Kitchen Env.
        """
        if add_noise:
            x = self.to_cuda(sample['x'].float()) 
            x = x + self.to_cuda(torch.randn(x.size()))*0.1
        else:
            x = self.to_cuda(sample['x'].float())
        u = self.to_cuda(sample['u'].float())
        input_z = self.to_cuda(sample['z'].to(torch.int64))
        z_dim = self.model.config['model.z_dim']
        if z_dim == 1:
            z = input_z.unsqueeze(-1)
        else:
            z = self.to_cuda(F.one_hot(input_z, num_classes=z_dim))

        # TODO(Ahmed) sometimes data is sampled to N-1 where horizon is N
        # leading to indexing issue. Training Timestep this happens on depends on horizon size
        horizon = min(self.model.config['training.data_horizon'], x.size()[1]) 
        
       
        # batch compute features and targets for BLR
        phi = self.to_cuda(self.model.encoder(x))
        # get prior statistics
        stats = self.model.prior_params()
        stats = [p.unsqueeze(0) for p in stats] # add dim for batch eval
        stats = [self.to_cuda(p) for p in stats] # move to cuda
        # compute log probs after conditioning
        logps = []
        # loop over context data, and condition BLR
        for j in range(horizon):
            phi_ = phi[:,None,j,...]
            u_ = u[:,j,:]
            z_ = z[:,j,:]
            # get posterior likelihood
            logp, stats = self.model.log_predictive_prob(phi_, u_, z_, stats, self.to_cuda, update_params=True)
            logps.append(logp)
        
        total_logp = torch.stack(logps, dim=-1).sum(-1) / horizon
        total_nll = -total_logp.mean()
        return total_nll
    
    def train(self, dataloader, num_train_updates, state_setting=0, batch_size=250, val_dataloader=None, verbose=False, add_noise=False):
        """
        Trains the dynamics model on data.
        Inputs: dataloader: torch DataLoader that returns batches of samples that can 
                            be indexed by 'x', 'u', and 'xp'
                num_train_updates: int specifying the number of gradient steps to take

                state_setting: int specifying the type of observation we feed to the Franka Robot Env.
                
                val_dataloader: torch DataLoader, a dataloader used for validation (optional)
                verbose: bool, whether to print training progress.
                add_noise: bool, whether to add a scaled down isotropic gaussian noise every other timestep.
                Most useful for Franka Kitchen Env.
                
        Progress is logged to a tensorboard summary writer.
        
        Outputs: None. self.model is modified.
        """
        self.reset()
        config = self.model.config
        
        validation_freq = 100
        val_iters = 5

        print_freq = validation_freq if verbose else num_train_updates+1
        data_iter = iter(dataloader)
        
        with trange(num_train_updates, disable=(not verbose or not config['training.tqdm'])) as pbar:
            for idx in pbar:
                # save model params
                if ((idx + 1) % 2500) == 0 or idx == 0:
                    params = self.model.state_dict()
                    path = '/iris/u/ahmedah/alpaca_kitchen_model_checkpoints/setting_{}/alp_param_kitchen_b_{}_{}.pt'.format(state_setting, batch_size, idx)
                    torch.save(params, path)
                try:
                    sample = next(data_iter)
                except StopIteration:
                    # reset data iter
                    data_iter = iter(dataloader)
                    sample = next(data_iter)

                self.optimizer.zero_grad()
                self.model.train()
                if (idx % 2) == 0 and add_noise:
                    total_loss = self.evaluate(sample, add_noise)
                else:
                    total_loss = self.evaluate(sample)

                # compute validation loss
                if idx % validation_freq == 0 and val_dataloader is not None:
                    total_loss_val = []

                    self.model.eval()
                    for k, val_sample in enumerate(val_dataloader):
                        total_loss_val.append( self.evaluate(val_sample) )

                        if k == val_iters-1:
                            total_nll_val = torch.stack(total_loss_val).mean().detach().numpy()
                            self.writer.add_scalar('NLL/Val', total_nll_val, self.train_step)
                            break

                # grad update on logp
                total_nll = total_loss
                total_nll.backward()

            
                nn.utils.clip_grad_norm_(self.model.parameters(),config['training.grad_clip_value'])
                self.optimizer.step()
                if config['training.learning_rate_decay']:
                    self.scheduler.step()

                # ---- logging / summaries ------
                self.train_step += 1
                step = self.train_step

                # tensorboard logging
                self.writer.add_scalar('NLL/Train', total_nll.item(), step)
                
                # tqdm logging
                logdict = {}
                logdict["tr_loss"] = total_nll.cpu().detach().numpy()
                if val_dataloader is not None:
                    logdict["val_loss"] = total_nll_val
                    
                pbar.set_postfix(logdict)
                self.reset()