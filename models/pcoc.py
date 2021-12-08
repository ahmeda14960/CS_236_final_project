import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from .torch_dynamics import TorchAdaptiveDynamics
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from copy import deepcopy

base_config = {
    # torch settings
    'torch.device': 'cuda:0',

    # model settings
    'model.x_dim': 4,
    'model.u_dim': 2,
    'model.z_dim': 8,
    'model.use_input_feats': False,
    'model.hidden_dim': 128,
    'model.phi_dim': 32,
    'model.sigma_eps': [0.02],
    'model.dirichlet_scale': 1,
    'model.term': False,
    'model.condit_z': False,
    'model.multihead': True,
    'model.multihead_dim': 4,
    # training settings
    'training.learning_rate': 5e-3,
    'training.data_horizon': 100,
    'training.learnable_noise': True,
    'training.learning_rate_decay': False,
    'training.lr_decay_rate': 1e-3,
    'training.learnable_dirichlet': True,
    'training.from_mnist':False,

}

class PcocImitation(nn.Module):
    """
    Not set up to run as an independent model within camelid
    For an independent standalone model, use adaptiveDynamicsTorch
    """
    def __init__(self, config={}, model_path=None):
        super().__init__()

        self.config = deepcopy(base_config)
        if model_path is not None:
            data = torch.load(model_path)
            config = data["config"]        
            
        self.config.update(config)

        self.x_dim = self.config['model.x_dim']
        self.u_dim = self.config['model.u_dim']
        self.z_dim = 2*self.config['model.z_dim'] if self.config['model.term'] else self.config['model.z_dim']
        self.torch_dev = self.config['torch.device']
        self.h_dim = self.config['model.multihead_dim']

        self.phi_dim = 5 if self.config['model.use_input_feats'] else self.config['model.phi_dim'] 

        self.sigma_eps = self.config['model.sigma_eps']*self.phi_dim
        self.sig_diag = torch.from_numpy(np.log(self.sigma_eps)).float()
        self.logSigEps = nn.Parameter(self.sig_diag.repeat(self.z_dim, 1), requires_grad=self.config['training.learnable_noise'])
        
        self.L_asym = nn.Parameter(torch.randn(self.u_dim, self.phi_dim, self.phi_dim)/self.phi_dim**2)
        self.L_base = nn.Parameter(torch.linspace(-5,0, self.phi_dim).repeat(self.u_dim,1))

        # PCOC update:
        dir_scale = self.config['model.dirichlet_scale']
        self.logL = nn.Parameter(torch.randn( self.phi_dim))
        self.Q = nn.Parameter(torch.randn(self.phi_dim, 1)*4/(np.sqrt(self.phi_dim)+ np.sqrt(self.u_dim)))
        self.log_dir_prior = nn.Parameter(dir_scale*torch.ones(1), requires_grad=config['training.learnable_dirichlet'])
        # PCOC update/
        # u_dim here is dimensionality k mentioned in gaussian pdf
        self.normal_nll_const = self.u_dim*np.log(2*np.pi)
    
        # conv encoder weights
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5).to(torch.device(self.torch_dev))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2).to(torch.device(self.torch_dev))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2).to(torch.device(self.torch_dev))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5).to(torch.device(self.torch_dev))
        self.mat1 = nn.Linear(320, 50).to(torch.device(self.torch_dev))
        self.mat2 =  nn.Linear(50, 10).to(torch.device(self.torch_dev))

        if self.config['model.lstm']:
            self.lstm = nn.LSTM(input_size=self.x_dim, hidden_size=32, num_layers=2, batch_first=True).cuda()

        # ensure NN is on GPU
        self.backbone = self.get_encoder(self.config)
        self.cuda = int(config['torch.device'][-1])
        if model_path is not None:
            print("loading state dict")
            self.load_state_dict(data['state_dict'])

    def conv_encoder(self, x):
        out_1 = F.relu(self.maxpool1(self.conv1(x)))
        out_2 = F.relu(self.maxpool2(F.dropout2d(self.conv2(out_1))))
        out_3 = out_2.reshape(100, -1)
        final = self.mat2(F.dropout(F.relu(self.mat1(out_3))))
        return final

    def to_cuda_(self, x):
        if self.cuda<0:
            return x.cpu()
        else:
            return x.to(self.config['torch.device'])#x.cuda(self.cuda)
    
        
    @property
    def L(self):
        return torch.diag(torch.exp(self.logL)).to(torch.device(self.torch_dev))
    @property
    def L_vec(self):
        return torch.exp(self.logL).to(torch.device(self.torch_dev))
    @property
    def dir_weights(self):
        return torch.exp(self.log_dir_prior).to(torch.device(self.torch_dev))

    @property
    def logdetSigEps(self):
        return torch.sum(self.logSigEps).to(torch.device(self.torch_dev))

    @property
    def invSigEps(self):
        return torch.diag_embed(torch.exp(-self.logSigEps)).to(torch.device(self.torch_dev))

    @property
    def invSigEpsVec(self):
        return torch.exp(-self.logSigEps).to(torch.device(self.torch_dev))

    @property
    def SigEpsVec(self):
        return torch.exp(self.logSigEps).to(torch.device(self.torch_dev))

    @property
    def SigEps(self):
        return torch.diag_embed(torch.exp(self.logSigEps)).to(torch.device(self.torch_dev))

    
    def get_encoder(self, config):
        activation = nn.Tanh()
        hid_dim = config['model.hidden_dim']
        phi_dim = config['model.phi_dim']
        x_dim = config['model.x_dim'] + config['model.z_dim'] if config['model.condit_z'] else config['model.x_dim']
        
        if config['training.from_mnist']:
            encoder = self.conv_encoder
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
    
    def encoder(self, x, mnist=False):
        # shapes last output of phi to be 1 (to batch across ydim) x phidim x 1
        if self.config['model.use_input_feats']:
            ones = torch.ones_like(x)[...,0:1]
            phi = torch.cat((x,ones),dim=-1) 
        elif mnist:
            phi = self.backbone(x)
            return phi
        else:
            self.backbone = self.to_cuda_(self.backbone)
            phi = self.backbone(x)

        return phi[...,None].unsqueeze(-3)        
    
    def prior_params(self):
        return (self.Q.unsqueeze(-3).repeat(self.z_dim,1,1),
                self.L.unsqueeze(-3).repeat(self.z_dim,1,1),
                self.dir_weights.repeat(self.z_dim))

    def recursive_update(self, phi, z, params):
        """
            inputs: phi: shape (..., 1, phi_dim, 1)
                    z:   shape (..., z_dim)   (one-hot)
                    params: tuple of Q, L
                        Q: shape (..., z_dim, phi_dim, 1)
                        L: shape (..., z_dim, phi_dim, phi_dim)
                        dir_weights: shape (..., z_dim)
        """
        Q, L, dir_weights = params
        z = z.to(torch.device(self.torch_dev))
        # Indexing with None adds a new dimension
        old_z = z
        z = z[..., None, None] # new z tensor is (batch_size, 1, 1)
        sig_update = self.invSigEps * z # should be (..., z_dim, phi_dim,  phi_dim)
       
        # TODO(ensure I'm using diag matrix, and ensure they are same dim)
        L = L + sig_update
      
        Q_update = sig_update @ phi.squeeze(-3) # should be (..., z_dim, phi_dim, 1)
       
        Q = Q + Q_update
       
        dir_weights = dir_weights + old_z  # count based, should be (..., z_dim)
        return (Q, L, dir_weights)

    def log_predictive_prob(self, phi, z, posterior_params, update_params=False, verbose=False):
        """
            input:  phi: shape (..., 1, phi_dim)
                    z: shape (..., z_dim)
                    posterior_params: tuple of Q, L:
                        Q: shape (..., z_dim, phi_dim, 1)
                        L: shape (..., z_dim, phi_dim, phi_dim)
                        dir_weights: shape (..., z_dim)
                    update_params: bool, whether to perform recursive update on
                                   posterior params and return updated params
            output: logp: log p(y | x, posterior_parms)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """

        # compute full predictions for each skill dim; weight predictions by skill prob
        Q, L, dir_weights = posterior_params
        # PCOC updates
        # (..., z_dim, phi_dim, phi_dim) @  (..., z_dim, phi_dim, 1) -> (..., z_dim, phi_dim, 1)
        # Linv @ Q
        Linv = torch.inverse(L)
        mu =   Linv @ Q  # (..., z_dim, phi_dim, 1)
        
        err = phi.squeeze(1) - mu  # (..., z_dim, phi_dim, 1)
        # (..., z_dim, phi_dim)
        # invert L before
        # no need for a diagonal here, L should be diagonal already.
        
        L_diag = torch.diagonal(Linv, dim1=-1, dim2=-2)
        pred_cov_diag = L_diag + self.SigEpsVec
        # print('pred cov diag', pred_cov_diag.size())
        # should be (..., z_dim, phi_dim)
        nll_quadform = err.squeeze(-1)**2 / pred_cov_diag
        nll_logdet = -torch.log(pred_cov_diag)
        # Should be dim B and summed across last dimension, (B, phi_dim)
        logp = -0.5*(self.normal_nll_const + nll_quadform + nll_logdet).squeeze(-1).squeeze(-1)
        # averaging step --- re-evaluate when moving to soft labels
        # prod is (..., z_dim, phi_dim, phi_dim) in Alpaca it's (..., z_dim, phi_dim, phi_dim)
        # now should be (..., z_dim, phi_dim)
        prod = logp #* z.unsqueeze(-1)
        prod = torch.mean(prod, dim=-1)

        if verbose:
            print('logp', logp.size())
            print('phi', phi.squeeze(1).size())
            print('test err', err.size())
            print('test pred_cov_diag', pred_cov_diag.size())
            print('test pred_cov_diag', pred_cov_diag.size())
            print('test nll_quadform', nll_quadform.size())
            print('test nll_logdet', nll_logdet.size())

        if update_params:
            updated_params = self.recursive_update(phi, z,posterior_params)
            return prod, updated_params
        
        #TODO(Ahmed) ensure dim is matched correctly
        # torch.sum(prod, dim=-4)
        return prod


    def forward(self, x, posterior_params, mnist=False, verbose=False):
        """
            input: x
            output: (LOGITS for) log p(x | y) for all y (..., z_dim)
        """
        # PCOC
        Q, L, dir_weights = posterior_params
        Linv = torch.inverse(L)
        phi = self.encoder(x, mnist)
        # PCOC updates
        # (..., z_dim, phi_dim, phi_dim) @  (..., z_dim, phi_dim, 1) -> (..., z_dim, phi_dim, 1)
        mu =   Linv @ Q  # (..., z_dim, phi_dim, 1)
        if mnist:
            phi = phi.reshape((-1, 1, 10)).unsqueeze(-1).unsqueeze(1)
        err = phi.squeeze(1) - mu  # (..., z_dim, phi_dim, 1)
        
        # (..., z_dim, phi_dim)
      
        pred_cov_diag = torch.diagonal(Linv, dim1=-1, dim2=-2) + self.SigEpsVec
        # should be (..., z_dim, phi_dim)
        # print('pred diag', pred_cov_diag.size())
        # print('err', err.size())
        nll_quadform = err.squeeze(-1)**2 / pred_cov_diag
        nll_logdet = -torch.log(pred_cov_diag)
       
        # Should be dim B and summed across last dimension, (B, phi_dim)
        logp = -0.5*(self.normal_nll_const + nll_quadform + nll_logdet).squeeze(-1).squeeze(-1)
        logp = torch.mean(logp, dim=-1)
        p_y = torch.log(dir_weights / dir_weights.sum(-1, keepdim=True))
        # multiply by p(y) posterior to get p(x, y) (adding bc logs)
        # p_y is size (..., z_dim)
        logp += p_y

        if verbose:
            print('py', p_y.size())
            print('logp', logp.size())
            print('phi', phi.squeeze(1).size())
            print('test err', err.size())
            print('test pred_cov_diag', pred_cov_diag.size())
            print('test pred_cov_diag', pred_cov_diag.size())
            print('test nll_quadform', nll_quadform.size())
            print('test nll_logdet', nll_logdet.size())

        return logp

class PcocImitationStateful():
    """
    Wrapper class that maps torch PCOC with training functions and
    online prediction and adaptation functions.
    """
    def __init__(self, model, cuda=-1):
        """
        Inputs:
        model: PCOCTorch object
        f_nom: function mapping tensors x -> to tensor y
        
        Sets up SummaryWriter to log to for Tensorboard visualization.
        """
        super().__init__()
        self.model = model
        self.reset()

        # used for annealing during training
        self.train_step = 0
        
        path = 'Pcoctorch'
        self.writer = SummaryWriter('./runs/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))

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

    def incorporate_transition_torch(self, params, x, z):
        """
        returns posterior params after updating params with transition x, u, xp
        """
        phi = self.model.encoder(self.to_cuda(x))
        Q, L, dir_weights = params
        params = (self.to_cuda(Q), self.to_cuda(L), self.to_cuda(dir_weights))
        # Make one-hot tensor of shape (..., z_dim) with num_classes == z_dim
        z = F.one_hot(z,num_classes=self.model.z_dim)
        return self.model.recursive_update(phi, z, params)

    # TRAINING FUNCTIONS
    def evaluate(self, sample, binv, reweight=False):
        """
        uses model to evaluate a sample from the dataloader
        conditions on some number of data points before evaluating the rest with the posterior
        mean over time horizon (dim 1), mean over batch (dim 0). returns a scalar
        """
        horizon = self.model.config['training.data_horizon']
        
        if self.model.config['training.from_mnist']:
            x, y = sample
            x = self.to_cuda(x.float()).reshape(100, -1)
            y = self.to_cuda(y.float())
        else:
            if self.model.config['model.term']:
                x = self.to_cuda(sample['x'].float())
                z = self.to_cuda(F.one_hot(self.to_cuda(sample['b'].to(torch.int64)), num_classes=self.model.z_dim))
            elif binv:
                x = self.to_cuda(sample['x_bin'].float())
                z = self.to_cuda(F.one_hot(self.to_cuda(sample['h_bin'].to(torch.int64))))
            else:
                x = self.to_cuda(sample['x'].float())
                z = self.to_cuda(F.one_hot(self.to_cuda(sample['z'].to(torch.int64)), num_classes=self.model.z_dim))
        # batch compute features and targets for BLR
        if self.model.config['model.condit_z']:
            # condition on skill from last time timestep to better predict future skills
            skills = self.to_cuda(F.one_hot(self.to_cuda(sample['z'].to(torch.int64)), num_classes=self.model.config['model.z_dim']))
            new_z = self.to_cuda(torch.zeros(skills.size()))
            new_z[1:] = skills[:-1]
            phi = self.to_cuda(self.model.encoder(torch.cat([x, new_z], dim=2)))

        elif self.model.config['model.lstm']:
            out, hid = self.model.lstm(x)
            h0, c0 = hid
            phi = out.unsqueeze(2).unsqueeze(-1)
        else:
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
            z_ = z[:,j,:]
            # get posterior likelihood
            logp, stats = self.model.log_predictive_prob(phi_, z_, stats, update_params=True)
            _, _, dir_weights = stats
            if reweight:
                # re-weight gradients by inverse of dir weights
                logp = logp * (1 / torch.log(dir_weights))
            logps.append(logp)

        total_logp = torch.stack(logps, dim=-1).sum(-1) / horizon
        return -total_logp.mean()
    
    def train(self, dataloader, num_train_updates, val_dataloader=None, verbose=False, binv=False):
        """
        Trains the dynamics model on data.
        Inputs: dataloader: torch DataLoader that returns batches of samples that can 
                            be indexed by 'x', 'u', and 'xp'
                num_train_updates: int specifying the number of gradient steps to take
                
                val_dataloader: torch DataLoader, a dataloader used for validation (optional)
                verbose: bool, whether to print training progress.
                
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
                try:
                    sample = next(data_iter)
                except StopIteration:
                    # reset data iter
                    data_iter = iter(dataloader)
                    sample = next(data_iter)

                self.optimizer.zero_grad()
                self.model.train()
                total_loss = self.evaluate(sample, binv=binv)

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

            
                # remove clipping
                #nn.utils.clip_grad_norm_(self.model.parameters(),config['training.grad_clip_value'])
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
