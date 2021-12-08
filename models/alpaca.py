import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

base_config = {
    # model parameters
    'model.x_dim': None,
    'model.u_dim': None,
    'model.z_dim': None,
    'model.hidden_dim': 128,
    'model.phi_dim': 128,
    'model.phi_activation': nn.ELU(),
    'model.sigep_scale': 0.1,
    'model.use_input_feats': False,
    'model.init.L_scale': -1e-2,
    'model.glorot_init': False,
    'model.simp_sig_ep': False,
    
    # training parameters
    'training.learnable_noise': True,
    'training.data_horizon': 100,
    'training.learning_rate': 1e-3,
    'training.tqdm': True,
    'training.learning_rate_decay': False,
    'training.lr_decay_rate': 1e-3,
    'training.grad_clip_value':1,
}


class AlpacaMH(nn.Module):
    """
    Multi-head implementation of ALPaCA, a meta-learning algorithm
    for online bayesian regression for a skill conditioned imitation setting.
    """

    def __init__(self, device, config={}):
        super().__init__()

        self.device = device
        self.config = deepcopy(base_config)
        self.config.update(config)

        # model dimensions
        self.x_dim = self.config['model.x_dim']
        self.u_dim = self.config['model.u_dim']
        self.z_dim = self.config['model.z_dim']
        self.phi_dim = self.config['model.phi_dim']
        self.hid_dim = self.config['model.hid_dim']

        if self.config['model.glorot_init']:
            # ALPaCA statistics
            # Init Q matrix with Glorot initialization. Note that when comparing to the original ALPaCA
            # paper, we are technically modeling Q^t for ease of batching. We also add a placeholder dimension
            # to highlight we maintain one Q matrix for each output entry, unlike for Linv which we model per output entry.
            self.Q = nn.Parameter(torch.randn(1, self.u_dim, 1, self.phi_dim) * (np.sqrt(2/(self.u_dim + self.phi_dim)))).to(device)

            # L matrix used for Cholesky decomposition, i.e. A = LL^T. Also with Glorot initialization
            self.L_asym = nn.Parameter(torch.randn(1, self.u_dim, self.phi_dim, self.phi_dim) * (np.sqrt(2/(self.phi_dim + self.phi_dim)))).to(device)

            # Scaling matrix. Must be a diagonal matrix with all positive entries
            self.L_scale = nn.Parameter(torch.eye(self.phi_dim) * self.config['model.init.L_scale']).to(device)
        else:
            # init from James' repo
            print('diff init')
            self.Q = nn.Parameter(torch.randn(1, self.u_dim, 1, self.phi_dim)*4/(np.sqrt(self.phi_dim)+ np.sqrt(self.u_dim))).to(device)
            self.L_asym = nn.Parameter(torch.randn(1, self.u_dim, self.phi_dim, self.phi_dim)/self.phi_dim**2).to(device)
            self.L_scale = nn.Parameter(torch.linspace(-5,0, self.phi_dim).repeat(self.u_dim,1)).to(device)
        
        self.normal_ll_const = self.u_dim*np.log(2*np.pi)

        # parameters for diagonal noise covariance. Modeled as log so we can exponentiate
        # to maintain a positive definite matrix.
        if self.config['model.simp_sig_ep']:
            self.logSigEps = nn.Parameter(torch.ones(self.u_dim)*self.config['model.sigep_scale'], requires_grad=self.config['training.learnable_noise']).to(device)
        else:
            print('using original sig ep implementation')
            self.sigma_eps = [self.config['model.sigep_scale']]*self.u_dim
            self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)).float(), requires_grad=self.config['training.learnable_noise']).to(device)
        
        # encoding network
        self.phi_activation = self.config['model.phi_activation']
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.phi_dim)
        ).to(device)

    def prior_params(self):
        # repeat parameters to maintain separate statistics for each skill
        return(self.Linv.unsqueeze(-4).repeat(1, self.z_dim, 1, 1, 1), 
                self.Q.unsqueeze(-4).repeat(1, self.z_dim, 1, 1, 1))


    # init Linv Block, return LL^T plus scaling matrix to ensure positive definite matrix.
    # Notice that Linv[k] is (u_dim, phi_dim, phi_dim). This is because we 
    # maintaing a separate prior covariance matrix for each output dimension for 
    # increased expressiveness in modeling epistemic uncertainty, so it's block diagonal. 
    # Everything else carries over from the original ALPaCA paper
    @property
    def Linv(self):
        if self.config['model.glorot_init']:
            return self.L_asym @ self.L_asym.transpose(-1, -2) + torch.exp(self.L_scale)
        else:
            return self.L_asym @ self.L_asym.transpose(-1, -2) + torch.diag_embed(torch.exp(self.L_scale))

    # diagonal entries of noise covariance (since it's a diagonal matrix)
    @property
    def SigEp(self):
        return torch.exp(self.logSigEps)

    # ensure that Linv remains positive definite by checking 
    # if we can take cholesky decomp. Otherwise, an error is thrown
    def test_Linv(self, Linv):
        return torch.linalg.cholesky(Linv)

    def recursive_update(self, phi, u, z, stats):
        """
        Recursive update rule for ALPaCA statistics
        Args:
            phi: encoding tensor (batch_dim, phi_dim)
            u: action tensor (batch_dim, u_dim)
            z: one-hot skill tensor (batch_dim, z_dim)
        
        Returns:
            stats: tuple of Linv, Q
                Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)
            None. Updates ALPaCA statistics internally.
        """
        Linv, Q = stats
        # multiply L tensor by expanded skill to zero-out other skills in
        # the recursive update
        z_condit = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        L_onehot = z_condit * Linv

        # Do the same for the action tensor
        u_condit = z_condit.squeeze(-1).squeeze(-1) * u

        # multiply Linv and phi to get one of the terms for the outer product.
        Linv_phi = L_onehot @ phi
        Linv_phi_Linv = Linv_phi @ Linv_phi.transpose(-1, -2)

        # bottom term for scaling update 
        sigma_scale = 1. + phi.transpose(-1, -2) @ L_onehot @ phi

        # Update for Linv
        Linv_update = 1./sigma_scale * Linv_phi_Linv
        Linv = Linv - Linv_update

        # update rule for Q^t tensor
        Q_update = u_condit.unsqueeze(-1) @ phi.transpose(-1, -2).squeeze(1)
        # unsqueeze to match dimension selecting Linv in Q
        Q = Q + Q_update.unsqueeze(-2)

        return (Linv, Q)


    def log_predictive_prob(self, phi, u, z, stats):
        """
        Helper function to get posterior predictive density from ALPaCA statistics
            Args:
                phi: encoding tensor (batch_dim, phi_dim)
                u: action tensor (batch_dim, u_dim)
                z: one-hot skill tensor (batch_dim, z_dim)
                stats: tuple of ALPaCA statistics
                    Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                    Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)
            Returns:
                logp: tensor for log-predictive prob, i.e.
                log( p(y | x, posterior_params) )
        """
        Linv, Q = stats
        # Define matrix K which can be thought of as the last "layer"
        # of our encoder.
        # note that we technically model K^t relative to the original ALPaCA paper
        K = Q @ Linv.transpose(-1, -2)
        # squeeze to get rid of dimension selecting Linv
        K = K.squeeze(-2)

        # define term to scale our noise matrix to get predictive covariance.
        # Unsqueeze to add dummy batch dimension to Linv
        sigma_scale = 1. +  phi.transpose(-1, -2) @ Linv @ phi

        # squeeze so we are left with scalar per batch, skill and output dim
        sigma_scale = sigma_scale.squeeze(-1).squeeze(-1)

       
        pred_cov_diag = self.SigEp*sigma_scale
        # calculate log determinant of predictive covariance.
        # Given diagonal assumption, this is just the sum of
        # the elementwise log of the diagonal terms
        if self.config['training.sum_act']:
            log_det = torch.log(pred_cov_diag).sum(dim=-1)
        else:
            log_det = torch.log(pred_cov_diag)


        # mean, defined as K^t phi(x). unsqueeze phi to remove
        # dimension selecting Linv.
        
        mu = K @ phi.squeeze(1)
        # residual after regressing on action from model mean
        
        err = u - mu.squeeze(-1)

        # inverse of diagonal matrix is 1 over diagonal entries
        
        pred_cov_inv = torch.ones(pred_cov_diag.shape).to(self.device) / pred_cov_diag

        # calculate (y - mu)^t @ Sigma^-1 @ (mu - y)
        # since Sigma is modeled with a diagonal matrix, this amounts to squaring err
        # and scaling by pred_cov_inv
        
        # experiment with not summing over last dimension over actions
        if self.config['training.sum_act']:
            quad_term = (err.pow(2) * pred_cov_inv).sum(-1)
        else:
            quad_term = (err.pow(2) * pred_cov_inv)
        

        # log prob of multivariate normal, given pdf p(x):
        # log p(x) = -0.5(k*log(2*pi)) + log(det(Sigma)) + MH(y, mu, Sigma)^2)
        # where MH is mahalanobois distance.
        logp = -0.5*(self.normal_ll_const + log_det + quad_term)

        # multiply by one-hot z to only backprop 
        # through statistics for the current skill
        z = z if self.config['training.sum_act'] else z.unsqueeze(-1)
        logp = logp * z

        return logp

    def forward(self, x, stats):
        """
        Forward pass to grab mean and covariance tensors for current ALPaCA statistics
        Args:
        x  : input tensor (batch_dim, x_dim)
        stats: tuple of ALPaCA statistics
                    Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                    Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)

        Returns:
        mu : mean tensor for posterior predictive (z_dim, y_dim)
        var : variance tensor for posterior predictive (z_dim, y_dim, y_dim)
        """
        
        Linv, Q = stats
        # grab encoding for phi, unsqueeze to match
        # unsqueeze ops ensure that phi has the shape
		# (batch_dim, z_dim, u_dim, phi_dim, 1)
		# where z_dim = u_dim = 1 since we use the same encoding
		# across skills and output dimensions.
        phi = self.encoder(x).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        # (batch_dim, 1, 1, phi_dim, 1)

        # Define matrix K which can be thought of as the last "layer"
        # of our encoder.
        # note that we technically model K^t relative to the original ALPaCA paper
        K = Q @ Linv.transpose(-1, -2)
        # squeeze to get rid of dimension selecting Linv
        K = K.squeeze(-2)

        # mean, defined as K^t phi(x). unsqueeze phi to remove
        # dimension selecting Linv.
        mu = K @ phi.squeeze(1)


        # define term to scale our noise matrix to get predictive covariance.
        # Unsqueeze to add dummy batch dimension to Linv
        sigma_scale = 1. +  phi.transpose(-1, -2) @ Linv @ phi

        # squeeze so we are left with scalar per batch, skill and output dim
        sigma_scale = sigma_scale.squeeze(-1).squeeze(-1)

       
        pred_cov_diag = self.SigEp*sigma_scale
        return mu, pred_cov_diag

class AlpacaMoca(nn.Module):
    """
    Multi-head implementation of ALPaCA, a meta-learning algorithm
    for online bayesian regression but for MOCA. The number
    of heads here correspond to the possible number of run-lengths
    """

    def __init__(self, device, config={}):
        super().__init__()

        self.device = device
        self.config = deepcopy(base_config)
        self.config.update(config)

        # model dimensions
        self.x_dim = self.config['model.x_dim']
        self.u_dim = self.config['model.u_dim']
        self.z_dim = self.config['model.z_dim']
        self.phi_dim = self.config['model.phi_dim']
        self.hid_dim = self.config['model.hid_dim']

        if self.config['model.glorot_init']:
            # ALPaCA statistics
            # Init Q matrix with Glorot initialization. Note that when comparing to the original ALPaCA
            # paper, we are technically modeling Q^t for ease of batching. We also add a placeholder dimension
            # to highlight we maintain one Q matrix for each output entry, unlike for Linv which we model per output entry.
            self.Q = nn.Parameter(torch.randn(1, self.u_dim, 1, self.phi_dim) * (np.sqrt(2/(self.u_dim + self.phi_dim)))).to(device)

            # L matrix used for Cholesky decomposition, i.e. A = LL^T. Also with Glorot initialization
            self.L_asym = nn.Parameter(torch.randn(1, self.u_dim, self.phi_dim, self.phi_dim) * (np.sqrt(2/(self.phi_dim + self.phi_dim)))).to(device)

            # Scaling matrix. Must be a diagonal matrix with all positive entries
            self.L_scale = nn.Parameter(torch.eye(self.phi_dim) * self.config['model.init.L_scale']).to(device)
        else:
            # init from James' repo
            print('diff init')
            self.Q = nn.Parameter(torch.randn(1, self.u_dim, 1, self.phi_dim)*4/(np.sqrt(self.phi_dim)+ np.sqrt(self.u_dim))).to(device)
            self.L_asym = nn.Parameter(torch.randn(1, self.u_dim, self.phi_dim, self.phi_dim)/self.phi_dim**2).to(device)
            self.L_scale = nn.Parameter(torch.linspace(-5,0, self.phi_dim).repeat(self.u_dim,1)).to(device)
        
        self.normal_ll_const = self.u_dim*np.log(2*np.pi)

        # parameters for diagonal noise covariance. Modeled as log so we can exponentiate
        # to maintain a positive definite matrix.
        if self.config['model.simp_sig_ep']:
            self.logSigEps = nn.Parameter(torch.ones(self.u_dim)*self.config['model.sigep_scale'], requires_grad=self.config['training.learnable_noise']).to(device)
        else:
            print('using original sig ep implementation')
            self.sigma_eps = [self.config['model.sigep_scale']]*self.u_dim
            self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)).float(), requires_grad=self.config['training.learnable_noise']).to(device)
        
        # encoding network
        self.phi_activation = self.config['model.phi_activation']
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.hid_dim),
            self.phi_activation,
            nn.Linear(self.hid_dim, self.phi_dim)
        ).to(device)

    def prior_params(self):
        # repeat parameters to maintain separate statistics for each skill
        return(self.Linv.unsqueeze(-4).repeat(1, self.z_dim, 1, 1, 1), 
                self.Q.unsqueeze(-4).repeat(1, self.z_dim, 1, 1, 1))


    # init Linv Block, return LL^T plus scaling matrix to ensure positive definite matrix.
    # Notice that Linv[k] is (u_dim, phi_dim, phi_dim). This is because we 
    # maintaing a separate prior covariance matrix for each output dimension for 
    # increased expressiveness in modeling epistemic uncertainty, so it's block diagonal. 
    # Everything else carries over from the original ALPaCA paper
    @property
    def Linv(self):
        if self.config['model.glorot_init']:
            return self.L_asym @ self.L_asym.transpose(-1, -2) + torch.exp(self.L_scale)
        else:
            return self.L_asym @ self.L_asym.transpose(-1, -2) + torch.diag_embed(torch.exp(self.L_scale))

    # diagonal entries of noise covariance (since it's a diagonal matrix)
    @property
    def SigEp(self):
        return torch.exp(self.logSigEps)

    # ensure that Linv remains positive definite by checking 
    # if we can take cholesky decomp. Otherwise, an error is thrown
    def test_Linv(self, Linv):
        return torch.linalg.cholesky(Linv)

    def recursive_update(self, phi, u, stats):
        """
        Recursive update rule for ALPaCA statistics
        Args:
            phi: encoding tensor (batch_dim, phi_dim)
            u: action tensor (batch_dim, u_dim)
        
        Returns:
            stats: tuple of Linv, Q
                Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)
            None. Updates ALPaCA statistics internally.
        """
        Linv, Q = stats

        # multiply Linv and phi to get one of the terms for the outer product.
        Linv_phi = Linv @ phi
        Linv_phi_Linv = Linv_phi @ Linv_phi.transpose(-1, -2)

        # bottom term for scaling update 
        sigma_scale = 1. + phi.transpose(-1, -2) @ Linv @ phi

        # Update for Linv
        Linv_update = 1./sigma_scale * Linv_phi_Linv
        Linv = Linv - Linv_update

        # update rule for Q^t tensor
        Q_update = u.unsqueeze(-1) @ phi.transpose(-1, -2).squeeze(1)
        # unsqueeze to match dimension selecting Linv in Q
        Q = Q + Q_update.unsqueeze(-2)

        return (Linv, Q)


    def log_predictive_prob(self, phi, u, stats):
        """
        Helper function to get posterior predictive density from ALPaCA statistics
            Args:
                phi: encoding tensor (batch_dim, phi_dim)
                u: action tensor (batch_dim, u_dim)
                stats: tuple of ALPaCA statistics
                    Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                    Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)
            Returns:
                logp: tensor for log-predictive prob, i.e.
                log( p(y | x, posterior_params) )
        """
        Linv, Q = stats
        # Define matrix K which can be thought of as the last "layer"
        # of our encoder.
        # note that we technically model K^t relative to the original ALPaCA paper
        K = Q @ Linv.transpose(-1, -2)
        # squeeze to get rid of dimension selecting Linv
        K = K.squeeze(-2)

        # define term to scale our noise matrix to get predictive covariance.
        # Unsqueeze to add dummy batch dimension to Linv
        sigma_scale = 1. +  phi.transpose(-1, -2) @ Linv @ phi

        # squeeze so we are left with scalar per batch, skill and output dim
        sigma_scale = sigma_scale.squeeze(-1).squeeze(-1)

       
        pred_cov_diag = self.SigEp*sigma_scale
        # calculate log determinant of predictive covariance.
        # Given diagonal assumption, this is just the sum of
        # the elementwise log of the diagonal terms
        if self.config['training.sum_act']:
            log_det = torch.log(pred_cov_diag).sum(dim=-1)
        else:
            log_det = torch.log(pred_cov_diag)


        # mean, defined as K^t phi(x). unsqueeze phi to remove
        # dimension selecting Linv.
        
        mu = K @ phi.squeeze(1)
        # residual after regressing on action from model mean
        
        err = u - mu.squeeze(-1)

        # inverse of diagonal matrix is 1 over diagonal entries
        
        pred_cov_inv = torch.ones(pred_cov_diag.shape).to(self.device) / pred_cov_diag

        # calculate (y - mu)^t @ Sigma^-1 @ (mu - y)
        # since Sigma is modeled with a diagonal matrix, this amounts to squaring err
        # and scaling by pred_cov_inv
        
        # experiment with not summing over last dimension over actions
        if self.config['training.sum_act']:
            quad_term = (err.pow(2) * pred_cov_inv).sum(-1)
        else:
            quad_term = (err.pow(2) * pred_cov_inv)
        

        # log prob of multivariate normal, given pdf p(x):
        # log p(x) = -0.5(k*log(2*pi)) + log(det(Sigma)) + MH(y, mu, Sigma)^2)
        # where MH is mahalanobois distance.
        logp = -0.5*(self.normal_ll_const + log_det + quad_term)

        return logp

    def forward(self, x, stats):
        """
        Forward pass to grab mean and covariance tensors for current ALPaCA statistics
        Args:
        x  : input tensor (batch_dim, x_dim)
        stats: tuple of ALPaCA statistics
                    Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
                    Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)

        Returns:
        mu : mean tensor for posterior predictive (z_dim, y_dim)
        var : variance tensor for posterior predictive (z_dim, y_dim, y_dim)
        """
        Linv, Q = stats
        # grab encoding for phi, unsqueeze to match
        # unsqueeze ops ensure that phi has the shape
		# (batch_dim, z_dim, u_dim, phi_dim, 1)
		# where z_dim = u_dim = 1 since we use the same encoding
		# across skills and output dimensions.
        phi = self.encoder(x).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        # (batch_dim, 1, 1, phi_dim, 1)

        # Define matrix K which can be thought of as the last "layer"
        # of our encoder.
        # note that we technically model K^t relative to the original ALPaCA paper
        K = Q @ Linv.transpose(-1, -2)
        # squeeze to get rid of dimension selecting Linv
        K = K.squeeze(-2)

        # mean, defined as K^t phi(x). unsqueeze phi to remove
        # dimension selecting Linv.
        mu = K @ phi.squeeze(1)


        # define term to scale our noise matrix to get predictive covariance.
        # Unsqueeze to add dummy batch dimension to Linv
        sigma_scale = 1. +  phi.transpose(-1, -2) @ Linv @ phi

        # squeeze so we are left with scalar per batch, skill and output dim
        sigma_scale = sigma_scale.squeeze(-1).squeeze(-1)

       
        pred_cov_diag = self.SigEp*sigma_scale
        return mu, pred_cov_diag
