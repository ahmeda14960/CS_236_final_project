import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse

from models import AlpacaMH
from envs import LqrExpertDataset


def plot(alp_model, params, dataset, T=500):
    x = (np.random.randn(4)*2)[:,None]
    U = np.empty((T,2))
    X = np.empty((T,4))
    Z = np.empty(T)
    z = 0
    for t in range(T):
        if (t % (T/4)) == 0:
            z = (z + 1) % 4
        X[t] = x.T
        # unsqueeze to add batch dim for x_tensor
        x_tensor = torch.tensor(x)[:,0].float().cuda().unsqueeze(0)
        # squeeze to remove batch dim
        mu, sigma = [p.squeeze(0) for p in alp_model.forward(x_tensor, params)]
        mu = mu[z].cpu().detach().numpy()
        cov_tensor = sigma[z].cpu().detach()
        cov = cov_tensor.numpy()
        
        u = np.expand_dims(np.random.multivariate_normal(np.squeeze(mu), np.diag(cov)), axis=1)
        U[t] = u.T
        xp = dataset.dynamics(x,u)
        x = xp.copy()
        Z[t] = z
    fig = plt.figure()
    plt.plot(X[:,0],X[:,1])
    plt.show() 
    plt.savefig(f'/iris/u/ahmedah/incremental-skill-learning/exps/alpaca/x_path_{T}.png')
    plt.clf()

    # Viz skills over time
    plt.plot(Z)
    plt.show()
    plt.savefig(f'/iris/u/ahmedah/incremental-skill-learning/exps/alpaca/z_path_{T}.png')
    plt.clf()

if __name__ == '__main__':
    # set up argument parser
    parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
    parser.add_argument('-n', '--noise', action='store_true')
    parser.add_argument('--env', type=str, default='lqr')
    parser.add_argument('-b', '--bsize', type=int, default=500)
    parser.add_argument('-sig', type=float, default=1e-2)
    parser.add_argument('-lscale', type=float, default=-1e-2)
    parser.add_argument('-lr', type=float, default=5e-3)
    parser.add_argument('-chk', '--checkpoint', type=int, default=0)
    args = parser.parse_args()
    config = {
    'env' : 'point_mass',
    'model.x_dim': 4,
    'model.u_dim': 2,
    'model.z_dim': 4,
    'model.use_input_feats': False,
    'model.hid_dim': 128,
    'model.phi_dim': 32,
    'model.sigep_scale': args.sig,
    'model.init.L_scale': args.lscale,
    'model.glorot_init': False,
    'model.simp_sig_ep': False,
    'model.phi_activation': nn.ELU(),

    'training.learning_rate': args.lr,
    'training.data_horizon': 100,
    'training.learnable_noise': False,
    'training.learning_rate_decay': False,
    'training.lr_decay_rate': 1e-3,
    }
    #TODO add proper cuda changes to this test file
    device = torch.device('cuda:0')
    alpaca_model = AlpacaMH(device=device, config=config)
    alpaca_model.to(device)
    dataset = LqrExpertDataset(trajlen=500, datalen=1e6, goal_scale=10)
    dataloader = DataLoader(dataset, batch_size=500)

    alpaca_model.load_state_dict(torch.load(f'/iris/u/ahmedah/incremental-skill-learning/checkpoints/alpaca_point_mass/{args.checkpoint}.pt'))
    stats = alpaca_model.prior_params()
    stats = [p.cuda() for p in stats]
    
    # condition on new data to adapt params
    sample = dataset.__getitem__(0)
    for t in range(dataset.trajlen):
        # add unsqueeze to include dummy batch dimension
        x = torch.tensor(sample['x'][t]).float().cuda().unsqueeze(0)
        u = torch.tensor(sample['u'][t]).float().unsqueeze(0).cuda().unsqueeze(1)
        z = F.one_hot(torch.tensor(sample['z'][t]).to(torch.int64).cuda(), num_classes=alpaca_model.z_dim).unsqueeze(0)
        phi = alpaca_model.encoder(x).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        
        stats = alpaca_model.recursive_update(phi, u, z, stats)

    plot(alp_model=alpaca_model, params=stats, dataset=dataset)


