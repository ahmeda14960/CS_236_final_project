import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse

from models import AlpacaImitation, AlpacaImitationStateful
from envs import LqrExpertDataset


def plot(alp_model, params, dataset, T=500):
    x = (np.random.randn(4))[:,None]
    U = np.empty((T,2))
    X = np.empty((T,4))
    Z = np.empty(T)
    z = 3
    for t in range(T):
        if (t % (T/4)) == 0:
            z = (z + 1) % 4
        X[t] = x.T
        # unsqueeze to add batch dim for x_tensor
        #import ipdb; ipdb.set_trace()
        x_tensor = torch.tensor(x)[:,0].float().cuda().unsqueeze(0)
        # squeeze to remove batch dim
        mu,sig = og_alp_model.model.forward(torch.tensor(x)[:,0].float().cuda(), params)
        mu = mu[z].cpu().detach().numpy()
        cov_tensor = sig[z].cpu().detach()
        cov = cov_tensor.numpy()
        cov = np.linalg.inv(cov)
        u = np.expand_dims(np.random.multivariate_normal(np.squeeze(mu), np.squeeze(cov)), axis=1)
        U[t] = u.T
        xp = dataset.dynamics(x,u)
        x = xp.copy()
        Z[t] = z
    fig = plt.figure()
    plt.plot(X[:,0],X[:,1])
    plt.show() 
    plt.savefig(f'/iris/u/ahmedah/incremental-skill-learning/exps/og_alpaca/og_x_path_{T}.png')
    plt.clf()

    # Viz skills over time
    plt.plot(Z)
    plt.show()
    plt.savefig(f'/iris/u/ahmedah/incremental-skill-learning/exps/og_alpaca/og_z_path_{T}.png')
    plt.clf()

if __name__ == '__main__':
    # set up argument parser
    parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
    parser.add_argument('-i', '--iter', nargs='?', type=int, default=200)
    parser.add_argument('-n', '--noise', action='store_true')
    parser.add_argument('--env', type=str, default='lqr')
    parser.add_argument('-b', '--bsize', type=int, default=500)
    parser.add_argument('-sig', type=float, default=1e-2)
    parser.add_argument('-lscale', type=float, default=-1e-2)
    parser.add_argument('-lr', type=float, default=5e-3)
    args = parser.parse_args()
    
    og_config = {
    'model.x_dim': 4,
    'model.u_dim': 2,
    'model.z_dim': 4,
    'model.use_input_feats': False,
    'model.hidden_dim': 128,
    'model.phi_dim': 32,
    'model.sigma_eps': [0.02],

    'training.learning_rate': 5e-3,
    'training.data_horizon': 100,
    'training.learnable_noise': True,
    'training.learning_rate_decay': False,
    'training.lr_decay_rate': 1e-3,
    }

    og_alp_nn = AlpacaImitation(og_config, cuda=0)
    og_alp_model = AlpacaImitationStateful(og_alp_nn, cuda=0)
    dataset = LqrExpertDataset(trajlen=500)
    dataloader = DataLoader(dataset, batch_size=500)

    og_alp_model.train(dataloader, args.iter, val_dataloader=None, verbose=True)
    og_stats = og_alp_model.model.prior_params()
    og_stats = [p.cuda() for p in og_stats]
    
    # condition on new data to adapt params
    sample = dataset.__getitem__(0)
    for t in range(200):
        og_x = torch.tensor(sample['x'][t]).float()
        og_u = torch.tensor(sample['u'][t]).float()
        og_z = torch.tensor(sample['z'][t]).to(torch.int64)
        og_stats = og_alp_model.incorporate_transition_torch(og_stats, og_x, og_u, og_z)
    plot(alp_model=og_alp_model, params=og_stats, dataset=dataset)


