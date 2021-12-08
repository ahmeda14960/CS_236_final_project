import argparse
import datetime
from tqdm import trange
from torch import autograd
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import AlpacaMoca, AlpacaMocaJit, Moca
from envs import LqrExpertDataset, KitchenRobotData, KitchenRobotDataM

def eval_moca(x, u, alp_model, moca_model, dev):
	"""
	Helper function for evaluating MOCA's NLL loss on a given sample.
	Given some horizon, this model iterates through by evaluting on the current
	timestep on the horizon, conditioning on the data, and then continuing to do so
	for every timestep in the horizon. 

	Note that we maintain ALPaCA statistics for every possible run_length,
	so statistics will be (batch_dim, run_dim, ...)

	We take the mean over the time horizon (dim 1), and after stacking these
	we then take the mean over the batch (dim 0)

	Args:
			alp_model: ALPaCA model to be evaluated
			x: input tensor (batch_dim, x_dim)
			u: action tensor (batch_dim, u_dim)
			z: one-hot skill tensor (batch_dim, z_dim)
			dev: device for torch
	Returns:
			total_nll: scalar torch tensor with total nll for given sample
	"""
	# grab prior statistics
	stats = alp_model.prior_params()
	stats = [p.to(dev) for p in stats]
	# (1, run_dim)
	log_prgx = moca_model.prior_params().to(dev)

	# batch compute features
	phi = alp_model.encoder(x)

	horizon = alp_model.config['training.data_horizon']
	log_probs = []
	for j in range(horizon):
		# unsqueeze ops ensure that phi has the shape
		# (batch_dim, z_dim, u_dim, phi_dim, 1)
		# where z_dim = u_dim = 1 since we use the same encoding
		# across skills and output dimensions.
		phi_ = phi[:,j,:].unsqueeze(1).unsqueeze(1).unsqueeze(-1)
		# unsqueeze here to ensure u has the shape
		# (batch_dim, z_dim, u_dim) where z_dim = 1
		# for ease of broadcasting.
		u_ = u[:,j,:].unsqueeze(1)
		# evaluate current prob
		logp = alp_model.log_predictive_prob(phi_, u_, stats)
		# (1, T)
		log_prgx = moca_model.changepoint_filtering(logp, log_prgx)
		# recursive updated to condition on new data
		# for each run length
		#import ipdb; ipdb.set_trace()
		stats = alp_model.recursive_update(phi_, u_, stats)
		# use log sum exp trick to normalize across run_length dimension
		final_logp = torch.logsumexp(logp + log_prgx, dim=1)
		log_probs.append(final_logp)
	
	# sum over and divide by horizon to get average log prob
	total_logp = torch.stack(log_probs, dim=-1).sum(-1) / horizon
	
	return -total_logp.mean()

def train(iters, env, batch_size, alp_config, moca_config, exp_name, jit, dev):
	train_step = 0
	# create experiment directory and set up Tensorboard logging
	curr_time = str(datetime.datetime.now())
	tb_dir = f'/iris/u/ahmedah/incremental-skill-learning/exps/alpaca_{env}/{exp_name}/'
	if not os.path.exists(tb_dir):
		os.makedirs(tb_dir)
	tb = SummaryWriter(tb_dir + curr_time)

	if env == 'point_mass':
		dataset = LqrExpertDataset(trajlen=500)
		dataloader = DataLoader(dataset, batch_size=500)
	elif env == 'kitchen':
		# state setting is 0 to use all input features
		micro_pth = '/iris/u/ahmedah/custom_kitchen/microwave_skill.npz'
		bottom_pth = '/iris/u/ahmedah/custom_kitchen/bottomknob_skill.npz'
		paths = [micro_pth, bottom_pth]
		dataset = KitchenRobotDataM(paths, trajlen=alp_config['training.data_horizon'], state_set=0)
		dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True, persistent_workers=True)
	# init alpaca and moca models.
	alpaca_model = AlpacaMocaJit(config=alp_config, cuda=0) if jit else AlpacaMoca(config=alp_config, device=dev)
	moca_model = Moca(config=moca_config, device=dev)
	total_params = list(alpaca_model.parameters()) + list(moca_model.parameters())
	optimizer = torch.optim.Adam(total_params, lr=alp_config['training.learning_rate'])
	
	alpaca_model.train()
	moca_model.train()
	data_iter = iter(dataloader)
	with trange(iters) as pbar:
		for idx in pbar:
			# save model params
			if ((idx + 1) % 500) == 0 or idx == 0:
				alp_params = alpaca_model.state_dict()
				moca_params = moca_model.train()
				ckh_path = f'/iris/u/ahmedah/incremental-skill-learning/checkpoints/moca_{env}/{exp_name}/'
				if not os.path.exists(ckh_path):
					os.makedirs(ckh_path)
				torch.save(alp_params, ckh_path + f'alp_{idx}.pt')
				torch.save(moca_params, ckh_path + f'moca_{idx}.pt')
			try:
				sample = next(data_iter)
			except StopIteration:
				# reset data iter
				data_iter = iter(dataloader)
				sample = next(data_iter)
			# init optimizers and model training mode
			optimizer.zero_grad()
			
			# load in current datapoints.
			# 
			x, u = sample['x'].float(), sample['u'].float()
			# add to device
			# (batch_dim, traj_dim, ...)
			x, u = x.to(dev), u.to(dev)

			total_loss = eval_moca(x, u, alpaca_model, moca_model, dev=dev)
			total_loss.backward()

			optimizer.step()

			train_step += 1

			# tf log
			tb.add_scalar("NLL_loss", total_loss.item(), train_step)

			# tqdm log
			logdict = {}
			logdict["tr_loss"] = total_loss.cpu().detach().numpy()
			pbar.set_postfix(logdict)

if __name__ == '__main__':
	# set up argument parser
	parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
	parser.add_argument('-i', '--iter', nargs='?', type=int, default=200)
	parser.add_argument('-n', '--noise', action='store_true')
	parser.add_argument('--env', type=str, default='point_mass')
	parser.add_argument('-b', '--bsize', type=int, default=500)
	parser.add_argument('-sig', type=float, default=2e-2)
	parser.add_argument('-lscale', type=float, default=-1e-2)
	parser.add_argument('-lr', type=float, default=5e-3)
	parser.add_argument('-exp', type=str, default='tmp')
	parser.add_argument('-sh', '--shuffle', action='store_true')
	parser.add_argument('-jit', action='store_true')
	parser.add_argument('-c', '--cuda', type=int, default=0)
	parser.add_argument('-r', '--run_dim', type=int, default=60)
	args = parser.parse_args()
	

	config_kitchen = {
	'env' : 'kitchen',
	'model.x_dim': 60,
	'model.u_dim': 9,
	'model.z_dim': args.run_dim,
	'model.use_input_feats': False,
	'model.hid_dim': 128,
	'model.phi_dim': 16,
	'model.sigep_scale': args.sig,
	'model.init.L_scale': args.lscale,
	'model.glorot_init': False,
	'model.simp_sig_ep': False,
	'training.grad_clip_value': 1,
	'model.phi_activation': nn.ELU(),

	'training.learning_rate': args.lr,
	'training.data_horizon': 60,
	'training.learnable_noise': True,
	'training.learning_rate_decay': False,
	'training.lr_decay_rate': 1e-3,
	'training.sum_act': True
	}

	config_pm = {
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
	'training.grad_clip_value': 1,
	'model.phi_activation': nn.ELU(),

	'training.learning_rate': args.lr,
	'training.data_horizon': 100,
	'training.learnable_noise': True,
	'training.learning_rate_decay': False,
	'training.lr_decay_rate': 1e-3,
	'training.sum_act': True
	}

	moca_config = {
	'x_dim': 60,
	'u_dim': 9,
	'learnable_noise': True,
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
	'max_run_length': args.run_dim,
	'log_params': True,
	'log_nlls': True
	}

	device = torch.device('cpu') if args.cuda == -1 else torch.device(f'cuda:{args.cuda}')
	if args.env == 'point_mass':
		config = config_pm
	elif args.env == 'kitchen':
		config = config_kitchen
	device = torch.device('cpu') if args.cuda == -1 else torch.device(f'cuda:{args.cuda}')
	train(iters=args.iter, env=args.env, batch_size=args.bsize, exp_name=args.exp, jit=args.jit, dev=device, alp_config=config, moca_config=moca_config)