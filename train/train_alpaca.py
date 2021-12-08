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

from models import AlpacaMH, AlpacaMHJit
from envs import LqrExpertDataset, KitchenRobotData, KitchenRobotDataM


def eval_alp(x, u, z, alp_model, dev):
	"""
	Helper function for evaluating alpaca's NLL loss on a given sample.
	Given some horizon, this model iterates through by evaluting on the current
	timestep on the horizon, conditioning on the data, and then continuing to do so
	for every timestep in the horizon. 

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
	horizon = alp_model.config['training.data_horizon']

	log_probs = []
	# loop over data, predict on current timestep,
	# then condition on observed data

	# batch compute features
	phi = alp_model.encoder(x)
	# grab prior statistics
	stats = alp_model.prior_params()
	stats = [p.to(dev) for p in stats]
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
		z_ = z[:,j,:]
		# evaluate current prob
		logp = alp_model.log_predictive_prob(phi_, u_, z_, stats)
		#TODO(Ahmed) add exception here
		if logp.isnan().any():
			print(f' nan error on iter {j}')
			sys.exit()
		log_probs.append(logp)
		# recursive updated to condition on new data
		stats = alp_model.recursive_update(phi_, u_, z_, stats)
	
	# sum over and divide by horizon to get average log prob
	total_logp = torch.stack(log_probs, dim=-1).sum(-1) / horizon
	total_nll = -total_logp.mean()

	return total_nll

def train(iters, env, batch_size, add_noise, curr_config, shuffle, exp_name, jit, dev, 
			upskill, uplambda):
	if shuffle:
		print('shuffling data points')
	if add_noise:
		print('adding noise')
	curr_time = str(datetime.datetime.now())
	tb_dir = f'/iris/u/ahmedah/incremental-skill-learning/exps/alpaca_{env}/{exp_name}/'
	if not os.path.exists(tb_dir):
		os.makedirs(tb_dir)
	tb = SummaryWriter(tb_dir + curr_time)
	train_step = 0
	alpaca_model = AlpacaMHJit(config=curr_config, cuda=0) if jit else AlpacaMH(config=curr_config, device=dev)
	if env == 'point_mass':
		dataset = LqrExpertDataset(trajlen=500)
		dataloader = DataLoader(dataset, batch_size=500)
	elif env == 'kitchen':
		# state setting is 0 to use all input features
		micro_pth = '/iris/u/ahmedah/custom_kitchen/microwave_skill.npz'
		bottom_pth = '/iris/u/ahmedah/custom_kitchen/bottomknob_skill.npz'
		paths = [micro_pth, bottom_pth]
		dataset = KitchenRobotDataM(paths, trajlen=250, state_set=0, upskill=upskill, uplambda=uplambda)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, persistent_workers=True)


	optimizer = torch.optim.Adam(alpaca_model.parameters(), lr=curr_config['training.learning_rate'])
	data_iter = iter(dataloader)
	with trange(iters) as pbar:
		for idx in pbar:
			# save model params
			if ((idx + 1) % 500) == 0 or idx == 0:
				params = alpaca_model.state_dict()
				ckh_path = f'/iris/u/ahmedah/incremental-skill-learning/checkpoints/alpaca_{env}/{exp_name}/'
				if not os.path.exists(ckh_path):
					os.makedirs(ckh_path)
				torch.save(params, ckh_path + f'{idx}.pt')
			try:
				sample = next(data_iter)
			except StopIteration:
				# reset data iter
				data_iter = iter(dataloader)
				sample = next(data_iter)
			# init optimizers and model training mode
			optimizer.zero_grad()
			alpaca_model.train()

			# load in current datapoints.
			x, u = sample['x'].float(), sample['u'].float()
			# Ensure z is one hot, but no need for one-hot if z_dim == 1
			if alpaca_model.z_dim > 1:
				z = F.one_hot(sample['z'].to(torch.int64), num_classes=alpaca_model.z_dim)
			elif alpaca_model.z_dim == 1:
				#TODO(Ahmed) add exceptions here: 
				z = sample['z'].to(torch.int64).unsqueeze(-1)
				z = torch.ones(z.shape)
				# thrown an error for any zero values, as this will zero out
				# the backprop for single-skill learning
				if torch.numel(z) > torch.count_nonzero(z):
					print('zero entries detected in skill tensor!')
					sys.exit()
			else:
				print('must have at least one skill!')
				sys.exit()
			# add to device
			x, u, z = x.to(dev), u.to(dev), z.to(dev)
			if add_noise:
				if (idx % 2) == 0:
					x = x + torch.randn(x.shape).to(dev)*0.1

			total_loss = eval_alp(x, u, z, alpaca_model, dev=dev)

			# update parameters
			total_nll = total_loss
			total_nll.backward()

			# clip gradients
			nn.utils.clip_grad_norm_(alpaca_model.parameters(), alpaca_model.config['training.grad_clip_value'])
			optimizer.step()

			train_step += 1

			# tf log
			tb.add_scalar("NLL_loss", total_loss.item(), train_step)

			# tqdm log
			logdict = {}
			logdict["tr_loss"] = total_loss.cpu().detach().numpy()
			pbar.set_postfix(logdict)

	return alpaca_model.state_dict()


if __name__ == '__main__':
	# set up argument parser
	parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
	parser.add_argument('-i', '--iter', nargs='?', type=int, default=200)
	parser.add_argument('-n', '--noise', help='adds a small amount of gaussian noise to states', action='store_true')
	parser.add_argument('--env', type=str, help='test env, either kitchen or point_mass', default='point_mass')
	parser.add_argument('-b', '--bsize', help='batch size', type=int, default=500)
	parser.add_argument('-sig', type=float, help='initial scale for Alpaca Sigma', default=2e-2)
	parser.add_argument('-lscale', type=float, help='initial scale for Alpaca lambda', default=-1e-2)
	parser.add_argument('-lr', type=float, help='learning rate', default=5e-3)
	parser.add_argument('-exp', type=str,  help='experiment directory', default='tmp')
	parser.add_argument('-sh', '--shuffle', help='shuffle datapoints', action='store_true')
	parser.add_argument('-jit', help='use jitted models', action='store_true')
	parser.add_argument('-c', '--cuda', help='cuda device, -1 for CPU', type=int, default=0)
	parser.add_argument('-us', '--upskill', help='skills to upsample', nargs='+', default=[])
	parser.add_argument('-ul', '--uplambda', help='number of times to upsample', type=int, default=2)
	args = parser.parse_args()

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

	config_kitchen = {
	'env' : 'kitchen',
	'model.x_dim': 60,
	'model.u_dim': 9,
	'model.z_dim': 2,
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

	if args.env == 'point_mass':
		config = config_pm
	elif args.env == 'kitchen':
		config = config_kitchen

	device = torch.device('cpu') if args.cuda == -1 else torch.device(f'cuda:{args.cuda}')
	alpaca_params = train(iters=args.iter, env=args.env, batch_size=args.bsize, add_noise=args.noise, curr_config=config, shuffle=args.shuffle, exp_name=args.exp, jit=args.jit, 
							dev=device, upskill=args.upskill, uplambda=args.uplambda)
	path = f'/iris/u/ahmedah/incremental-skill-learning/checkpoints/alpaca_{args.env}/{args.exp}/final_{args.iter}.pt'
	torch.save(alpaca_params, path)
	