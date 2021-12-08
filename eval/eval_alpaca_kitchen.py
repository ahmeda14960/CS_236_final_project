import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gym
import torch.nn as nn 
import argparse
import datetime
import sys, os
import random
import time
from moviepy.editor import ImageSequenceClip

from models import AlpacaMH, AlpacaMHJit
from envs import KitchenRobotData, KitchenRobotDataM
from utils import VideoMaker
import adept_envs

"""
Helper script to inspect Frank Kitchen demonstrations

Keys are:
['t', 'qp', 'qv', 'obj_qp', 'obj_qv', 'goal', 'obs', 'image', 'image_gripper', 'action', 'reward']
"""

def eval_alp_kitchen(alp_model, env, stats, iters, init_ob, device):
	"""
	Helper function to visualize roll-outs from the FrankaKitchen Env for ALPaCA
	Args:
		alp_model: AlpacaMH or AlpacaMHJit model to be used for test inference
		env: FrankaKitchen Env to be used.
		stats: tuple of ALPaCA statistics
					Linv: block precision tensor (batch_dim, z_dim, u_dim, phi_dim, phi_dim)
					Q: TODO(Ahmed) find a good name for this tensor (batch_dim, z_dim, u_dim, 1, phi_dim)
		iters (int) : number of timesteps to roll-out policy
		init_ob: np array, initial observation from resetting Franka Env
		device: Torch device object, indicating CPU/GPU
	"""
	# import ipdb; ipdb.set_trace()
	obs = init_ob
	frames = []
	z = 0
	# track total rewards
	rewards = []
	for i in range(iters):
		# render
		img = env.render(mode='rgb_array')
		# resize to save space
		img = cv2.resize(img, dsize=(240, 300))
		frames.append(img)

		x = torch.tensor(obs).float().unsqueeze(0).to(device)
		# grab ALPaCA posterior predictive mean and cov
		if alp_model.z_dim == 1:
			# squeeze to remove batch and skill dim
			mu, sigma = [p.squeeze(0).squeeze(0) for p in alp_model.forward(x, stats)]
			mu = mu.cpu().detach().numpy() if device.type == 'cuda' else mu.detach().numpy()
			cov = sigma.cpu().detach().numpy() if device.type == 'cuda' else sigma.detach().numpy()
		else:
			if (i % (iters/alp_model.z_dim)) == 0 and i != 0:
				z += 1
				print('incrementing to next skill')
			#import ipdb; ipdb.set_trace()
			# squeeze to remove batch and skill dim
			mu, sigma = [p.squeeze(0).squeeze(0) for p in alp_model.forward(x, stats)]
			mu = mu[z].cpu().detach().numpy() if device.type == 'cuda' else mu[z].detach().numpy()
			cov = sigma[z].cpu().detach().numpy() if device.type == 'cuda' else sigma[z].detach().numpy()
		
		# sample action
		u = np.expand_dims(np.random.multivariate_normal(np.squeeze(mu), np.diag(cov)), axis=1)
		u = u.squeeze(-1)
		# should just be (u_dim,)
		obs, reward, done, info = env.step(u)
		rewards.append(reward)
	rewards = np.asarray(rewards)
	total_reward = rewards.sum()
	return frames, total_reward




if __name__ == '__main__':
	start_time = time.time()
	# set up argument parser
	parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
	parser.add_argument('-chk', '--checkpoint', type=int, default=200)
	parser.add_argument('-jit', action='store_true')
	parser.add_argument('-sig', type=float, default=2e-2)
	parser.add_argument('-lscale', type=float, default=-1e-2)
	parser.add_argument('-lr', type=float, default=5e-3)
	parser.add_argument('--exp', type=str, default='tmp')
	parser.add_argument('-i', type=int, default=300)
	parser.add_argument('-c', '--cuda', type=int, default=-1)
	parser.add_argument('--ext', type=str, default='gif')
	parser.add_argument('-ns', '--num_samps', type=int, default=5)
	args = parser.parse_args()
	# using CUDA or no
	device = torch.device('cpu') if args.cuda == -1 else torch.device(f'cuda:{args.cuda}')

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
	# init franka env
	env = gym.make('kitchen_relax-v1')
	#TODO(Ahmed) inspect these two to make sure no weird inputs are passed in.
	obs = env.reset()
	# do this to avoid concating empty goal.
	# TODO(Ahmed) clean robot env and remove.
	env.goal_concat = False
	# init alpaca and load checkpoint parameters
	alpaca_model = AlpacaMHJit(device=device, config=config_kitchen) if args.jit else AlpacaMH(device=device, config=config_kitchen)
	alpaca_model.to(device)

	alpaca_model.load_state_dict(torch.load(f'/iris/u/ahmedah/incremental-skill-learning/checkpoints/alpaca_kitchen/{args.exp}/{args.checkpoint}.pt'))

	# make directory for experiment to save videos
	exp_dir = f'/iris/u/ahmedah/incremental-skill-learning/viz/alpaca_kitchen/{args.exp}/'
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	
	# create video maker for saving GIF/MP4
	# video_maker = VideoMaker(write_path=exp_dir + f'{args.checkpoint}.{args.ext}',read_path=None)
	# reset ALPaCA params, condition for skill adaptation on some data
	cpth = '/iris/u/ahmedah/custom_kitchen/microwave_skill.npz'
	micro_pth = '/iris/u/ahmedah/custom_kitchen/microwave_skill.npz'
	bottom_pth = '/iris/u/ahmedah/custom_kitchen/bottomknob_skill.npz'
	paths = [micro_pth, bottom_pth]
	dataset = KitchenRobotDataM(paths, trajlen=250, state_set=0)
	# sample random trajectory for each skill
	samples = []
	for _ in range(args.num_samps):
		# N trajectories for each skill
		for idx in range(alpaca_model.z_dim):
			sample = dataset.get_skill(idx)
			# sanity check to make sure skills 
			# in sample corespond 
			out = np.unique(sample['z'])
			print(f'sample {idx}', out)
			samples.append(sample)
	stats = alpaca_model.prior_params()

	# condition on skill based trajectories
	for sample in samples:
		for t in range(dataset.trajlen):
			# add unsqueeze to include dummy batch dimension
			x = torch.tensor(sample['x'][t]).float().unsqueeze(0)
			u = torch.tensor(sample['u'][t]).float().unsqueeze(0).unsqueeze(1)
			# Ensure z is one hot, but no need for one-hot if z_dim == 1
			if alpaca_model.z_dim > 1:
				z = F.one_hot(torch.tensor(sample['z'][t]).to(torch.int64), num_classes=alpaca_model.z_dim)
			elif alpaca_model.z_dim == 1:
				#TODO(Ahmed) add exceptions here: 
				z = torch.tensor(sample['z'][t]).to(torch.int64).unsqueeze(-1)
				# thrown an error for any zero values, as this will zero out
				# the backprop for single-skill learning
				if torch.numel(z) > torch.count_nonzero(z):
					print('zero entries detected in skill tensor!')
					sys.exit()
			else:
				print('must have at least one skill!')
				sys.exit()
			# move to cuda or remain on CPU
			x, u, z = x.to(device), u.to(device), z.to(device)
			phi = alpaca_model.encoder(x).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
			stats = alpaca_model.recursive_update(phi, u, z, stats)
	curr_frames, total_reward = eval_alp_kitchen(alp_model=alpaca_model, env=env, stats=stats, iters=args.i, init_ob=obs, device=device)
	clip = ImageSequenceClip(curr_frames, fps=30)
	clip.write_gif(exp_dir + f'{args.i}_{total_reward}.gif')
	# video_maker.write_to_file(curr_frames)
	# video_maker.close_writer()
	print(f'--- executes in {time.time() - start_time} seconds')
