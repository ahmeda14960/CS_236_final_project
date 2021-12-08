import numpy as np
from torch.utils.data import Dataset

# returns termination class for a given current and past skill pair
# z_t is past skill, z_t_1 is next skill
# doesn't work
def select_b(z_t, z_t_1):
	if z_t != z_t_1:
		return 1
	else:
		return 0
class LqrExpertDataset(Dataset):
	def __init__(self, p_still=0.0, datalen=1e6, trajlen=250, goal_scale=10):
		self.goal_states = [
			np.array([[1, 1, 0, 0]]).T, # blue
			np.array([[1, -1, 0, 0]]).T, # green
			np.array([[-1, -1, 0, 0]]).T, # magenta
			np.array([[-1, 1, 0, 0]]).T # red
		]

		# scale the size of the square
		self.goal_states = [goal_scale*arr for arr in self.goal_states]
		self.num_skills = 4
		self.threshold = 0.5
		self.datalen = int(datalen)
		print('datalen', self.datalen)
		self.trajlen = trajlen
		
		self.verbose = False
		self.p_still = p_still
		self.x_dim = 4
		self.u_dim = 2

		self.shuffle = False
		
		# reward function
		self.Q = np.diag([1,1,0.1,0.1])
		self.R = np.diag([0.01,0.01])
		
		# dynamics
		dt = 0.1
		self.A = np.eye(4)
		self.A[0,2] = dt
		self.A[1,3] = dt
		
		self.B = np.zeros((4,2))
		self.B[2,0] = dt
		self.B[3,1] = dt
		
		self.stdev = np.array([0.02,0.02,0.01,0.01])
		
		self.compute_expert_policies()
		self.generate_dataset()
		
	def __len__(self):
		return int(self.datalen - self.trajlen)
	
	def dynamics(self,x,u):
		xp = self.A @ x + self.B @ u + (self.stdev * np.random.randn(4))[:,None]
		return xp
	
	def terminations(self,x,z, still):
		if still:
			#print('still')
			return z
		gs = self.goal_states[z]
		dist = np.linalg.norm(x - gs)
		if self.verbose:
			print('distance from goal', dist)
			print('goal state', gs)
		if dist < self.threshold:
			# if close to goal, randomly sample next skill
			z = np.random.randint(0, high=4)
			# Do NOT try incrementing to next goal.
			# this will lead to poor training, as the model
			# will learn a specific sequence of skills
			# and will not learn to generalize such that it
			# can reach any skill from any skill.
			#z = (z + 1) % 4
		return z 
	
	def compute_expert_policies(self):
		self.K = np.empty((self.num_skills,2,4))
		self.k = np.empty((self.num_skills,2,1))
		for i, gs in enumerate(self.goal_states):
			q = (-gs.T @ self.Q).T


			T = 50
			P = self.Q.copy()
			p = q.copy()
			for t in range(T):
				Su = (p.T @ self.B).T
				Suu = self.R + self.B.T @ P @ self.B
				Sux = self.B.T @ P @ self.A 
				
				self.K[i] = - np.linalg.inv(Suu) @ Sux
				self.k[i] = - np.linalg.inv(Suu) @ Su
				
				P = self.Q + self.A.T @ P @ self.A - self.K[i].T @ Suu @ self.K[i]
				p = q + self.A.T @ p + Sux.T @ self.k[i]
#                 print(self.k[i])
#                 print(self.K[i])
			
	def generate_dataset(self):
		x = (np.random.randn(4)*0.5)[:,None]
		z = np.random.choice(4)
		# prob agent stays at the same goal
		p_still = self.p_still
		self.X = np.empty((self.datalen, self.x_dim))
		self.Z = np.empty(self.datalen)
		self.H = np.empty(self.datalen)
		self.U = np.empty((self.datalen, self.u_dim))
		self.T = np.empty(self.datalen)
		still = False
		run_start = 0
		run_end = 0
		for t in range(self.datalen):
			self.X[t] = x.T
			self.Z[t] = z
			u = self.K[z] @ x + self.k[z]
			self.U[t] = u.T
			xp = self.dynamics(x,u)
			if t % 100 == 0:
				samp = np.random.uniform()
				if samp < p_still:
					still = True
				else:
					still = False
			z = self.terminations(x,z, still)
			self.H[t] = select_b(self.Z[t], z)
			# if we switch skills, calculate 
			# time until skill switch from 0 to 1
			if self.Z[t - 1] != self.Z[t]:
				if run_end == run_start:
					self.T[run_end] = 1
					run_start = run_end + 1
					run_end = run_start + 1
				elif run_end - run_start == 1:
					self.T[run_start:run_end] = 1
					run_start = run_end
					run_end = run_start + 1
				else: 
					phase = np.linspace(0, 1, num=run_end - run_start)
					self.T[run_start:run_end] = phase
					run_start = run_end
					run_end = run_end + 1
			# if we don't switch and data generation terminates
			# estimate time until next switch
			elif t == self.datalen - 1:
				new_z = (z + 1) % 4
				new_goal = self.goal_states[new_z]
				new_dist = np.linalg.norm(x - new_goal)

				gs = self.goal_states[z]
				dist = np.linalg.norm(x - gs)
				# ratio of how close we are to switching
				ratio = dist / new_dist
				self.T[run_start:run_end] = np.linspace(0, 1*ratio, num=run_end - run_start)
			else:
				run_end += 1
			x = xp.copy()
		# upsample the transition states
		# use repeat instead
		trans_idx_pre = [idx for idx, el in enumerate(self.H) if el == 1]
		# for each index where a transition occurs
		# pad the next and last 3 indicies as transitions
		# the min is for cases when there aren't 3 surronding
		# indicies, such as the start of a trajectory
		for i in trans_idx_pre:
			self.H[i-3:i+4] = np.ones(min(7, len(self.H[i-3:i+4])))
		
		print('trans len pre %d' % len(trans_idx_pre))
		# recollect new transitions states for more upsampling
		trans_idx = [idx for idx, el in enumerate(self.H) if el == 1]
		print('trans len post %d' % len(trans_idx))
		new_X = self.X[trans_idx]
		new_Z = self.Z[trans_idx]
		new_U = self.U[trans_idx]
		new_H = self.H[trans_idx]
		new_T = self.T[trans_idx]
		print('old X len', self.X.shape)
		print('transition len', new_X.shape)

		# upsample states, skills, and whether we switched or not
		self.X_bin = np.empty_like(self.X)
		self.Z_bin = np.empty_like(self.Z)
		self.H_bin = np.empty_like(self.H)

		self.X_bin[:] = self.X 
		self.Z_bin[:] = self.Z
		self.H_bin[:] = self.H
		up_iters = 0
		for _ in range(up_iters):
			self.X_bin = np.concatenate([self.X_bin, new_X])
			self.Z_bin = np.concatenate([self.Z_bin, new_Z])
			self.H_bin = np.concatenate([self.H_bin, new_H])
		print('new X len', self.X_bin.shape)
		

		if self.shuffle:
			shuffler = np.random.permutation(len(self.X))
			self.X = self.X[shuffler]
			self.H = self.H[shuffler]
			self.U = self.U[shuffler]
			self.Z = self.Z[shuffler]
			self.T = self.T[shuffler]

		self.Switch = self.H > 3
		self.Switch = self.Switch.astype(int)
	def __getitem__(self,idx):    
		
		sample = {
			'x': self.X[idx:idx+self.trajlen,:],
			'u': self.U[idx:idx+self.trajlen,:],
			'z': self.Z[idx:idx+self.trajlen],
			'b': self.H[idx:idx+self.trajlen],
			'x_bin': self.X_bin[idx:idx+self.trajlen],
			'z_bin': self.Z_bin[idx:idx+self.trajlen],
			'h_bin': self.H_bin[idx:idx+self.trajlen]
		}
		
		return sample