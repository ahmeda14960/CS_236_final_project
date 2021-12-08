import numpy as np
from torch.utils.data import Dataset
import torch
import h5py

class KitchenRobotData(Dataset):
    """
    Dataset class for learning one skill in the Franka
    Kitchen environment.
    """
    def __init__(self, path, trajlen=29, ext='npz', fixed=False, fix_skill=None, state_set=0):
        if ext == 'npz':
            data = np.load(path)
            if state_set == 0:
                self.X = np.concatenate([data['qp'], data['qv'], data['obj_qp'], data['obj_qv']], axis=1)
            elif state_set == 1:
                self.X = np.concatenate([data['qp'], data['qv'], data['obj_qp']], axis=1)
            elif state_set == 2:
                self.X = np.concatenate([data['qp'], data['obj_qp']], axis=1)
            elif state_set == 3:
                self.X = np.concatenate([data['qp'], data['qv'], data['obj_qv']], axis=1)
            self.U = data['action']
        elif ext == 'hdf5':
            with h5py.File(path, "r") as f:
                obs, acts = list(f['observations']), list(f['actions'])
                self.X = np.asarray(obs)
                self.U = np.asarray(acts)      
        self.Z = np.ones(self.U.shape[0])
        if fixed:
            self.Z[0:70] = self.Z[0:70]*0
            self.Z[70:120] = self.Z[70:120]
            self.Z[120:180] = self.Z[120:180]*2
            self.Z[180:] = self.Z[180:]*3
        if fix_skill == 0:
            self.X = self.X[0:70]
            self.U = self.U[0:70]
            self.Z = self.Z[0:70]
        self.trajlen = trajlen
        print('traj', self.trajlen)
        print('X', self.X.shape)
        print('U', self.U.shape)
        print('Z', self.Z.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        # need to make this index around to avoid batching issues with torch
        if idx+self.trajlen > self.X.shape[0]:
            sample = {
                'x': self.X[-self.trajlen:,:],
                'u': self.U[-self.trajlen:,:],
                'z': self.Z[-self.trajlen:]
            }
        else:    
            sample = {
                'x': self.X[idx:idx+self.trajlen,:],
                'u': self.U[idx:idx+self.trajlen,:],
                'z': self.Z[idx:idx+self.trajlen]
            }

        return sample

class KitchenRobotDataM(Dataset):
    """
    Dataset class for multi-task learning on
    the Franka Kitchen environment. Currently only
    supports npz.
    Args:
        trajlen: int specifying the length of each trajectory
        state_set: int in [0:4], specifiying which state features to learn
        from
            0: learn from all state features
            1: ignore object velocity
            2: ignore arm and object velocity
            3: ignore object position
        upskill: list containing int of which skill we wish to upsample.
        Useful when training with datasets that have a small number of a certain 
        skill's transition
        uplambda: number of times to upsample skills
    """
    def __init__(self, paths, trajlen=100, state_set=0, upskill=[], uplambda=2):
        # for each path, grab skill data and add to running total.
        # Note that the order of the paths determines the skill id.
        self.X = []
        self.U = []
        self.Z = []
        # Also track indices of start and end of each skill
        # so we can retrieve skill-specific trajectories for
        # test-time conditioning of ALPaCA
        self.skill_switches = {}
        # Track number of transitions per skill
        self.skill_n = {}
        skill_start = 0
        skill_end = 0
        for idx, path in enumerate(paths):
            data = np.load(path)
            self.skill_n[idx] = 0
            # if skill is upsampled add more data.
            i = uplambda if str(idx) in upskill else 1
            for _ in range(i):
                if state_set == 0:
                    self.X.append(np.concatenate([data['qp'], data['qv'], data['obj_qp'], data['obj_qv']], axis=1))
                elif state_set == 1:
                    self.X.append(np.concatenate([data['qp'], data['qv'], data['obj_qp']], axis=1))
                elif state_set == 2:
                    self.X.append(np.concatenate([data['qp'], data['obj_qp']], axis=1))
                elif state_set == 3:
                    self.X.append(np.concatenate([data['qp'], data['qv'], data['obj_qv']], axis=1))
                self.U.append(data['action'])
                curr_skill = np.ones(data['action'].shape[0])*idx
                self.skill_n[idx] += curr_skill.shape[0]
                self.Z.append(curr_skill)
            
            if idx == 0:
                # only need to set end for first skill
                skill_end = data['action'].shape[0] - 1
            elif idx == 0 and idx in upskill:
                # upsample first skill
                skill_end = data['action'].shape[0]*uplambda - 1
            elif idx in upskill:
                # in general case, start is end of last skill plus 1
                skill_start = skill_end + 1
                # end is the batch size from the start, times upsample factor
                skill_end = skill_start + data['action'].shape[0]*uplambda - 1
            else:
                # in general case, start is end of last skill plus 1
                skill_start = skill_end + 1
                # end is the batch size from the start
                skill_end = skill_start + data['action'].shape[0] - 1
            # update skill switch library
            self.skill_switches[f'skill_{idx}'] = (skill_start, skill_end)
            
        self.X = np.concatenate(self.X)
        self.U = np.concatenate(self.U)
        self.Z = np.concatenate(self.Z)

        self.trajlen = trajlen
        print('traj', self.trajlen)
        print('X', self.X.shape)
        print('U', self.U.shape)
        print('Z', self.Z.shape)
        print('num skills', len(paths))
        for key, value in self.skill_n.items():
            print(f'skill {key} has {value} samples')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        # need to make this index around to avoid batching issues with torch
        if idx+self.trajlen > self.X.shape[0]:
            sample = {
                'x': self.X[-self.trajlen:,:],
                'u': self.U[-self.trajlen:,:],
                'z': self.Z[-self.trajlen:]
            }
        else:    
            sample = {
                'x': self.X[idx:idx+self.trajlen,:],
                'u': self.U[idx:idx+self.trajlen,:],
                'z': self.Z[idx:idx+self.trajlen]
            }

        return sample

    def get_skill(self, z):
        """
        Helper function to grab skill specific trajectories
        for test-time inference of alpaca
        """
        start, end = self.skill_switches[f'skill_{z}']
        invalid_traj = True

        # until we get a trajectory within the bounds of the
        # current skill, keep sampling
        while invalid_traj:
            idx = np.random.randint(start, end+1)
            if idx+self.trajlen < end:
                invalid_traj = False
        
        sample = {
                'x': self.X[idx:idx+self.trajlen,:],
                'u': self.U[idx:idx+self.trajlen,:],
                'z': self.Z[idx:idx+self.trajlen]
            }
        return sample