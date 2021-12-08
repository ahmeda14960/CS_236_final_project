import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
import argparse

"""
Helper script to inspect Frank Kitchen demonstrations

Keys are:
['t', 'qp', 'qv', 'obj_qp', 'obj_qv', 'goal', 'obs', 'image', 'image_gripper', 'action', 'reward']
"""
# set up argument parser
parser = argparse.ArgumentParser(description='Process list inputs for hyperparameters in experiments launches')
parser.add_argument('-k', nargs='?', type=int, default=0)
args = parser.parse_args()

path = '/iris/u/ahmedah/kitchen_demos_multitask/friday_microwave_bottomknob_switch_slide/'
curr_npz = 'kitchen_playdata_2019_06_28_12_41_11.npz'
data = np.load(path + curr_npz)

# viz images
k = args.k
f, arrs = plt.subplots(2, 1)
qp = data['qp']
qv = data['qv']
obj_qp = data['obj_qp']
obj_qv = data['obj_qv']
img_s = data['image']
img_g = data['image_gripper']
arrs[0].imshow(img_s[k])
arrs[1].imshow(img_g[k])
plt.savefig('/iris/u/ahmedah/incremental-skill-learning/viz/viz_kitchen.jpg')

# remember to index to i+1 of the switch index so [:i+1]