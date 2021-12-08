import numpy as np
import csv
from os.path import exists
import matplotlib.pyplot as plt
from PIL import Image

# COMMAND TO PRINT NPZ FILES WITH ONE FILE PER LINE: ls *.npz -a | cat
"""
Helper script to collate Frank Kitchen demonstrations in a given folder

Keys are:
['t', 'qp', 'qv', 'obj_qp', 'obj_qv', 'goal', 'obs', 'image', 'image_gripper', 'action', 'reward']
"""
multi = 'friday_microwave_bottomknob_switch_slide'
folder_pth = f'/iris/u/ahmedah/kitchen_demos_multitask/{multi}/'
curr_skill = 'bottomknob'

csv_pth = f'/iris/u/ahmedah/custom_kitchen/{multi}.csv'
skill_npz = '{}_skill.npz'.format(curr_skill)

qp_arr, qv_arr, obj_qp_arr, obj_qv_arr, action_arr = [], [], [], [], []
# init arrays with the data from prior folders
np_dir = '/iris/u/ahmedah/custom_kitchen/%s' % skill_npz
if exists(np_dir):
    old_data = np.load(np_dir)
    qp_arr.append(old_data['qp'])
    qv_arr.append(old_data['qv'])
    obj_qp_arr.append(old_data['obj_qp'])
    obj_qv_arr.append(old_data['obj_qv'])
    action_arr.append(old_data['action'])
else:
    print('Numpy file not found, saving new file for %s' % curr_skill)


with open(csv_pth, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # print fields for sanity check
    fields = next(csvreader)
    print(fields)
    # for each npy file, append slice relevant to given skill to dataset
    for i, row in enumerate(csvreader):
        npz_pth = row[0]
        if curr_skill == 'microwave':
            start = int(row[1])
            end = int(row[2])
        elif curr_skill == 'bottomknob':
            start = int(row[3])
            end = int(row[4])
        data = np.load(folder_pth + npz_pth)
        qp_arr.append(data['qp'][start:end+1])
        qv_arr.append(data['qv'][start:end+1])
        obj_qp_arr.append(data['obj_qp'][start:end+1])
        obj_qv_arr.append(data['obj_qv'][start:end+1])
        action_arr.append(data['action'][start:end+1])

    qp = np.concatenate(qp_arr)
    qv = np.concatenate(qv_arr)
    obj_qp = np.concatenate(obj_qp_arr)
    obj_qv = np.concatenate(obj_qv_arr)
    action = np.concatenate(action_arr)

    # sanity check sample size
    print('length is %d' % qp.shape[0])
    np.savez('/iris/u/ahmedah/custom_kitchen/{}'.format(skill_npz), qp=qp, qv=qv, obj_qp=obj_qp, obj_qv=obj_qv, action=action)