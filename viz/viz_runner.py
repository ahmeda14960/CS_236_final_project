import argparse
import sys
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, default='tmp')
parser.add_argument('--chk', type=int, default=99999)
args = parser.parse_args()
for i in range(args.chk):
	if ((i+1) % 500) == 0:
		command = f'python /iris/u/ahmedah/incremental-skill-learning/viz/viz_kitchen.py --env kitchen -chk {i} --exp {args.exp}'.split()
		subprocess.run(command)
