import argparse
from pygifsicle import optimize


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process directory for gif compress')
	parser.add_argument('-i', '--iter', type=int, default=499)
	args = parser.parse_args()
	curr_name = args.iter
	optimize(f'/iris/u/ahmedah/incremental-skill-learning/viz/alpaca_kitchen/no_shuffle_no_noise/{curr_name}.gif')
