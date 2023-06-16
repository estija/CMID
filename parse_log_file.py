import argparse
import os
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np


def parse_csv(validation_file, test_file, num_groups, output_file):


	with open(output_file, 'w') as f:
		df = pd.read_csv(validation_file)

		col = df['avg_acc']
		max_index = col.idxmax()

		df2 = pd.read_csv(test_file)

		count = 5

		# try with avg_acc, avg_actual_loss, avg_per_sample_loss
		previous = 0
		ending_epoch = 0
		for i in range(len(df['avg_acc'])):
			current = (df['avg_acc'][int(i)])
			if (count % 5 == 0):
				if (previous > current):
					ending_epoch = i 
					break
				previous = current
			count+=1



		f.write('Early stopping validation accuracy: ' + str(df['avg_acc'][ending_epoch]) + '\n')
		f.write('Epoch: ' + str(df['epoch'][ending_epoch]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df[header][ending_epoch]) + '\n')



		f.write('\nBest validation accuracy: ' + str(df['avg_acc'][max_index]) + '\n')
		f.write('Epoch: ' + str(df['epoch'][max_index]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df[header][max_index]) + '\n')


		f.write('\nEarly stopping testing accuracy: ' + str(df2['avg_acc'][ending_epoch]) + '\n')
		f.write('Epoch: ' + str(df2['epoch'][ending_epoch]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df2[header][ending_epoch]) + '\n')

		f.write('\nTest accuracy at best validation: ' + str(df2['avg_acc'][max_index]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df2[header][max_index]) + '\n')

		col = df2['avg_acc']
		max_index = col.idxmax()
		f.write('\nBest test accuracy: ' + str(df2['avg_acc'][max_index]) + '\n')
		f.write('Epoch: ' + str(df2['epoch'][max_index]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df2[header][max_index]) + '\n')


		if num_groups==4:
			col = df[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3']].min(axis=1)
		elif num_groups==6:
			col = df[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3','avg_acc_group:4','avg_acc_group:5']].min(axis=1)
		elif num_groups==16:
			col = df[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3',
                      'avg_acc_group:4','avg_acc_group:5','avg_acc_group:6','avg_acc_group:7',
                      'avg_acc_group:8','avg_acc_group:9','avg_acc_group:10','avg_acc_group:11',
                      'avg_acc_group:12','avg_acc_group:13','avg_acc_group:14','avg_acc_group:15']].min(axis=1)
		max_index = col.idxmax()
        #max_index = col.idxmax()
		f.write('\nBest worst-case group validation accuracy: ' + str(df['avg_acc'][max_index]) + '\n')
		f.write('Epoch: ' + str(df['epoch'][max_index]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df[header][max_index]) + '\n')

		if num_groups==4:
			col = df2[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3']].min(axis=1)
		elif num_groups==6:
			col = df2[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3','avg_acc_group:4','avg_acc_group:5']].min(axis=1)
		elif num_groups==16:
			col = df2[['avg_acc_group:0','avg_acc_group:1','avg_acc_group:2','avg_acc_group:3',
                      'avg_acc_group:4','avg_acc_group:5','avg_acc_group:6','avg_acc_group:7',
                      'avg_acc_group:8','avg_acc_group:9','avg_acc_group:10','avg_acc_group:11',
                      'avg_acc_group:12','avg_acc_group:13','avg_acc_group:14','avg_acc_group:15']].min(axis=1)
		max_index = col.idxmax()
		f.write('\nBest worst-case group test accuracy: ' + str(df2['avg_acc'][max_index]) + '\n')
		f.write('Epoch: ' + str(df2['epoch'][max_index]) + '\n')
		f.write('Group accuracies:\n')
		for i in range(0, num_groups):
			header = 'avg_acc_group:' + str(i)
			f.write(header + ": " + str(df2[header][max_index]) + '\n')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, default='./log')
	parser.add_argument('--num_groups', type=int, default=4)
	parser.add_argument('--val', type=str, default='val.csv')
	parser.add_argument('--test', type=str, default='test.csv')


	args = parser.parse_args()
	if args.val=='val.csv':
		outf='parsed_log_file.txt'
	else:
		outf='parsed_log_file_r.txt'

	validation_file = os.path.join(args.log_dir, args.val)
	test_file = os.path.join(args.log_dir, args.test)
	output_file = os.path.join(args.log_dir, outf)
	parse_csv(validation_file, test_file, args.num_groups, output_file)



	#parse_file(args.log_file, args.num_groups, args.output_file)


if __name__ == '__main__':
	main()