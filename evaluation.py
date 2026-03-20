import pandas as pd
import argparse
import numpy as np


def get_args():
	# ground truth header: 'License plate', 'Timestamp', 'First frame', 'Last frame', 'Category'
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='dataset/Output.csv')
	parser.add_argument('--ground_truth_path', type=str, default='dataset/groundTruth.csv')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()
	student_results = pd.read_csv(args.file_path)
	ground_truth = pd.read_csv(args.ground_truth_path)
	totalInput = len(student_results['License plate'])
	totalPlates = len(ground_truth['License plate'])
	# firstFrames = ground_truth['First frame'].tolist()
	# lastFrames = ground_truth['Last frame'].tolist()
	result = np.zeros((totalPlates, 4))
	# 0: TP, 1: FP, 2: LTP

	# Find the last frame and number of plates for each category
	numCategories = len(ground_truth['Category'].unique())
	numPlates = np.zeros(numCategories)
	lastframe = np.zeros(numCategories)
	for i,x in enumerate(ground_truth['Category'].unique()):
		numPlates[i] = len(ground_truth[ground_truth['Category']==x])
		lastframe[i] = ground_truth[ground_truth['Category']==x]['Last frame'].tolist()[-1]
	# For each line in the input list
	for i in range(totalInput):
		licensePlate = student_results['License plate'][i]
		frameNo = student_results['Frame no.'][i]
		timeStamp = student_results['Timestamp(seconds)'][i]
		# Find the lines of solution where frameNo fits into the interval
		interval = ground_truth[(ground_truth['First frame'] <= frameNo) & (ground_truth['Last frame'] >= frameNo)]
		for j in range(len(interval)):
			index = interval.index[j]
			solutionPlate = ground_truth['License plate'][index]
			solutionTimeStamp = ground_truth['Timestamp'][index]
			if licensePlate == solutionPlate:
				if timeStamp <= solutionTimeStamp + 2:
					result[index, 0] += 1
				else:
					result[index, 2] += 1
				if j == 1:
					result[index-1, 1] -= 1
				elif j == 0:
					break
			else:
				result[index, 1] += 1

	# Initialize arrays to save the final results per category
	TP = np.zeros(numCategories)
	FP = np.zeros(numCategories)
	FN = np.zeros(numCategories)
	LTP = np.zeros(numCategories)

	print('---------------------------------------------------------')
	print('%20s'%'License plate', '%10s'%'Result')
	for i in range(totalPlates):
		cat = int(ground_truth['Category'][i]-1)
		if result[i, 0] + result[i, 2] + result[i, 1] == 0:
			finalResult = 'FN'
			FN[cat] += 1
		else:
			if result[i, 0] > 0:
				TP[cat] += 1
				if result[i, 1] == 0:
					finalResult = 'TP'
				else:
					finalResult = 'TP+FP'
					FP[cat] += 1
			elif result[i, 2] > 0:
				LTP [cat] += 1
				if result[i, 1] == 0:
					finalResult = 'LTP'
				else:
					finalResult = 'LTP+FP'
					FP[cat] += 1
			else:
				finalResult = 'FP'
				FP[cat] = FP[cat]+1
		print('%4d'%i,'%14s'%ground_truth['License plate'][i],'%10s'%finalResult)

	output = np.zeros((5, numCategories*2+2))
	for i in range(numCategories):
		output[0, 2*i] = TP[i]
		output[0, 2*i+1] = TP[i]/numPlates[i]*100
		output[1, 2*i] = FP[i]
		output[1, 2*i+1] = 0
		output[2, 2*i] = FN[i]
		output[2, 2*i+1] = FN[i]/numPlates[i]*100
		output[3, 2*i] = LTP[i]
		output[3, 2*i+1] = LTP[i]/numPlates[i]*100
		output[4, 2*i] = (TP[i])/(FP[i]+FN[i]+TP[i])

	output[0, 2*numCategories] = np.sum(TP)
	output[0, 2*numCategories+1] = np.sum(TP)/totalPlates*100
	output[1, 2*numCategories] = np.sum(FP)
	output[1, 2*numCategories+1] = 0
	output[2, 2*numCategories] = np.sum(FN)
	output[2, 2*numCategories+1] = np.sum(FN)/totalPlates*100
	output[3, 2*numCategories] = np.sum(LTP)
	output[3, 2*numCategories+1] = np.sum(LTP)/totalPlates*100
	output[4, 2*numCategories] = (np.sum(TP))/(np.sum(FP)+np.sum(FN)+np.sum(TP))

	true_positives = output[0, :].reshape(-1, 2)
	false_positives = output[1, :].reshape(-1, 2)
	false_negatives = output[2, :].reshape(-1, 2)
	late_true_positives = output[3, :].reshape(-1, 2)
	total_score = output[4, :].reshape(-1, 2)
	TP12 = output[0,0]+output[0,2]#+output[3,0]+output[3,2] (exclude LTP from score)
	FP12 = output[1,0]+output[1,2]
	FN12 = output[2,0]+output[2,2]
	LTP12 = output[3,0]+output[3,2]
	c12score = TP12/(TP12+FP12+FN12)
	c3score = output[4,4]
	c4score = output[4,6]

	col_width = 18  # width for each category column
	label_width = 30  # width for the first column (labels)

	print('********************************************************************')
	print('RESULTS:')
	print(f"{'':{label_width}}", end='')
	for h in ['Category I', 'Category II', 'Category III', 'Category IV', 'Total']:
		print(f"{h:<{col_width}}", end='')
	print()

	def print_row(label, data, do_print_percentage=True):
		label += ': '
		print(f"{label:>{label_width}}", end='')
		for value, percent in data:
			# Remove trailing zeros
			value_str = f"{value:.2f}".rstrip('0').rstrip('.')
			percent_str = f"{percent:.2f}".rstrip('0').rstrip('.') if do_print_percentage else ''
			
			if do_print_percentage:
				s = f"{value_str} ({percent_str}%)"
			else:
				s = f"{value_str}"
			print(f"{s}".ljust(col_width), end='')
		print()
	
	print_row('True positives(TP)', true_positives)
	print_row('False positives(FP)', false_positives, False)
	print_row('False negatives(FN)', false_negatives)
	print_row('Too late true positives(LTP)', late_true_positives)
	print('----------------------------------------------------')
	print_row('Score', total_score, do_print_percentage=False)
	print('%29s'%'Score of Category I & II:', c12score)
	print('%29s'%'Score of Category III:', c3score)
	print('%29s'%'Score of Category IV:', c4score)

	use_old_format = False
	if use_old_format:
		print()
		print()
		print('********************************************************************')
		print('RESULTS:')
		print('%29s'%' ','%14s'%'Category I','%14s'%'Category II','%14s'%'Category III','%14s'%'Category IV','%14s'%'Total')
		print('%29s'%'True positives(TP)', output[0,:])
		print('%29s'%'False positives(FP)', output[1,:])
		print('%29s'%'False negatives(FN)', output[2,:])
		print('%29s'%'Too late true positives(LTP)', output[3,:])
		print('----------------------------------------------------')
		print('%29s'%'Score', output[4,:])

		print('%29s'%'Score of Category I & II:', c12score)