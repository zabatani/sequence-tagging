import sys
import os


def main(argv):
	truth = "../data/ass1-tagger-test"
	predicted = "viterbi_out"

	correct = 0
	total = 0
	with open(truth, 'r') as truth_f, open(predicted, 'r') as predicted_f:
		for truth_line, predicted_line in zip(truth_f, predicted_f):
			truth_line = truth_line.split()
			predicted_line = predicted_line.split()
			for truth_line, predicted_line in zip(truth_f, predicted_f):
				truth_tags = [pair.rsplit('/', 1)[1] for pair in truth_line.split()]
				predicted_tags = [pair.rsplit('/', 1)[1] for pair in predicted_line.split()]
				for truth_tag, predicted_tag in zip(truth_tags, predicted_tags):
					if truth_tag==predicted_tag:
						correct+=1
					total+=1

	print(str((correct/total)*100))


if __name__ == "__main__":
	main(sys.argv)
