import sys
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
import numpy as np


def main(argv):
	input_file_name = argv[1]
	model_filename = argv[2]
	feature_map_file = argv[3]
	output_filename = argv[4]

	clf = joblib.load(model_filename)

	features_map = {}

	file = open(feature_map_file, 'r')
	for line in file:
		if line == "####META-TAGS-MAP####\n":
			break
		(key, val) = line.split()
		features_map[key] = val

	tags_map = []
	for line in file:
		if line == "####META-TAGS-PER-WORD####\n":
			break
		tags_map.append(line.split()[0])

	tags_per_word = {}
	for line in file:
		line = line.split()
		word = line[0]
		tags = [int(tag) for tag in line[1:]]
		tags_per_word[word] = tags

	file.close()

	output_file = open(output_filename, 'w')
	with open(input_file_name, 'r') as input_file:
		for i, line in enumerate(input_file):
			words = line.split()
			tags = viterbi(words, tags_per_word, features_map, tags_map, clf)
			temp = [words[i] + '/' + tags[i] for i in range(len(words))]
			output_file.write(' '.join(map(str, temp)) + "\n")

	output_file.close()


def viterbi(words, tags_per_word, features_map, tags_map, clf):
	tags_len = len(tags_map)
	words_len = len(words)
	V = np.zeros([len(words), tags_len, tags_len]) - np.inf
	bp = np.zeros([len(words), tags_len, tags_len])
	tags_range = range(tags_len)
	for i in range(words_len):
		if i >= 1:
			if words[i - 1] in tags_per_word:
				t_range = tags_per_word[words[i-1]]
			else:
				t_range = tags_range
		else:
			t_range = [0]
		for t in t_range:
			if words[i] in tags_per_word:
				r_range = tags_per_word[words[i]]
			else:
				r_range = tags_range

			def scorer(t_T): return scorer_aux(words, i, features_map, tags_map, clf, t, t_T)

			if i >= 2:
				if words[i - 2] in tags_per_word:
					t_T_range = tags_per_word[words[i - 2]]
				else:
					t_T_range = tags_range
			else:
				t_T_range = [0]

			scores = [(np.log(scorer(t_T)), t_T) for t_T in t_T_range]
			for r in r_range:
				if i == 0:
					temp = np.array([score[0][r] for score in scores])
				else:
					temp = np.array([V[i - 1, score[1], t] + score[0][r] for score in scores])
				V[i, t, r] = temp.max()
				bp[i, t, r] = scores[temp.argmax()][1]
	tags = np.empty(len(words), dtype=object)
	if words_len >= 2:
		tags[-2:] = np.unravel_index(V[words_len-1,:,:].argmax(), V[words_len-1,:,:].shape)
		for i in range(words_len-3,-1,-1):
			tags[i] = int(bp[i+2, tags[i+1], tags[i+2]])
	else:
		tags[0] = np.unravel_index(V[words_len - 1, 0, :].argmax(), V[words_len - 1, 0, :].shape)[0]
	tags = [tags_map[tag] for tag in tags]
	return tags


def scorer_aux(words, i, features_map, tags_map, clf, t, t_T):
	words_len = len(words)
	cols = []
	features = []

	word_feature = "form=" + words[i]
	if word_feature in features_map:
		features.append(word_feature)
	else:
		for j in range(1, 4):
			if len(words[i]) > i:
				features.append("prefix=" + words[i][:j])
				features.append("suffix=" + words[i][-j:])

		if any(char.isdigit() for char in words[i]):
			features.append("digit=true")
		if any(char.isupper() for char in words[i]):
			features.append("upper=true")
		if "-" in words[i]:
			features.append("hyphen=true")

	if i >= 1:
		features.append("pw=" + words[i - 1])
		features.append("pt=" + tags_map[t])
		if i >= 2:
			features.append("ppw=" + words[i - 2])
			features.append("ppt=" + tags_map[t_T])
		else:
			# TODO: fix this shite
			features.append("ppw=**START**")
			features.append("ppt=**START**" + " " + tags_map[t])
	else:
		features.append("pw=**START**")
		features.append("pt=**START**")
		features.append("ppw=**START**")
		features.append("ppt=**START****START**")

	if i < words_len - 1:
		features.append("nw=" + words[i + 1])
		if i < words_len - 2:
			features.append("nnw=" + words[i + 2])
		else:
			features.append("nnw=**END**")
	else:
		features.append("nw=**END**")
		features.append("nnw=**END**")

	for feature in features:
		if feature in features_map:
			cols.append(features_map[feature])

	cols = np.array(cols)
	rows = np.zeros(cols.shape)
	data = np.ones(cols.shape)
	X_test = csr_matrix((data, (rows, cols)), shape=(1, len(features_map)))

	return clf.predict_proba(X_test)[0]


if __name__ == "__main__":
	main(sys.argv)
