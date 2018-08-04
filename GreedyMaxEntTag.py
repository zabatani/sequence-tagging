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

	tags_map = {}
	for line in file:
		if line == "####META-TAGS-PER-WORD####\n":
			break
		(val, key) = line.split()
		tags_map[key] = val

	file.close()

	output_file = open(output_filename, 'w')
	with open(input_file_name, 'r') as input_file:
		for line in input_file:
			words = line.split()
			tags = []
			for i, word in enumerate(words):
				cols = []
				features = []

				word_feature = "form=" + word
				if word_feature in features_map:
					features.append(word_feature)
				else:
					for j in range(1, 4):
						if len(word) > i:
							features.append("prefix=" + word[:j])
							features.append("suffix=" + word[-j:])

					if any(char.isdigit() for char in word):
						features.append("digit=true")
					if any(char.isupper() for char in word):
						features.append("upper=true")
					if "-" in word:
						features.append("hyphen=true")

				if i >= 1:
					features.append("pw=" + words[i - 1])
					features.append("pt=" + tags[-1])
					if i >= 2:
						features.append("ppw=" + words[i - 2])
						features.append("ppt=" + tags[-2])
					else:
						# TODO: fix this shite
						features.append("ppw=**START**")
						features.append("ppt=**START**" + " " + tags[-1])
				else:
					features.append("pw=**START**")
					features.append("pt=**START**")
					features.append("ppw=**START**")
					features.append("ppt=**START****START**")

				if i < len(words) - 1:
					features.append("nw=" + words[i + 1])
					if i < len(words) - 2:
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
				tags.append(tags_map[str(int(np.asscalar(clf.predict(X_test))))])

			temp = [words[i] + '/' + tags[i] for i in range(len(words))]
			output_file.write(' '.join(map(str, temp)) + "\n")

	output_file.close()


if __name__ == "__main__":
	main(sys.argv)
