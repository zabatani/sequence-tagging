import sys
from collections import defaultdict


def main(argv):
	corpus_filename = argv[1]
	features_filename = argv[2]

	# Rare words handler
	rare_words = defaultdict(int)
	with open(corpus_filename, 'r') as corpus_file:
		for line in corpus_file:
			words = [pair.rsplit('/', 1)[0] for pair in line.split()]
			for word in words:
				rare_words[word] += 1
	rare_words_threshold = 5

	# Create features
	features_file = open(features_filename, 'w')
	with open(corpus_filename, 'r') as corpus_file:
		for line in corpus_file:
			pairs = [pair.rsplit('/', 1) for pair in line.split()]
			for i, pair in enumerate(pairs):
				features = []
				word = pair[0]
				tag = pair[1]

				# Rare word check
				if rare_words[word] >= rare_words_threshold:
					features.append("form=" + word)

				else:
					for j in range(1, 4):
						if len(word) > j:
							features.append("prefix=" + word[:j])
							features.append("suffix=" + word[-j:])

					if any(char.isdigit() for char in word):
						features.append("digit=true")
					if any(char.isupper() for char in word):
						features.append("upper=true")
					if "-" in word:
						features.append("hyphen=true")

				if i >= 1:
					features.append("pw=" + pairs[i - 1][0])
					features.append("pt=" + pairs[i - 1][1])
					if i >= 2:
						features.append("ppw=" + pairs[i - 2][0])
						features.append("ppt=" + pairs[i - 2][1] + pairs[i - 1][1])
					else:
						features.append("ppw=**START**")
						features.append("ppt=**START**" + " " + pairs[i - 1][1])
				else:
					features.append("pw=**START**")
					features.append("pt=**START**")
					features.append("ppw=**START**")
					features.append("ppt=**START****START**")

				if i < len(pairs) - 1:
					features.append("nw=" + pairs[i + 1][0])
					if i < len(pairs) - 2:
						features.append("nnw=" + pairs[i + 2][0])
					else:
						features.append("nnw=**END**")
				else:
					features.append("nw=**END**")
					features.append("nnw=**END**")

				features_file.write(str(tag) + ' ' + ' '.join(map(str, features)) + "\n")
	features_file.close()


if __name__ == "__main__":
	main(sys.argv)
