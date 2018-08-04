import sys
from collections import defaultdict


def main(argv):
	features_filename = argv[1]
	feature_vecs_filename = argv[2]
	feature_map_filename = argv[3]

	tags = set()
	features = set()
	with open(features_filename, 'r') as features_file:
		for line in features_file:
			words = line.split()
			tags.add(words[0])
			features |= set(words[1:])

	tags = {tag: i for i, tag in enumerate(tags)}
	features = {feature: i for i, feature in enumerate(features)}
	tags_per_word = defaultdict(set)
	feature_vecs_file = open(feature_vecs_filename, 'w')
	with open(features_filename, 'r') as features_file:
		for line in features_file:
			words = line.split()
			tag = tags[words[0]]
			temp = [str(features[feature]) + ":1" for feature in words[1:]]
			temp = sorted(temp, key=lambda x: int(x.split(":")[0]))
			if words[1].startswith("form="):
				tags_per_word[words[1].split("=")[1]].add(tag)
			feature_vecs_file.write(str(tag) + ' ' + ' '.join(map(str, temp)) + "\n")
	feature_vecs_file.close()

	feature_map_file = open(feature_map_filename, 'w')
	for k, v in features.items():
		feature_map_file.write(k + " " + str(v) + "\n")
	feature_map_file.write("####META-TAGS-MAP####\n")
	for k, v in tags.items():
		feature_map_file.write(k + " " + str(v) + "\n")
	feature_map_file.write("####META-TAGS-PER-WORD####\n")
	for k, v in tags_per_word.items():
		feature_map_file.write(k + ' ' + ' '.join(map(str, v)) + "\n")
	feature_map_file.close()


if __name__ == "__main__":
	main(sys.argv)
