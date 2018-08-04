import sys
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def main(argv):
	feature_vecs_file = argv[1]
	model_file = argv[2]
	X_train, y_train = load_svmlight_file(feature_vecs_file)
	clf = LogisticRegression(multi_class='multinomial', solver='saga')
	clf.fit(X_train, y_train)
	joblib.dump(clf, model_file)


if __name__ == "__main__":
	main(sys.argv)
