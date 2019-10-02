# Usage
# python regularization.py --dataset ../datasets/animals/images/


# Import Needed Packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from helpers.preprocessing import SimplePreprocessor
from helpers.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path To Input Dataset")
args = vars(ap.parse_args())


# Grab List Of Image Paths
print("[INFO] Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Init Image Preprocessor, Load Dataset, Reshape Data Matrix
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Encode Labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Create Train/Test Split. 75/25
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)


# Iterate Over Set Of Regularizers
for r in (None, "l1", "l2"):
	# Train SGD Classifier Using Softmax Loss Function And The 
	# Specified Regularization Function For 10 Epochs
	print("[INFO] Training Model With '{}' Penalty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant",
						  eta0=0.01, random_state=42)
	model.fit(trainX, trainY)
	
	# Evaluate Classifier
	acc = model.score(testX, testY)
	print("[INFO] '{}' Penalty Accuracy: {:.2f}%".format(r, acc*100))