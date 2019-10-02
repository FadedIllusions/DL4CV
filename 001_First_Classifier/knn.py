# Usage
# python knn.py --dataset ../datasets/animals/images/

# KNN -- K-Nearest Neighbor Classifier
# Algorithm Directly Relies On The Distance Between Feature Vectors (In Our
# Case, The Raw RGB Pixel Intensities Of Images.). Classifies Unknown Data Points
# By Finding Most Common Class Among K-Closest Examples. Each Point In K-Closest
# Data Points Casts A Vote And The Category With Highest Number Of Votes Wins


# Import Needed Packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helpers.preprocessing import SimplePreprocessor
from helpers.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# Consturct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path To Input Dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# Of Nearest Neighbors")
ap.add_argument("-j", "--jobs", type=int, default=-1, 
				help="# Of Jobs For KNN Distance. (-1 Uses All Available Cores.)")
args = vars(ap.parse_args())


# Grab List Of Images We'll Be Describing
print("[INFO] Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))


# Init Image Preprocessor, Load Dataset, Reshape Data Matrix
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# Display Memory Consumption Data For Images
print("[INFO] Features Matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))


# Encode Labels
le = LabelEncoder()
labels = le.fit_transform(labels)


# Create Train/Test Split (75/25)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# Train And Evaluate KNN Classifier On Raw Pixel Intensities
print("[INFO] Evaluating KNN Classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX,trainY)

# Display Classification Report
print(classification_report(testY, model.predict(testX), target_names=le.classes_))