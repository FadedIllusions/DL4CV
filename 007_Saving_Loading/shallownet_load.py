# Usage:
# python shallownet_load.py --dataset ../datasets/animals/images/ --model shallownet_weights.hdf5


# Import Needed Packages
from helpers.preprocessing import ImageToArrayPreprocessor
from helpers.preprocessing import SimplePreprocessor
from helpers.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2


# Construct Argument Parser And Parse Argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path To Input Dataset")
ap.add_argument("-m", "--model", required=True, help="Path To Pre-Trained Model")
args = vars(ap.parse_args())


# Init Class Labels
classLabels = ["cat", "dog", "panda"]

# Grab Image Paths, Randomly Sample
print("[INFO] Sampling Images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# Innit Image Processors
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

# Load Dataset, Scale Intensities
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0


# Load Pre-Trained Network
print("[INFO] Loading Pre-Trained Network...")
model = load_model(args["model"])

# Image Predictions
print("[INFO] Predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Iterate Over (Sample) Image Paths
for (i, imagePath) in enumerate(imagePaths):
	# Load Image, Predict, Display
	image = cv2.imread(imagePath)
	cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10,30), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
	cv2.imshow("{}".format(classLabels[preds[i]]), image)
	cv2.waitKey(0)