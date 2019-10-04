# Usage
# python shallownet_animals.py --dataset ../datasets/animals/images/


# Import Needed Packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helpers.preprocessing import ImageToArrayPreprocessor
from helpers.preprocessing import SimplePreprocessor
from helpers.datasets import SimpleDatasetLoader
from helpers.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path To Dataset")
args = vars(ap.parse_args())


# Grab List Of Image Paths
print("[INFO] Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Inint Image Preprocessors
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

# Load Dataset, Scale Raw Px To Range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0


# Train/Test Split. 75/25.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Encode/Binarize Labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# Init Optimizer And Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train Network
print("[INFO] Training Network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
							target_names=["cat", "dog", "panda"]))


# Plot Training Loss And Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="Training Acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()