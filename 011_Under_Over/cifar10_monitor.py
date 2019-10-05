# Usage:
# python cifar10_monitor.py --output output


# Set matplotlib Backend So Figures Can Be Saved In Background
import matplotlib
matplotlib.use("Agg")


# Import Needed Packages
from sklearn.preprocessing import LabelBinarizer
from helpers.callbacks import TrainingMonitor
from helpers.nn.conv import MiniVGGNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import argparse
import os


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path To Output Directory")
args = vars(ap.parse_args())


# Show Infor On Process ID
print("[INFO] Process ID: {}".format(os.getpid()))


# Load Training/Test Data, Encode [0,1]
print("[INFO] Loading CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Encode Labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Init Label Names
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
			  "dog", "frog", "horse", "ship", "truck"]


# Init SGD Optimizer, Without Any Learning Rate Decay,
# Build Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct Set Of Callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# Train Network
print("[INFO] Training Network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, 
		  callbacks=callbacks, verbose=1)