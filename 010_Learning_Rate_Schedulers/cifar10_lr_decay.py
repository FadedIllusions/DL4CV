# Usage:
# python cifar10_lr_decay.py --output output/lr_decay_f0.25_plot.png


# Set matplotlib Backend So Figures Can Be Saved In Background
import matplotlib
matplotlib.use("Agg")

# Import Needed Packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from helpers.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):
	# Init Base Learning Rate, Drop Factor, Epochs To Drop
	initAlpha = 0.01
	factor = 0.25
	dropEvery = 5
	
	# Compute Learning Rate For Current Epoch
	alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
	
	# Return Learning Rate
	return float(alpha)


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path To Output Loss/Accuracy Plot")
args = vars(ap.parse_args())


# Load Training And Testing Data, Encode [0,1]
print("[INFO] Loading CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Encode Labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Init Label Names
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog",
			  "frog", "horse", "ship", "truck"]


# Define Callbacks
callbacks = [LearningRateScheduler(step_decay)]

# Init Optimizer And Network
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train Network
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, 
											   callbacks=callbacks, verbose=1)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=64)
print (classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))


# Plot Training Loss And Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0,40), H.history["val_loss"], label="Test Loss")
plt.plot(np.arange(0,40), H.history["acc"], label="Training Acc")
plt.plot(np.arange(0,40), H.history["val_acc"], label="Test Acc")
plt.title("Training Loss And Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])