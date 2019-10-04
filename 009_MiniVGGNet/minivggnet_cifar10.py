# Usage:
# python minivggnet_cifar10.py --output output/


# Set matplotlib Backend So Figures Can Be Saved In Background
import matplotlib
matplotlib.use("Agg")

# Import Needed Packages
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from helpers.nn.conv import MiniVGGNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path To Output Loss/Accuracy Plot")
args = vars(ap.parse_args())


# Load Training And Tesst Data, Encode
print("[INFO] Loading CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Encode Labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Init Label Names For CIFAR-10 Dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", 
			  "dog", "frog", "horse", "ship", "truck"]


# Init Optimmizer And Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train Network
print("[INFO] Training Network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,
											   epochs=40, verbose=1)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
							target_names=labelNames))


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
plt.show()