# Usage:
# python keras_cifar10.py --output output/keras_cifar10.png


# Import Needed Packages
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path To Output Loss/Accuracy Plot")
args = vars(ap.parse_args())


# Load Training And Test Data, Scale It To Range [0,1]
# Reshape Design Matrix
print("[INFO] Loading CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# Convert Labels From Ints To Vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Init Label Names For CIFAR-10 Dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]


# Define 3072-1024-512-10 Architecture Using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Train Model Using SGD
print("[INFO] Training Network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# Evaluate Network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
							target_names=labelNames))


# Plot The Training Loss And Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="Train Acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])