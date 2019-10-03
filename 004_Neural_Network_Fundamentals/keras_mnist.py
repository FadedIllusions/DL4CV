# Usage:
# python keras_mnist.py --output output/keras_mnist.png


# Import Needed Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path To Output Loss/Accuracy Plot")
args = vars(ap.parse_args())


# Grab MNIST Dataset
# If First Time Running Script, Download May Take A Minute
# (55MB MNIST Dataset Will Be Downloaded)
print("[INFO] Loading MNIST (Full) Dataset...")
dataset = datasets.fetch_openml('mnist_784', version=1, cache=True)

# Scale Raw Px Intensities To Range [0,1.0]
# Construct Train/Test Split. 75/25.
data = dataset.data.astype("float")/255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

# Convert Labels From Ints To Vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# Define The 784-256-128-10 Architecture Using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))


# Train Model Using SGD
print("[INFO] Training Network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
							target_names=[str(x) for x in lb.classes_]))


# Plot Training Loss And Accuracy
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