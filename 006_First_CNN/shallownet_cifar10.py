# Import Needed Packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from helpers.nn.conv import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


# Load Training And Test Data
# Scale It To Range [0,1]
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


# Init Optimizer And Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train Network
print("[INFO] Training Network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, 
			  epochs=40, verbose=1)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
							target_names=labelNames))


# Plot Training Loss And Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="Training Acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()