# Import Needed Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from helpers.nn.conv import LeNet
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


# Get MNIST Dataset
print("[INFO] Accessing MNIST...")
dataset = datasets.fetch_openml('mnist_784', version=1, cache=True)
data = dataset.data

# If Channels First, Reshape Design Matrix:
# num_samples x depth x row x columns
if K.image_data_format() == "channels_first":
	data = data.reshape(data.shape[0], 1, 28, 28)
	
else:
	data = data.reshape(data.shape[0], 28, 28, 1)
	

# Scale Input Data [0,1]. Init Train/Test Split. 75/25.
(trainX, testX, trainY, testY) = train_test_split(data/255.0, dataset.target.astype("int"),
												  test_size=0.25, random_state=42)

# Encode Labels
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)


# Init Optimizer And Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train Network
print("[INFO] Training Network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
							target_names=[str(x) for x in le.classes_]))


# Plot Training Loss And Accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0,20), H.history["val_loss"], label="Test Loss")
plt.plot(np.arange(0,20), H.history["acc"], label="Training Acc")
plt.plot(np.arange(0,20), H.history["val_acc"], label="Test Acc")
plt.title("Training Loss And Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()