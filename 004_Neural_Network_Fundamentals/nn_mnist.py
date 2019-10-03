# Import Needed Packages
from helpers.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# Load MNIST Dataset And Apply Min/Max Scaling To Scale Px Intensities
# To Range [0,1]
# Each Image Represented By 8x8=64 Dimension Feature Vector
print("[INFO] Loading MNIST (Sample) Dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max()-data.min())
print("[INFO] Samples: {} Dimensions: {}".format(data.shape[0], data.shape[1]))

# Train/Test Split. 75/25
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# Binarize Labels -- Cvt From Ints To Vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# Train Network
print("[INFO] Training Network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# Evaluate Network
print("[INFO] Evaluating Network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))