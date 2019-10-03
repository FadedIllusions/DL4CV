# Import Needed Packages
from helpers.nn import Perceptron
import numpy as np


# Construct AND Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])


# Define Perceptron And Train It
print("[INFO] Training Perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# Evaluate Perceptron
print("[INFO] Testing Perceptron...")

# Iterate Over Data Points
for (x, target) in zip(X, y):
	# Make Prediction On Data Point And Display Result
	pred = p.predict(x)
	print("[INFO] Data: {}, Truth: {}, Prediction: {}".format(
		x, target[0], pred))