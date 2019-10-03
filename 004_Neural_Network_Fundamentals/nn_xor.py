# Import Needed Packages
from helpers.nn import NeuralNetwork
import numpy as np


# Construct AND Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])


# Define 2-2-1 Neural Network And Train
nn = NeuralNetwork([2,2,1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# Iterate XOR Data Points
for (x, target) in zip(X, y):
	 # Predict On Data Point And Display Result
	 pred = nn.predict(x)[0][0]
	 step = 1 if pred>0.5 else 0
	 
	 print("[INFO] Data: {}, Truth: {}, Pred: {:.4f}, Step: {}".format(
		 x, target[0], pred, step))