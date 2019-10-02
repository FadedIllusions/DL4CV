# Usage:
# python sgd.py


# Import Needed Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
	# Compute Sigmoid Activation Value For Given Input
	return 1.0/(1+np.exp(-x))

def sigmoid_deriv(x):
	# Compute Derivative Of Sigmoid Function /assuming/ Input
	# 'x' Has Already Been Passed Through Activation Function
	return x*(1-x)

def predict(X,W):
	# Take Dot Product Between Features And Weight Matrix
	preds = sigmoid_activation(X.dot(W))
	
	# Apply Step Function To Threshold Outputs To Binary Class Labels
	preds[preds<=0.5]=0
	preds[preds>0]=1
	
	return preds

def next_batch(X, y, batchSize):
	# Iterate Over Dataset 'X' In Mini-Batches, Yielding Tuple Of 
	# Current Batched Data And Labels
	for i in np.arange(0, X.shape[0], batchSize):
		yield (X[i:i+batchSize], y[i:i+batchSize])
		
		
# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# Of Epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="Learning Rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="Size Of SGD Mini-Batches")
args = vars(ap.parse_args())


# Getnerate 2-Class Classification Problem With 1000 Data Points,
# Where Each Data Point Is A 2D Feature Vector
(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0],1))

# Insert Column Of 1's As Last Entry In Feature Matrix -- Allows Us To Treat
# Bias As A Trainable Parameter Within The Weight Matrix
X = np.c_[X, np.ones((X.shape[0]))]

# Create Train/Test Split. 50/50
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)


# Init Weight Matrix And List Of Losses
print("[INFO] Training...")
W = np.random.randn(X.shape[1],1)
losses = []


# Iterate For Desired Number Of Epochs
for epoch in np.arange(0, args["epochs"]):
	# Init Total Loss For Epoch
	epochLoss = []
	
	# Iterate Over Data In Batches
	for (batchX,batchY) in next_batch(trainX,trainY, args["batch_size"]):
		# Dot Product Between Current Batch Of Features And Weight Matrix 'W'; Then, 
		# Pass Value Through Sigmoid Activation Function
		preds = sigmoid_activation(batchX.dot(W))
	
		# Determine 'Error'. Difference Between Preds And True Value
		error = preds - batchY
		epochLoss.append(np.sum(error**2))
		
		# Gradient Descent Update Is Dot Product Between Features And Error
		# Of Sigmoid Derivation Of Predictions
		d = error * sigmoid_deriv(preds)
		gradient = batchX.T.dot(d)
	
		# Update Stage: "Nudge" Weight Matrix In Negative Direction Of Gradient
		# By Taking A Small Step Toward A Set Of "More Optimal" Parameters
		W += -args["alpha"] * gradient
		
	# Update Our Loss History By Taking Average Loss Across Batches
	loss = np.average(epochLoss)
	losses.append(loss)
	
	# Check If Update Should Be Displayed
	if epoch==0 or (epoch+1)%5==0:
		print("[INFO] Epoch={}, Loss={:.7f}".format(int(epoch+1), loss))
		

# Evaluate Model
print("[INFO] Evaluating...")
preds = predict(testX,W)
print(classification_report(testY, preds))



# Plot (Testing) Classification Data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,0], testX[:,1], marker="o", c=testY[:,0], s=30)

# Construct Figure To Plot Loss Over Time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()