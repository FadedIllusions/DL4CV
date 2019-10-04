# Import Needed Packages
import numpy as np


class Perceptron:
	def __init__(self, N, alpha=0.1):
		# Init Weight Matrix And Store Learning Rate
		self.W = np.random.randn(N+1)/np.sqrt(N)
		self.alpha = alpha
		
	def step(self, x):
		# Apply Step Function
		return 1 if x>0 else 0
	
	def fit(self, X, y, epochs=10):
		# Insert Column Of 1's As Last Entry In Feature Matrix To Treat
		# Bias As A Trainable Parameter Within The Weight Matrix
		X = np.c_[X, np.ones((X.shape[0]))]
		
		# Iterate Desired Number Of Epochs
		for epoch in np.arange(0, epochs):
			# Iterate Over Individual Data Points
			for (x, target) in zip(X, y):
				# Dot Product Between Input Features And Weight Matrix
				# Pass Value Through Step Function To Obtain Prediction
				p = self.step(np.dot(x, self.W))
				
				# Perform Weight Update If Prediction Doesn't Match Target
				if p != target:
					# Determine Error
					error = p - target
					
					# Update Weight Matrix
					self.W += -self.alpha * error * x
					
	def predict(self, X, addBias=True):
		# Ensure Input Is A Matrix
		X = np.atleast_2d(X)
		
		# Check If Bias Column Should Be Added
		if addBias:
			# Insert Column Of 1's As Last Entry In Feature Matrix (Bias)
			X = np.c_[X, np.ones((X.shape[0]))]
			
		# Dot Product Input Features And Weight Matrix
		# Pass Value Through Step Function
		return self.step(np.dot(X, self.W))