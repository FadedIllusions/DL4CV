# Import Needed Packages
import numpy as np


class NeuralNetwork:
	def __init__(self, layers, alpha=0.1):
		# Init List Of Weight Matrices
		# Store Network Architecture And Learning Rate
		self.W = []
		self.layers = layers
		self.alpha = alpha
		
		# Iterate From Idx Of 1st Layer,
		# Stopping Before Reaching Last Two Layers
		for i in np.arange(0, len(layers)-2):
			# Randomly Init Weight Matrix Connecting Nodes In Each
			# Respective Layer Together, Adding Extra Node For Bias
			w = np.random.randn(layers[i]+1, layers[i+1]+1)
			self.W.append(w/np.sqrt(layers[i]))
			
		# Last Two Layers:
		# Input Connections Need Bias, Output Does Not
		w = np.random.randn(layers[-2]+1, layers[-1])
		self.W.append(w/np.sqrt(layers[-2]))
		
	def __repr__(self):
		# Construct And Return String Representation Of Network
		return "Neural Network: {}".format("-".join(str(l) for l in self.layers))
	
	def sigmoid(self, x):
		# Compute And Return Sigmoid Activation Value
		# For Given Input Value
		return 1.0/(1+np.exp(-x))
	
	def sigmoid_deriv(self, x):
		# Compute Derivative Of Sigmoid /Assuming/ 'x' Has Already
		# Been Passed Through Sigmoid Function
		return x*(1-x)
	
	def fit(self, X, y, epochs=1000, displayUpdate=100):
		# Insert Column Of 1's As Last Entry In Feature Matrix
		# So As To Make Bias A Trainable Parameter
		X = np.c_[X, np.ones((X.shape[0]))]
		
		# Iterate Desired Number Of Epochs
		for epoch in np.arange(0, epochs):
			# Iterate Over Individual Data Points And Train Network On Point
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)
				
			# Check If Should Display Training Update
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] Epoch: {}, Loss: {:.7f}".format(epoch+1, loss))
				
	def fit_partial(self, x, y):
		# Construct List Of Output Activations For Each Layer As Data Flows Through
		# Network. First Layer Input Feature Vector.
		A = [np.atleast_2d(x)]
		
		# ---   ---   ---   FEEDFORWARD:
		
		# Iterate Over Layers
		for layer in np.arange(0, len(self.W)):
			# Feedforward Activation At Current Layer By Taking Dot Product Of Activation
			# And Weight Mtrix -- "Net Input" Of Current Layer
			net = A[layer].dot(self.W[layer])
			
			# Compute "Net Output" By Applying Nonlinear Activation
			out = self.sigmoid(net)
			
			# Add To List Of Activations
			A.append(out)
			
		# ---   ---   ---   BACKPROPAGATION:
			
		# Compute Error
		error = A[-1] - y
			
		# Apply Chain Rule And Build List Of Deltas 'D'
		# First Layer In Deltas -- Error Of Output * Derivative Of Activation Function
		D = [error * self.sigmoid_deriv(A[-1])]
			
		# Iterate Over Layers, In Reverse Order (Ignoring Last Two)
		# Apply Chain Rule
		for layer in np.arange(len(A)-2, 0, -1):
			# Current Layer Delta = Previous Layer Delta Dotted With Current Layer Weight 
			# Matrix, Followed By Multiplying Delta By Derivation Of Nonlinear Activation
			# For Activations Of Current Layer
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)
				
			# Reverse Deltas Since Iterated In Reverse
		D = D[::-1]
			
		# ---   ---   --- WEIGHT UPDATE PHASE:
			
		# Iterate Layers
		for layer in np.arange(0, len(self.W)):
			# Update By Dot Product Of Layer Activations With Respective Deltas,
			# Multiply By Learning Rate And Add Weight Matrix.
			# This Is Where "Learning" Takes Place
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
				
	def predict(self, X, addBias=True):
		# Init Output Prediction As Input Features
		# Will Be (Forward) Propagated To Obtain Final Prediction
		p = np.atleast_2d(X)
			
		# Check If Bias Should Be Added
		if addBias:
			# Insert Column Of 1's As Last Entry In Feature Matrix
			# So As To Make Bias A Trainable Parameter
			p = np.c_[p, np.ones((p.shape[0]))]
				
		# Iterate Over Layers
		for layer in np.arange(0, len(self.W)):
			# Dot Current Activation Value 'p' And Current Layer Weight Matrix
			# And Pass Through Nonlinear Activation To Compute Output Prediction
			p = self.sigmoid(np.dot(p, self.W[layer]))
				
		# Return Predicted Value
		return p
		
	def calculate_loss(self, X, targets):
		# Predict For Input Data Points, Compute Loss
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * np.sum((predictions-targets)**2)
			
		return loss