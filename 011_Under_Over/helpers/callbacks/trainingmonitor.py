# Import Needed Packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		# Store Output Path For Figure, JSON, Serialized File
		# And Starting Epoch
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		self.startAt = startAt
		
	def on_train_begin(self, logs={}):
		# Init History Dictionary
		self.H = {}
		
		# If JSON History Path Exists, Load Training History
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())
			
				# Check If Starting Epoch Supplied
				if self.startAt > 0:
					# Iterate Entries In History Log
					# Trim Entries Past Starting Epoch
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]
					
	def on_epoch_end(self, epoch, logs={}):
		# Iterate Over Logs And Update Loss, Accuracy, Etc
		# For Entire Training Process
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l
			
		# Check If Training History Should Be Serialized
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()
			
		# Ensure At Least Two Epochs Passed Before Plotting
		# (Epoch Starts At Zero)
		if len(self.H["loss"]) > 1:
			# Plot Training Loss And Accuracy
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="Training Loss")
			plt.plot(N, self.H["val_loss"], label="Test Loss")
			plt.plot(N, self.H["acc"], label="Training Acc")
			plt.plot(N, self.H["val_acc"], label="Test Acc")
			plt.title("Training Loss And Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()
			
			# Save Figure
			plt.savefig(self.figPath)
			plt.close()