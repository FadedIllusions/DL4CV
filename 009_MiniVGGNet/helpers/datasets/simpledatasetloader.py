# Import Needed Packages
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
	
	def __init__(self, preprocessors=None):
		# Store Image Preprocessor
		self.preprocessors = preprocessors
		
		# If Preprocessors Are None, Init As Empty List
		if self.preprocessors is None:
			self.preprocessors=[]
			
	def load(self, imagePaths, verbose=-1):
		# Init List Of Features And Labels
		data=[]
		labels=[]
		
		# Iterate Over Input Images
		for (i, imagePath) in enumerate(imagePaths):
			# Load Image, Extract Class Label
			# Assuming Out Path Has Following Format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
			
			# Check If Preprocessors Not None
			if self.preprocessors is not None:
				# Iterate Over Preprocessors And Apply Each
				for p in self.preprocessors:
					image = p.preprocess(image)
					
			# Treat Processed Image As "Feature Vector"
			# By Updating Data List Followed By Labels
			data.append(image)
			labels.append(label)
			
			# Show An Update Every "Verbose" Images
			if verbose>0 and i>0 and (i+1)%verbose==0:
				print("[INFO] Processed {}/{} Images".format(i+1, len(imagePaths)))
				
		# Return Tuple Of Data And Labels
		return (np.array(data), np.array(labels))
