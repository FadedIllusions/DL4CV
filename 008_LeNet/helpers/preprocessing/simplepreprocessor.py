# Import Needed Packages
import cv2


class SimplePreprocessor:
	
	def __init__(self, width, height, inter=cv2.INTER_AREA):
	# Store Target Image Width, Height, And Interpolation 
	# Method Used When Resizing
	    self.width = width
	    self.height = height
	    self.inter = inter
	
	def preprocess(self, image):
		# Resize Image To Fixed Size, Ignoring Aspect Ratio
		return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
