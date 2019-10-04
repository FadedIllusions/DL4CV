# Import Needed Packages
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# Store Image Data Format
		self.dataFormat = dataFormat
		
	def preprocess(self, image):
		# Apply Keras Utility Function To Correctly Rearrange Dimensions
		return img_to_array(image, data_format=self.dataFormat)