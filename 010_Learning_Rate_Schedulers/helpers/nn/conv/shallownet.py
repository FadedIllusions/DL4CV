# Import Needed Packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Init Model And Input Shape To Be Channels Last
		model = Sequential()
		inputShape = (height, width, depth)
		
		# If Using Channels First, Update Input Shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			
		# Define First (And Only) CONV=>RELU Layer
		model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		
		# Softmax Classifier
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		# Return Constructed Network Architecture
		return model