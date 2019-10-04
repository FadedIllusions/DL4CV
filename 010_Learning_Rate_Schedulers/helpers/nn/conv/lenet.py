# Import Needed Packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Init Model
		model = Sequential()
		inputShape = (height, width, depth)
		
		# If Channels First, Update Input Shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			
		# First Set Of CONV=>RELU=>POOL Layers
		model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		
		# Second Set Of CONV=>RELU=>POOL Layers
		model.add(Conv2D(50, (5,5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		
		# First (And Only) Set Of FC=>RELU Layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		
		# Softmax Classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		# Return Constructed Network Architecture
		return model