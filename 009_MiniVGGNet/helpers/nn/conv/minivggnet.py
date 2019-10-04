# Import Needed Packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# Init Model Along With Shape And Dimensions, Channels Last
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		
		# If Channels First, Update Input Shape And Channels Dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
			
		# First CONV=>RELU=>CONV=>RELU=>POOL Layer Set
		model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		
		# Second CONV=>RELU=>CONV=>RELU=>POOL Layer Set
		model.add(Conv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		
		# First (And Only) Set Of FC=>RELU Layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		
		# Softmax Classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		# Return Constructed Network Architecture
		return model