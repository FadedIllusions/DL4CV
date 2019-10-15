# A Good Application Of Checkpointing Is To Serialize Your Network To Disk Each Time There Is
# An Improvement During Training -- Either A Decrease In Loss Or And Improvement In Accuracy.

# Import Needed Packages
from sklearn.preprocessing import LabelBinarizer
from helpers.nn.conv import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse


# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Path TO Best Model Weights File")
args = vars(ap.parse_args())


# Load Training/Test Data, Scale To Range [0,1]
print("[INFO] Loading CIFAR-10 Data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# Encode Labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Init Optimizer And Model
print("[INFO] Compiling Model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct Callback To Save Only Best Model To Disk Based On Validation Loss
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
callbacks = [checkpoint]


# Train Network
print("[INFO] Training Network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=64, epochs=40,
			  callbacks=callbacks, verbose=2)