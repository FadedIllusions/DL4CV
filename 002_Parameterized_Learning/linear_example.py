# Import Needed Packages
import numpy as np
import cv2


# Init Class Labels And Set Pseudorandom Generator Seed
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# Randomly Init Weight Matrix And Bias Vector
W = np.random.randn(3,3072)
b = np.random.randn(3)

# Load Image, Resize, Flatten Into "Feature Vector" Representation
original = cv2.imread("beagle.png")
image = cv2.resize(original, (32,32)).flatten()

# Compute Output Scores By Taking Dot Product Between W Matrix And Image PXs,
# Followed By Adding Bias
scores = W.dot(image)+b

# Iterate Over Scores And Labels And Display
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))
	
# Draw Label With Highest Score As Prediction
cv2.putText(original, "Label: {}".format(labels[np.argmax(scores)]), 
			(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# Display Image
cv2.imshow("Image", original)
cv2.waitKey(0)