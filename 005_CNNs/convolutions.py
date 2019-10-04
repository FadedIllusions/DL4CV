# Usage:
# python convolutions.py --image jemma.png


# Import Needed Packages
from skimage.exposure import rescale_intensity
import numpy  as np
import argparse
import cv2


def convolve(image, K):
	# Grab Dimensions Of Image And Kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = K.shape[:2]
	
	# Allocate Memory For Output Image
	# Pad Borders So Spatial Dimensions Not Reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float")
	
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# Extract ROI By Extracting Center Region Of (x,y) Coordinates Dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
		
			# Perform Convolution Via Element-Wise Multiplication
			# Between ROI And Kernel, Sum Matrix
			k = (roi * K).sum()
		
			# Store Convolved Calue In Output (x,y)-Coordinate Of Output Image
			output[y - pad, x - pad] = k

	# Rescale Output Image To Be In Range [0,255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
		
	# Return Output Image
	return output
	

# Construct Argument Parser And Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path To Input Image")
args = vars(ap.parse_args())


# Construct Average Blurring Kernels Used To Smooth Image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# Construct Sharpening Filter
sharpen = np.array((
	[0,-1,0],
	[-1,5,-1],
	[0,-1,0]), dtype="int")

# Construct Laplacian Kernel To Detect Edge-Like Regions
laplacian = np.array((
	[0,1,0],
	[1,-4,1],
	[0,1,0]), dtype="int")

# Construct Sobel X-Axis Kenel
sobelX = np.array((
	[-1,0,1],
	[-2,0,2],
	[-1,0,1]), dtype="int")

# Construct Sobel Y-Axis Kernel
sobelY = np.array((
	[-1,-2,-1],
	[0,0,0],
	[1,2,1]), dtype="int")

# Construct Emboss Kernel
emboss = np.array((
	[-2,-1,0],
	[-1,1,1],
	[0,1,2]), dtype="int")

# Construct Kernel Bank
# List Of Kernels To Apply Bia Both A Custom Convolve Function
# And OpenCV's filter2D Function
kernelBank = (
	("Small Blur", smallBlur),
	("Large Blur", largeBlur),
	("Sharpen", sharpen),
	("Laplacian", laplacian),
	("Sobel X", sobelX),
	("Sobel Y", sobelY),
	("Emboss", emboss))


# Load Image, Convert To Grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Iterate Over Kernels
for (kernelName, K) in kernelBank:
	# Apply Kernel Using Custom Convolve And OCV's filter2D
	print("[INFO] Applying {} Kernel...".format(kernelName))
	convolveOutput = convolve(gray, K)
	opencvOutput = cv2.filter2D(gray, -1, K)
	
	# Display Output Images
	cv2.imshow("Original", gray)
	cv2.imshow("{} - Convolve".format(kernelName), convolveOutput)
	cv2.imshow("{} - OpenCV".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
