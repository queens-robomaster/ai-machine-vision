# import necessary packages
import cv2
import numpy as np
import argparse

# construct the terminal arguments for specifying the image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# define the list of boundaries for what defines the colours red and blue
# current boundaries visualized: https://i.imgur.com/y1Eo0yj.png
boundaries = [
    ([17, 15, 100], [50, 56, 200]),  # red boundaries
    ([86, 31, 4], [220, 88, 50])    # blue boundaries
]



# # loop over the boundaries
# for (lower, upper) in boundaries:
#     # create NumPy arrays from the boundaries
#     lower = np.array(lower, dtype="uint8")
#     upper = np.array(upper, dtype="uint8")

#     # find the colors within the specified boundaries and apply the mask
#     mask = cv2.inRange(image, lower, upper)
#     output = cv2.bitwise_and(image, image, mask=mask)

redLower = np.array(boundaries[0][0], dtype = "uint8")
redUpper = np.array(boundaries[0][1], dtype = "uint8")
redMask = cv2.inRange(image, redLower, redUpper)


blueLower = np.array(boundaries[1][0], dtype = "uint8")
blueUpper = np.array(boundaries[1][1], dtype = "uint8")
blueMask = cv2.inRange(image, blueLower, blueUpper)


if np.sum(redMask) > np.sum(blueMask):
    print("redPlate")
else:
    print("bluePlate")


# mask = cv2.add(redMask, blueMask)
# output = cv2.bitwise_and(image, image, mask=mask)

# show the images
# cv2.imshow("images", np.hstack([image, output]))
# cv2.imwrite("plate_thresholded.png", output)
# cv2.waitKey(0)
