######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me

# Import packages
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
# For object tracking
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# colour thresholding
def isEnemy(plateImg, enemyTeam):

    # define the list of boundaries for what defines the colours red and blue
    # current boundaries visualized: https://i.imgur.com/y1Eo0yj.png
    boundaries = [
        ([17, 15, 100], [50, 56, 200]),  # red boundaries
        ([86, 31, 4], [220, 88, 50])    # blue boundaries
    ]

    redLower = np.array(boundaries[0][0], dtype="uint8")
    redUpper = np.array(boundaries[0][1], dtype="uint8")
    redMask = cv2.inRange(plateImg, redLower, redUpper)

    blueLower = np.array(boundaries[1][0], dtype="uint8")
    blueUpper = np.array(boundaries[1][1], dtype="uint8")
    blueMask = cv2.inRange(plateImg, blueLower, blueUpper)

    if np.sum(redMask) > np.sum(blueMask):
        colour = "red"
    else:
        colour = "blue"
    return colour == enemyTeam


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'plate_model'
VIDEO_NAME = 'test_images/clip_19.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels', 'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()

graph_options = tf.GraphOptions(
                    optimizer_options=tf.OptimizerOptions(
                        opt_level=tf.OptimizerOptions.L1,
                    )
                )
OptConfig = tf.ConfigProto(
                graph_options=graph_options
            )
			
			
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=OptConfig)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# construct argument parser and parse teh arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to video")
ap.add_argument("-t", "--tracker", type=str,
                default="mosse", help="tracker type")
ap.add_argument("-e", "--enemy", type=str,
                default="blue", help="path to video")

args = vars(ap.parse_args())

ENEMY_TEAM = args["enemy"]
# OpenCV object tracker dictionary
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
# initialize the FPS throughput estimator
fps = None

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
out = cv2.VideoWriter('test_images/clipBOXED_19.mp4',
                      cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), 30.0, (1280, 720))

past_centroidX = 10000
past_centroidY = 10000


while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.80)

    # this segment of code will obtain the boxes of highest confidence in the frame and check each box to see if it is an enemy, stop looping once highest confidence enemy plate is found
    # remove one dimensionality
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)

    imgHeight = frame.shape[0]
    imgWidth = frame.shape[1]

    # numboxes is equal to 5 maximum (we will only check the first 5 boxes) or equal to the amount if there is less than 5.
    # this will limit how many boxes we will check with highest confidence
    numboxes = 5 if len(boxes) >= 5 else len(boxes)

    # will hold score, area, confidence, plate info (xmin,ymin,xmax,ymax)
    plates = []

    # collect all data into a plates array
    for i in range(0, len(boxes)):
        xmin = int(boxes[i][1]*imgWidth)
        ymin = int(boxes[i][0]*imgHeight)
        xmax = int(boxes[i][3]*imgWidth)
        ymax = int(boxes[i][2]*imgHeight)

        area = (xmax-xmin)*(ymax-ymin)
        confidence = scores[i]
        score = confidence*pow(area, 2)
        plate = [xmin, ymin, xmax, ymax]
        plates.append([score, area, confidence, plate])

    # PLATES NumPy ARRAY (score, area, confidence, info[xmin,ymin,xmax,ymax])
    plates = np.asarray(plates)

    # using logical indexing, sort out all plates with confidence less than a certain amount
    plates = plates[plates[:, 2] > 0.8]

    # sort the plates array in descending order according to their "score" (area*confidence)
    plates = np.sort(plates, axis=0)[::-1]
    print(plates)

    bestPlate = None

    if (len(plates) != 0):
        for plate in plates:
            xmin = plate[3][0]
            ymin = plate[3][1]
            xmax = plate[3][2]
            ymax = plate[3][3]
            # print("xmin = %d, ymin = %d\nxmax = %d, ymax = %d" % (xmin, ymin, xmax, ymax))

            # extract the plate as an imageusing its coordinates, will be passed onto isEnemy() to colour threshold
            plateImg = frame[ymin:ymax, xmin:xmax]

            # specific to colour thresholding
            if(isEnemy(plateImg, ENEMY_TEAM)):
                # print("Is an enemy!")
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                              (0, 0, 255), thickness=2)
                bestPlate = plate
                # display pink dot at the specified coordinates for testing
                # cv2.circle(frame, (xmin, ymin), 5, (233, 68, 255), -1)
                break
            else:
                bestPlate = None
                # print("Not an enemy")

				
    # write frame to output video file
    out.write(frame)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
out.release()
cv2.destroyAllWindows()