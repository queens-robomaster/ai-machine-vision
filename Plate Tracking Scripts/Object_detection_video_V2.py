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

# but I changed it to make it more understandable to me.

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

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test_images/clip_39.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

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
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

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
                default="kcf", help="tracker type")
args = vars(ap.parse_args())
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
out = cv2.VideoWriter('test_images/clipBOXED_39.mp4',
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
        line_thickness=4,
        min_score_thresh=0.60)

    
    bestBox = np.squeeze(boxes)[0, :]
    imgHeight = frame.shape[0]
    imgWidth = frame.shape[1]

    xmin = int(bestBox[1]*imgWidth)
    ymin = int(bestBox[0]*imgHeight)
    xmax = int(bestBox[3]*imgWidth)
    ymax = int(bestBox[2]*imgHeight)

    print("xmin = %d, ymin = %d\nxmax = %d, ymax = %d" %
          (xmin, ymin, xmax, ymax))

    # draw circle at point on bestplate
    cv2.circle(frame, (xmin, ymin), 5, (233, 68, 255), -1)

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, imgHeight - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        initBB = (int(imgWidth*bestBox[1]), int(imgHeight*bestBox[0]), int(
            imgWidth*(bestBox[3]-bestBox[1])), int(imgHeight*(bestBox[2]-bestBox[0])))
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # cv2.rectangle(frame,(int(imgWidth*bestBox[1]), int(imgHeight*bestBox[0])), (int(imgWidth*bestBox[3]), int(imgHeight*bestBox[2])),(100,100,100),3)
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
