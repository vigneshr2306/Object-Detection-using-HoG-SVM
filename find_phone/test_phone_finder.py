'''
Code Author: Vignesh Ravikumar
Task: Finding phone in a RGB image

Gist: This file is used to load the model that was trained using SVM, and use the learned features to predict the phone using
a sliding window approach. We get the bounding box and confidence scores for several windows. Based on the maximum bounding box 
confidence score, the centroid of the window is calculated. 
'''

# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
import joblib
import cv2
import argparse as ap
from nms import nms
from config import *
import numpy as np
import pandas as pd
import os

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=int)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    filename = os.path.basename(args["image"])
    print(filename) #gets name of file from image path
    # Read the image
    im = imread(args["image"], -1)
    min_wdw_sz = (50, 50)
    step_size = (2, 2)
    downscale = args['downscale']
    visualize_det = args['visualize']

    model_path = '../data/models/svm.model'
    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if scale==1:
            break
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block)
            fd1 = np.array(fd.reshape(1,-1))
            pred = clf.predict(fd1)
            if pred == 1:
                print("Calculating...")
                detections.append((x, y, clf.decision_function(fd1),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
        # Move the the next scale
        scale+=1


    # Perform Non Maxima Suppression
    detections = nms(detections, threshold)
    detections_prob = []
    
    #storing the confidence scores of the detections in a list detections_prob
    for det in detections:
        detections_prob.append(float(det[2]))   

    #finding the box which has the highest confidence score
    max_prob_idx = detections_prob.index(max(detections_prob)) 

    #calculates the left top coordinate of the bounding box
    x_min = detections[max_prob_idx][0]
    y_min = detections[max_prob_idx][1]

    #given 50x50 as window size, centroid can be found by adding half of that (25) to the x_min and y_min
    x_center = float((x_min + 25)/483.89558845) #division factor for normalization
    y_center = float((y_min + 25)/327.91642484)
    print(x_center,'\t',y_center)
    

