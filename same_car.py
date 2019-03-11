__author__="Karthik Sivarama Krishnan"
"""
same_car.py

This file has two functions, where one function
checks if the car in 2 consecutive frames are equal and 
the other function detecting the color of the car detected.
"""

import cv2
import numpy as np


def sameCar(img1, img2, left, top, right, bottom):
    """
    This function is used to take in consecutive image frames,
    resize them and use orb based feature detectors to extract
    features and track them to find matches
    :param img1: path to image 1
    :param img2: path to image 2
    :param left: left corner predicted from YOLO v3
    :param top: top corner predicted from YOLO v3
    :param right: right corner predicted from YOLO v3
    :param bottom: bottom corner predicted from YOLO v3
    :return: boolean value
    """
    image_1 = cv2.imread(img1)
    image_2 = cv2.imread(img2)
    roi_image_1 = image_1[top:bottom, left:right]
    roi_image_2 = image_2[top:bottom, left:right]
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=14, WTA_K=2, patchSize=14)

    keypoints_image1, descriptors_image1 = orb.detectAndCompute(roi_image_1, None)
    keypoints_image2, descriptors_image2 = orb.detectAndCompute(roi_image_2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_image1, descriptors_image2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if len(good) > 10:
        #print("CAR IS SAME IN BOTH THE IMAGE")
        return True
    else:
        #print("NOT SAME")
        return False


def color(img, left, top, right, bottom):
    """
    This function is used to detect the color of the car.
    could detect black, white or red
    :param img: path to the image
    :param left: left corner predicted from YOLO v3
    :param top: top corner predicted from YOLO v3
    :param right: right corner predicted from YOLO v3
    :param bottom: bottom corner predicted from YOLO v3
    :return: string value depicting color
    """
    image = cv2.imread(img)
    image_roi = image[top:bottom, left:right]
    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

    # definig the range of white color
    white_lower = np.array([0, 0, 0], np.uint8)
    white_upper = np.array([0, 0, 255], np.uint8)

    # defining the Range of red color
    red_lower = np.array([0, 100, 100], np.uint8)
    red_upper = np.array([160, 100, 100], np.uint8)

    # defining the Range of black color
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 180, 180], np.uint8)

    # finding the range of red,blue and black color in the image
    white = cv2.inRange(hsv, white_lower, white_upper)
    red = cv2.inRange(hsv, red_lower, red_upper)
    black = cv2.inRange(hsv, black_lower, black_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")
    white = cv2.dilate(white, kernal)
    red = cv2.dilate(red, kernal)
    black = cv2.dilate(black, kernal)

    white_1 = cv2.bitwise_and(image_roi, image_roi, mask=white)
    red_1 = cv2.bitwise_and(image_roi, image_roi, mask=red)
    black_1 = cv2.bitwise_and(image_roi, image_roi, mask=black)

    if ((np.count_nonzero(white_1) > np.count_nonzero(red_1)) and (
            np.count_nonzero(white_1) > np.count_nonzero(black_1))):
        color = 'White'
    elif np.count_nonzero(red_1) > np.count_nonzero(white_1) and np.count_nonzero(red_1) > np.count_nonzero(black_1):
        color = 'Red'
    elif np.count_nonzero(black_1) > np.count_nonzero(white_1) and np.count_nonzero(black_1) > np.count_nonzero(red_1):
        color = 'Black'
    else:
        color = 'Could not recognize'

    return color
