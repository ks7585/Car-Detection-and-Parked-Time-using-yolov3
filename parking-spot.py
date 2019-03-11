__author__="Karthik Sivarama Krishnan"
"""
parking-spot.py

This file takes in the test image path and try to detect 
the orb features from each lane to detect whether the lane 
is occupied or not 
"""

import cv2
import argparse


def main():

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options 
        '''
    parser.add_argument(
        '--input', type=str,
        help='path to the input image file '
    )
    FLAGS = parser.parse_args()

    test_image = cv2.imread(FLAGS.input)

    lane_roi = [[189, 217, 69, 46], [136, 211, 55, 62], [87, 207, 46, 71], [45, 215, 46, 57], [17, 225, 38, 53]]
    for r in lane_roi:
        cropped_image = test_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=14, WTA_K=2, patchSize=14)
        keypoints = orb.detect(cropped_image, None)

        # # compute the descriptors with ORB
        keypoints, descriptors = orb.compute(cropped_image, keypoints)
        if len(keypoints) < 10:
            cv2.rectangle(test_image, (r[0], r[1]), (r[2]+r[0], r[3]+r[1]), (0, 255, 0), 2)

    cv2.imshow("Available Parking Spaces.", test_image)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()