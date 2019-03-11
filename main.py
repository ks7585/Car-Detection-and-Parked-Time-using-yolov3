#!/bin/bash
__author__="Karthik Sivarama Krishnan"
"""
main.py

This file has three functions, where the first function is used to 
download the .ts files and extract the first frame from the file. 
The second function extracts the lane information from the dataset and 
creates a lane mask data. The third function is the main function which
drives the entire flow of the code. 
"""
import cv2
import numpy as np
import glob
import os
import sys
import shutil
from same_car import *
from yolo_video import *
import argparse
from yolo import *
from multiprocessing.pool import ThreadPool

paths = os.getcwd()


def fetch(link):
    """
    This function gets in the list of url links to download
    and run the script "fetch-and-extract.sh".
    :param link: list of url links
    :return: None
    """
    name, link = link
    os.system("bash ../fetch-and-extract.sh {} {} ".format(link, name))


def extract_lane(image_path, lane):
    """
    This function extracts the lane from every image
    and creates a mask of the particular image and save it
    in folder "/Lane/".
    :param image_path: path to the image files
    :param lane: lane id
    :return: None i
    """

    img = cv2.imread(str(image_path))
    lane_boundaries = []
    if int(lane) == 1:
        lane_boundaries = np.array([[130, 184], [217, 292], [363, 224], [255, 120]], np.int32)
    elif int(lane) == 2:
        lane_boundaries = np.array([[100, 211], [151, 292], [240, 256], [164, 187]], np.int32)
    elif int(lane) == 3:
        lane_boundaries = np.array([[54, 212], [93, 298], [173, 265], [115, 191]], np.int32)
    else:
        raise Exception("Only Lanes 1, 2 and 3 are available currently")

    lane_boundaries = lane_boundaries.reshape((-1, 1, 2))
    lane_mask = np.zeros(img.shape).astype(img.dtype)
    mask_color = [255, 255, 255]
    cv2.fillPoly(lane_mask, [lane_boundaries], mask_color)
    result = cv2.bitwise_and(img, lane_mask)
    cv2.imwrite("../Lane/" + str(image_path), result)


def main():
    """
    This is the main driver function
    :return: None
    """
    flags = None
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--index', type=str,
        help='path to the index file '
    )

    parser.add_argument(
        '--start', type=int,
        help='start frame'
    )

    parser.add_argument(
        '--stop', type=int,
        help='end frame'
    )
    parser.add_argument(
        '--lane', type=int, default=1,
        help=' lane number 1,2,3'
    )

    parser.add_argument(
        '--concurrency', default=False, action="store_true",
        help='concurrent file download  '
    )

    flags = parser.parse_args()
    urls = []
    image_names = []

    with open(flags.index) as search:
        for line in search:
            line = line.rstrip()
            lines = line.rsplit('.', 1)[0]
            if flags.start <= int(lines) <= flags.stop:
                urls.append((lines, "__Website_to_download_dataset__" + str(lines) + ".ts"))
                image_names.append(str(lines) + '.jpg')

    search.close()

    if os.path.exists(paths + "/Data"):
        shutil.rmtree(paths + "/Data")
        os.makedirs(paths + "/Data")
    else:
        os.makedirs(paths + "/Data")
    os.chdir(paths + "/Data")

    if flags.concurrency:
        ThreadPool(2).map(fetch, urls)
    else:
        for url in urls:
            fetch(url)

    # Extract lane
    if os.path.exists(paths + "/Lane"):
        shutil.rmtree(paths + "/Lane")
        os.makedirs(paths + "/Lane")
    else:
        os.makedirs(paths + "/Lane")

    for files in glob.glob("*.jpg"):
        extract_lane(files, flags.lane)

    os.chdir(paths + "/Lane")
    if os.path.exists(paths + "/Predictions"):
        shutil.rmtree(paths + "/Predictions")
        os.makedirs(paths + "/Predictions")
    else:
        os.makedirs(paths + "/Predictions")

    prediction_timestamps = []
    prediction = ''
    color_of_car, same = None, None

    for x in range(0, len(image_names)):
        predict_label, left_corner, top_corner, right_corner, bottom_corner, out_image = predict(image_names[x])

        if predict_label.strip() == 'car' and x != (len(image_names)-1):

            prediction_timestamps.append(os.path.splitext(image_names[x])[0])
            color_of_car = color(image_names[x], left_corner, top_corner, right_corner, bottom_corner)
            same = sameCar(image_names[x], image_names[x + 1], left_corner, top_corner, right_corner, bottom_corner)

        elif predict_label.strip() == 'car' and x == (len(image_names)-1):
            time = os.path.splitext(image_names[x])[0]
            print("Found car of color ", color_of_car)
            print("was parked from ", min(prediction_timestamps), "to", time)
            print("Around ", int(int(time)-int(min(prediction_timestamps))/60), "Minutes")
            open_cv_image = np.array(out_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            cv2.imwrite(paths+"/Predictions/" + str(x) + ".jpg", open_cv_image)
            print("Prediction written to /Predictions/ folder")

        elif predict_label.strip() == '' and x == 0:
            print("No car found")

        elif predict_label.strip() != prediction and prediction == 'car' and same is False:
            print("Found car of color ", color_of_car)
            print("was parked from ", min(prediction_timestamps), "to", max(prediction_timestamps))
            print("Around ", int(int(max(prediction_timestamps)) - int(min(prediction_timestamps)) / 60), "Minutes")
            open_cv_image = np.array(out_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            cv2.imwrite(paths + "/Predictions/" + str(x) + ".jpg", open_cv_image)
            print("Prediction written to /Predictions/ folder")

        else:
            pass
        prediction = predict_label.strip()


if __name__ == '__main__':
    main()
