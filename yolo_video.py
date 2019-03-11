from yolo import YOLO
from PIL import Image
from types import SimpleNamespace


def detect_img(yolo, img):
    image = Image.open(img)
    r_image, label, (left, top), (right, bottom) = yolo.detect_image(image)
    return label, left, top, right, bottom, r_image


def predict(input_path):
    # O defines the default value, so suppress any default here

    args = SimpleNamespace(model_path='../model_data/yolo.h5',
                           anchors_path='../model_data/yolo_anchors.txt',
                           classes_path='../model_data/coco_classes.txt',
                           score=0.3,
                           iou=0.45,
                           model_image_size=(416, 416),
                           gpu_num=1)

    label = detect_img(YOLO(**vars(args)), input_path)

    return label
