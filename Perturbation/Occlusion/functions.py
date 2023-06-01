"""
Author: Pu Muxin
Date: 10/10/2022
"""
import cv2
import json
import csv
import numpy as np


def occlude_keypoint_by_rectangle(image, colour, coordinates, index, extend):
    if index == 1:
        x_min, y_min, x_max, y_max = coordinates[0][0] - extend, coordinates[0][1] - extend, \
                                     coordinates[0][0] + extend, coordinates[0][1] + extend
    else:
        x_min, y_min, x_max, y_max = get_actual_bbox_coordinates(coordinates[:index])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, -1)
    return image


def occlude_keypoint_by_circle(image, keypoint_coordinates, index, radius, color, thickness):
    image = cv2.circle(image, keypoint_coordinates[index], radius, color, thickness)
    return image


def read_json_data(file_path):
    with open(file_path) as file:
        content = json.load(file)
    return content


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)


def get_image_index(image_path):
    filename = get_filename(image_path)
    index = int(filename.split('.')[0])
    return index


def get_filename(image_path):
    filename = image_path.split('/')[-1]
    return filename


def projectPoints(xyz, K):
    """ Project 3D coordinates provided by the hands dataset into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def get_actual_bbox_coordinates(coordinates):
    """
    Get bounding box coordinates for a actual hand landmark.
    """
    all_x, all_y = [], []
    for coordinate in coordinates:
        all_x.append(coordinate[0])
        all_y.append(coordinate[1])
    return min(all_x), min(all_y), max(all_x), max(all_y)
