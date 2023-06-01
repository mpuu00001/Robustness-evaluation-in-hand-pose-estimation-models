"""
Author: Pu Muxin
Date: 10/10/2022
"""
import cv2
import json
import math
import csv
import numpy as np
#import mediapipe as mp

#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_hands = mp.solutions.hands

def get_predicted_hand_landmarks(results, image, includes_z=False):
    """
    Get hand landmark from image space
    """
    coordinates = []
    image_height, image_width, _ = image.shape
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        for i in range(len(mp_hands.HandLandmark)):
            coordinate = [hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width,
                          hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height]
            if includes_z:
                coordinate.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z)

            coordinates.append(coordinate)
    return coordinates


def draw_predicted_hand_landmarks(results, image):
    """
    Draw hand landmarks on the image.
    """
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
    return image


def get_bbox_coordinates_mediapipe(results, image):
    """
    Get bounding box coordinates for a predicted hand landmark.
    """
    all_x, all_y = [], []
    image_height, image_width, _ = image.shape
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        for i in mp_hands.HandLandmark:
            all_x.append(int(hand_landmarks.landmark[i].x * image_width))
            all_y.append(int(hand_landmarks.landmark[i].y * image_height))
    return min(all_x), min(all_y), max(all_x), max(all_y)


def get_bbox_coordinates(coordinates):
    """
    Get bounding box coordinates for a actual hand landmark.
    """
    all_x, all_y = [], []
    for coordinate in coordinates:
        all_x.append(coordinate[0])
        all_y.append(coordinate[1])
    return min(all_x), min(all_y), max(all_x), max(all_y)


def draw_bbox(results, image, extend, colour, is_predicted, coordinates=None):
    if is_predicted:
        x_min, y_min, x_max, y_max = get_predicted_bbox_coordinates(results, image)
    else:
        x_min, y_min, x_max, y_max = get_actual_bbox_coordinates(coordinates)
    cv2.rectangle(image, (x_min - extend, y_min - extend), (x_max + extend, y_max + extend), colour, 1)
    return image


def projectPoints(xyz, K):
    """ Project 3D coordinates provided by the hands dataset into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def display_world_landmarks(results):
    for hand_world_landmarks in results.multi_hand_world_landmarks:
        mp_drawing.plot_landmarks(
            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


def get_image_index(image_path):
    filename = get_filename(image_path)
    index = int(filename.split('.')[0])
    return index


def get_filename(image_path):
    filename = image_path.split('/')[-1]
    return filename


def compute_euclidean_distances(pred_coordinates, actu_coordinates):
    variances = []
    for i in range(len(actu_coordinates)):
        variance = compute_single_euclidean_distances(pred_coordinates[i], actu_coordinates[i])
        variances.append(variance)
    return variances


def compute_single_euclidean_distances(pred_coordinate, actu_coordinate):
    variance = math.sqrt(pow(pred_coordinate[0] - actu_coordinate[0], 2) +
                         pow(pred_coordinate[1] - actu_coordinate[1], 2))
    return variance


def compute_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def format_csv_result():
    header = ['filename', 'iou', 'mean_keypoint_error']
    for i in range(21):
        header.append("keypoint_error_" + str(i))
    return header


def tabulate_csv_result(filename, iou, coordinates_variances):
    coordinates_mean_error = sum(coordinates_variances) / len(coordinates_variances)
    data = [filename, iou, coordinates_mean_error]
    for each in coordinates_variances:
        data.append(each)
    return data


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)


def read_json_data(file_path):
    with open(file_path) as file:
        content = json.load(file)
    return content


def write_json_data(file_path, new_data, mode, key=None):
    if mode == 'w':
        with open(file_path, 'w') as file:
            json.dump(new_data, file)
    else:
        with open(file_path, 'r+') as file:
            content = json.load(file)
            content[key].append(new_data)
            file.seek(0)
            json.dump(content, file, indent=4)


def write_txt_file(file_path, new_data, mode):
    with open(file_path, mode) as file:
        file.write(new_data)
