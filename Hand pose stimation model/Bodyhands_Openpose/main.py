"""
Author: Pu Muxin
Date: 29/11/2022
"""
import sys
import math
import cv2
from os import listdir
from hand_rectangle_detection import detect_hand_rectangle
from hand_landmark_localisation import localise_hand_landmark

sys.path.insert(0, '../')
from functions import *


def single_hand_pose_estimation(image_path):
    image = cv2.imread(image_path)
    hand_bbox = detect_hand_rectangle(image)
    if len(hand_bbox) == 0:
        return None
    else:
        keypoint_coordinates = localise_hand_landmark(image, hand_bbox)
        return keypoint_coordinates


def document_dataset_results(input_dir, output_path, dataset='FreiHand', size=244):
    if dataset == 'FreiHand':
        dataset_dir = r'D:\PuMuxin\FreiHand'
        xyz_array = read_json_data(f'{dataset_dir}\\training_xyz.json')
        k_array = read_json_data(f'{dataset_dir}\\training_k.json')
        ground_truth_dir = None
    elif dataset == 'HandDB':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'D:\PuMuxin\HandDB\xyz'
    elif dataset == 'CMUhand':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'D:\PuMuxin\CMUhand\xyz'
    else:
        raise Exception('Error: Cannot find the given dataset')

    # Prepare for result documentation
    output_csv_path = output_path + 'prediction_result.csv'
    write_csv_data(output_csv_path, format_csv_result(), "w")
    print("Writing result in folder: " + output_csv_path)

    for filename in listdir(input_dir):  # , result_dir):
        try:
            if filename[-3:] == 'jpg':
                print("Process file " + filename)
                # Initialise result data
                iou, coordinates_variances, pre_coordinates = -1, [-1] * 21, []
                # Process estimation
                pre_coordinates = single_hand_pose_estimation(image_path=f'{input_dir}\\{filename}')
                if pre_coordinates is not None:

                    # Get ground-truth hand landmark 2D coordinates
                    if dataset == 'FreiHand':
                        index = get_image_index(filename)
                        actual_coordinates = projectPoints(xyz_array[index], k_array[index]).astype(np.int32)
                    elif dataset == 'HandDB':
                        ground_truth = read_json_data(ground_truth_dir + '\\' + f'{filename[:-4]}.json')
                        actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in
                                              ground_truth['hand_pts']]
                    elif dataset == 'CMUhand':
                        ground_truth = read_json_data(ground_truth_dir + '\\' + f'{filename[:-4]}.json')
                        actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in
                                              ground_truth['hand_pts']]
                    else:
                        raise Exception('Error: Cannot find the given dataset')
                    actual_bbox = get_bbox_coordinates(actual_coordinates)

                    coordinates_variances = [math.inf] * 21
                    for pre_coordinate in pre_coordinates:
                        if size != 244:
                            pre_coordinate = [[point[0] / size * 244, point[1] / size * 244] for point in
                                              pre_coordinate]
                        # Compute euclidean distances between prediction and baseline
                        new_coordinates_variances = compute_euclidean_distances(pre_coordinate, actual_coordinates)
                        old_variances_mean = sum(coordinates_variances) / len(coordinates_variances)
                        new_variances_mean = sum(new_coordinates_variances) / len(new_coordinates_variances)
                        if new_variances_mean < old_variances_mean:
                            coordinates_variances = new_coordinates_variances
                            # Compute the intersection of union for the prediction and ground-truth rectangles
                            predicted_bbox = get_bbox_coordinates(pre_coordinate)
                            iou = compute_intersection_over_union(predicted_bbox, actual_bbox)

                csv_result = tabulate_csv_result(filename, iou, coordinates_variances)
                write_csv_data(output_csv_path, csv_result, "a")
                # print('csv_result:' + str(csv_result))
        except Exception as e:
            error_report_data = [filename, e]
            write_csv_data('error_report.csv', error_report_data, 'a')
        # break


def test_on_naked_hands(dataset='FreiHand'):
    print("-------------------naked_hands-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\original\\naked_hands'
    output_path = f'result\\{dataset}\\baseline\\raw\\naked_hands_'

    document_dataset_results(input_dir=input_dir,
                             output_path=output_path,
                             dataset=dataset)


def test_on_hands_with_objects(dataset='FreiHand'):
    print("-------------------hands_with_objects-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\original\\hands_with_objects'
    output_path = f'result\\{dataset}\\baseline\\raw\\hands_with_objects_'
    document_dataset_results(input_dir=input_dir,
                             output_path=output_path)


def test_on_finger_occlusion(dataset='FreiHand'):
    print("-------------------finger_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\finger_occlusion'
    output_path = f'result\\{dataset}\\finger_occlusion\\raw'

    fingers = ['index_finger', 'middle_finger', 'pinky', 'ring_finger', 'thumb']

    for i in range(len(fingers)):
        new_input_dir = input_dir + f'\\{fingers[i]}'
        new_output_path = output_path + f'\\{fingers[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_regional_occlusion(dataset='FreiHand'):
    print("-------------------regional_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\regional_occlusion'
    output_path = f'result\\{dataset}\\regional_occlusion\\raw'

    category = ['bottom_5', 'top_5']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_level_occlusion(occluded_by='black_circle', dataset='FreiHand'):
    print("-------------------level_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\level_occlusion'
    output_path = f'result\\{dataset}\\level_occlusion\\raw'

    for i in range(1, 22):
        new_input_dir = input_dir + f'\\{occluded_by}\\level{i}'
        new_output_path = output_path + f'\\{occluded_by}\\level{i}_occlusion_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_exposure_rate(dataset='FreiHand'):
    print("-------------------exposure_rate-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\exposure_rate'
    output_path = f'result\\{dataset}\\exposure_rate\\raw'

    category = ['strongly_underexposed', 'underexposed', 'overexposed', 'strongly_overexposed']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_motion_blur(dataset='FreiHand'):
    print("-------------------motion_blur-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\motion_blur'
    output_path = f'result\\{dataset}\\motion_blur\\raw'

    category = ['diagonal', 'horizontal', 'vertical']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_motion_blurred_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_scaling_factor(dataset='FreiHand'):
    print("-------------------scaling_factor-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\scale'
    output_path = f'result\\{dataset}\\scale\\raw'

    category = ['122', '488']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_sized_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset,
                                 size=int(category[i]))


if __name__ == "__main__":
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')

    print("-------------------FreiHand-------------------")
    # test_on_hands_with_objects()
    # test_on_naked_hands(dataset='FreiHand')
    # test_on_finger_occlusion(dataset='FreiHand')
    # test_on_regional_occlusion(dataset='FreiHand')
    # test_on_level_occlusion(dataset='FreiHand')
    # test_on_different_motion_blur(dataset='FreiHand')
    # test_on_different_exposure_rate(dataset='FreiHand')
    test_on_different_scaling_factor(dataset='FreiHand')

    print("-------------------CMUhand-------------------")
    # test_on_naked_hands(dataset='CMUhand')
    # test_on_finger_occlusion(dataset='CMUhand')
    # test_on_regional_occlusion(dataset='CMUhand')
    # test_on_level_occlusion(dataset='CMUhand')
    # test_on_hands_with_objects(dataset='CMUhand')
    # test_on_different_motion_blur(dataset='CMUhand')
    # test_on_different_exposure_rate(dataset='CMUhand')
    test_on_different_scaling_factor(dataset='CMUhand')

    # print("-------------------HandDB-------------------")
    # test_on_naked_hands(dataset='HandDB')
    # test_on_finger_occlusion(dataset='HandDB')
    # test_on_regional_occlusion(dataset='HandDB')
    # test_on_level_occlusion(dataset='HandDB')
