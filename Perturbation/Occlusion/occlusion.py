"""
Author: Pu Muxin
Date: 10/26/2022
"""
from os import listdir
from background_colour_detector import BackgroundColorDetector
from functions import *


def occlude_keypoint(input_folder, output_folder, occlude_on, color=(0, 0, 0), radius=10, use_circle=True, dataset='FreiHand'):
    print("Process occlusion on points: " + str(list(occlude_on)))

    # Read ground-truth data
    if dataset == 'FreiHand':
        dataset_dir = r'/Users/muxin/PyCharm/FreiHAND'
        xyz_array = read_json_data(f'{dataset_dir}/training_xyz.json')
        k_array = read_json_data(f'{dataset_dir}/training_k.json')
        ground_truth_dir = None
    elif dataset == 'HandDB':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'/Users/muxin/PyCharm/HandDB/xyz'
    elif dataset == 'CMUhand':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'/Users/muxin/PyCharm/CMUhand/xyz'
    else:
        raise Exception('Error: Cannot find the given dataset')

    feed_size = (244, 244)

    set_up = False
    for filename in listdir(input_folder):
        if filename[-3:] == 'jpg':
            try:
                print("Process file " + filename)
                # Read the image
                image_path = input_folder + f'/' + filename
                image = cv2.imread(image_path)

                # Get ground-truth hand landmark 2D coordinates
                if dataset == 'FreiHand':
                    index = get_image_index(filename)
                    actual_coordinates = projectPoints(xyz_array[index], k_array[index]).astype(np.int32)
                elif dataset == 'HandDB':
                    ground_truth = read_json_data(ground_truth_dir + f'/{filename[:-4]}.json')
                    actual_coordinates = [[int(coordinate[i]) for i in range(2)] for coordinate in ground_truth['hand_pts']]
                elif dataset == 'CMUhand':
                    ground_truth = read_json_data(ground_truth_dir + f'/{filename[:-4]}.json')
                    actual_coordinates = [[int(coordinate[i]) for i in range(2)] for coordinate in ground_truth['hand_pts']]
                else:
                    raise Exception('Error: Cannot find the given dataset')

                # Occlude with shape of background colour
                if color is None:
                    background_color_detector = BackgroundColorDetector(image_path)
                    color = background_color_detector.detect()

                ori_size = image.shape[:-1]
                if not set_up:
                    radius = int(10 / feed_size[0] * ori_size[0])
                    set_up = True
                    print('radius: ' + str(radius))

                # Occlude on the given keypoint
                thickness = -1
                if use_circle:
                    for index in occlude_on:
                        image = cv2.circle(image, actual_coordinates[index], radius, color, thickness)
                else:
                    image = occlude_keypoint_by_rectangle(image, color, actual_coordinates, occlude_on, radius // 2)

                # Write partially occlude data
                output_path = output_folder + filename
                cv2.imwrite(output_path, image)

            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')
        # break
