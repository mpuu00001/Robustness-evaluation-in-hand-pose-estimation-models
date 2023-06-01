"""
Author: Pu Muxin
Date: 12/6/2022
"""
import sys

sys.path.insert(0, '/Users/muxin/PyCharm/MediaPipe')
from functions import *
from os import listdir
import json


def preprocess():
    """Unify the size of images to be feed size, and change the ground truth hand landmark coordinates accordingly"""
    dataset_dir = r'/Users/muxin/PyCharm/CMUhand'
    input_dir = f'{dataset_dir}/original/ori_hands_with_objects'     # ori_naked_hands'
    output_image_dir = f'{dataset_dir}/original/hands_with_objects'  # naked_hands'
    output_coordinates_dir = f'{dataset_dir}/xyz'

    actual_coordinate_array = read_json_data(f'{dataset_dir}/labels.json')

    feed_size = (244, 244)
    for filename in listdir(input_dir):
        if filename[-3:] == 'jpg':
            try:
                print("Process file " + filename)
                # Read the image
                image_path = input_dir + f'/' + filename
                image = cv2.imread(image_path)
                # Record the original size
                ori_size = image.shape[:-1]
                # Resize the image to be 244 x 244
                image = cv2.resize(image, feed_size, interpolation=cv2.INTER_AREA)
                # Save image
                cv2.imwrite(output_image_dir + f'/' + filename, image)

                # print('feed_size: ' + str(feed_size))
                # print('ori_size: ' + str(ori_size))
                # Change the ground truth hand landmark coordinates accordingly
                actual_coordinates_ori = actual_coordinate_array[filename]
                actual_coordinates = [[point[0] / ori_size[0] * feed_size[0], point[1] / ori_size[1] * feed_size[1]] for point in
                                      actual_coordinates_ori]

                ground_truth = {
                    "hand_pts": actual_coordinates
                }

                # Serializing json
                json_object = json.dumps(ground_truth)

                # Writing to sample.json
                with open(output_coordinates_dir + f'/{filename[:-4]}.json', "w") as outfile:
                    outfile.write(json_object)
            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    preprocess()
