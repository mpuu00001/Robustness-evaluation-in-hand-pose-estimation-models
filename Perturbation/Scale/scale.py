import cv2
import numpy as np
from os import listdir
import csv


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)


def apply_scale(dataset='FreiHand'):
    input_folder = ''
    if dataset == 'FreiHand':
        input_folder = f'/Users/muxin/PyCharm/FreiHand/original/naked_hands'
    if dataset == 'CMUhand':
        input_folder = f'/Users/muxin/PyCharm/CMUhand/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/scale'

    for filename in listdir(input_folder):
        if filename[-3:] == 'jpg':
            try:
                print("Process file " + filename)
                # Read the image
                image_path = input_folder + f'/' + filename
                image = cv2.imread(image_path)
                # Scale down
                size_122 = cv2.resize(image, (122, 122), interpolation=cv2.INTER_AREA)
                # Scale up
                size_488 = cv2.resize(image, (488, 488), interpolation=cv2.INTER_CUBIC)

                # Write new data
                output_path_size_122 = output_folder + f'/122/{filename}'
                cv2.imwrite(output_path_size_122, size_122)

                output_path_size_488 = output_folder + f'/488/{filename}'
                cv2.imwrite(output_path_size_488, size_488)
            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')
        # break


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_scale(dataset='FreiHand')
    apply_scale(dataset='CMUhand')
