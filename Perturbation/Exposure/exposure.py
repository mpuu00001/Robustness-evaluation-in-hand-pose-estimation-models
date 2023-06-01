import numpy as np
import cv2
import csv
from os import listdir


def adjust_gamma_for_single_image(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)


def apply_adjustment(input_folder, output_folder, gamma):
    for filename in listdir(input_folder):
        if filename[-3:] == 'jpg':
            try:
                print("Process file " + filename)
                # Read the image
                image_path = input_folder + f'/' + filename
                image = cv2.imread(image_path)
                # Change exposure rate of this image
                image = adjust_gamma_for_single_image(image, gamma)
                # Write new data
                output_path = output_folder + f'/' + filename
                cv2.imwrite(output_path, image)
            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')
        # break


gamma_values = [0.2, 0.5, 1, 2, 5]  # [0.2, 0.5, 1, 2, 5]
titles = ['Strongly underexposed', 'Underexposed', 'Correct exposure', 'Overexposed', 'Strongly overexposed']


def apply_strongly_underexposed(dataset='FreiHand'):
    input_folder = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/exposure_rate/strongly_underexposed'
    apply_adjustment(input_folder, output_folder, gamma_values[0])


def apply_underexposed(dataset='FreiHand'):
    input_folder = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/exposure_rate/underexposed'
    apply_adjustment(input_folder, output_folder, gamma_values[1])


def apply_overexposed(dataset='FreiHand'):
    input_folder = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/exposure_rate/overexposed'
    apply_adjustment(input_folder, output_folder, gamma_values[3])


def apply_strongly_overexposed(dataset='FreiHand'):
    input_folder = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/exposure_rate/strongly_overexposed'
    apply_adjustment(input_folder, output_folder, gamma_values[4])


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_strongly_underexposed(dataset='FreiHand')
    apply_underexposed(dataset='FreiHand')
    apply_overexposed(dataset='FreiHand')
    apply_strongly_overexposed(dataset='FreiHand')

    apply_strongly_underexposed(dataset='CMUhand')
    apply_underexposed(dataset='CMUhand')
    apply_overexposed(dataset='CMUhand')
    apply_strongly_overexposed(dataset='CMUhand')
