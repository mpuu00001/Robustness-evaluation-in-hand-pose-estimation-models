import cv2
import numpy as np
from os import listdir
import csv


def apply_motion_blur_on_single_image(image):
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 20

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Create a copy of the same for creating the diagonally kernel.
    kernel_d = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    for i in range(len(kernel_d)):
        kernel_d[i][i] = 1

    # # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    kernel_d /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(image, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(image, -1, kernel_h)

    # Apply the diagonal kernel.
    diagonal_mb = cv2.filter2D(image, -1, kernel_d)
    return vertical_mb, horizonal_mb, diagonal_mb


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)


def apply_motion_blur(dataset='FreiHand'):
    input_folder = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_folder = f'/Users/muxin/PyCharm/{dataset}/transformed/motion_blur'

    for filename in listdir(input_folder):
        if filename[-3:] == 'jpg':
            try:
                print("Process file " + filename)
                # Read the image
                image_path = input_folder + f'/' + filename
                image = cv2.imread(image_path)
                # Apply motion blur on this image
                vertical_mb, horizonal_mb, diagonal_mb = apply_motion_blur_on_single_image(image)

                # Write new data
                output_path_vertical_mb = output_folder + f'/vertical/{filename}'
                cv2.imwrite(output_path_vertical_mb, vertical_mb)

                output_path_horizontal_mb = output_folder + f'/horizontal/{filename}'
                cv2.imwrite(output_path_horizontal_mb, horizonal_mb)

                output_path_diagonal_mb = output_folder + f'/diagonal/{filename}'
                cv2.imwrite(output_path_diagonal_mb, diagonal_mb)

            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')
        # break


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_motion_blur(dataset='FreiHand')
    apply_motion_blur(dataset='CMUhand')