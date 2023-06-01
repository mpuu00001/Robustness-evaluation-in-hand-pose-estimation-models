"""
Author: Pu Muxin
Date: 11/3/2022
"""

from occlusion import *


def apply_finger_occlusion(input_folder, output_folder, dataset='FreiHand'):
    thumb = list(range(5))
    index_finger = [0] + list(range(5, 9))
    middle_finger = [0] + list(range(9, 13))
    ring_finger = [0] + list(range(13, 17))
    pinky = [0] + list(range(17, 21))
    fingers = [thumb, index_finger, middle_finger, ring_finger, pinky]
    finger_labels = ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky']

    for i in range(len(fingers)):
        new_output_folder = output_folder + '/' + finger_labels[i] + '/'
        occlude_keypoint(input_folder=input_folder, output_folder=new_output_folder,
                         occlude_on=fingers[i],
                         dataset=dataset)


def apply_on_dataset(dataset='FreiHand'):
    apply_finger_occlusion(input_folder=f'/Users/muxin/PyCharm/{dataset}/original/naked_hands',
                           output_folder=f'/Users/muxin/PyCharm/{dataset}/transformed/finger_occlusion',
                           dataset=dataset)


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_on_dataset(dataset='CMUhand')

