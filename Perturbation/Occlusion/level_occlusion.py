"""
Author: Pu Muxin
Date: 10/26/2022
"""

from occlusion import *


def apply_level_occlusion(input_folder, output_folder, dataset='FreiHand'):
    for i in range(1, 22):
        new_output_folder = output_folder + f'/level{str(i)}/'
        occlude_keypoint(input_folder=input_folder, output_folder=new_output_folder,
                         occlude_on=range(i),
                         dataset=dataset)


def apply_on_dataset(dataset='FreiHand'):
    apply_level_occlusion(input_folder=f'/Users/muxin/PyCharm/{dataset}/original/naked_hands',
                          output_folder=f'/Users/muxin/PyCharm/{dataset}/transformed/level_occlusion/black_circle',
                          dataset=dataset)


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_on_dataset(dataset='CMUhand')

