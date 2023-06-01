"""
Author: Pu Muxin
Date: 11/3/2022
"""

from occlusion import *


def apply_regional_occlusion(input_folder, output_folder, dataset='FreiHand'):
    top_5 = [0, 6, 10, 14, 17]
    bottom_5 = [5, 19, 9, 20, 13]
    regions = [top_5, bottom_5]
    region_labels = ['top_5', 'bottom_5']

    for i in range(len(regions)):
        new_output_folder = output_folder + '/' + region_labels[i] + '/'
        print(region_labels[i])
        print(new_output_folder)
        occlude_keypoint(input_folder=input_folder, output_folder=new_output_folder,
                         occlude_on=regions[i],
                         dataset=dataset)


def apply_on_dataset(dataset='FreiHand'):
    apply_regional_occlusion(input_folder=f'/Users/muxin/PyCharm/{dataset}/original/naked_hands',
                             output_folder=f'/Users/muxin/PyCharm/{dataset}/transformed/regional_occlusion',
                             dataset=dataset)


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')
    apply_on_dataset(dataset='CMUhand')

