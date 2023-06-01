import sys
sys.path.insert(0, '/Users/muxin/PyCharm/MediaPipe/result_analysis')
from result_tabulation import *

if __name__ == '__main__':
    # print("-------------------FreiHand-------------------")
    # document_baseline_results_naked_hands(dataset='FreiHand')
    # document_baseline_results_hands_with_objects(dataset='FreiHand')
    # document_finger_occlusion_results(dataset='FreiHand')
    # document_regional_occlusion_results(dataset='FreiHand')
    # document_level_occlusion_results(dataset='FreiHand')
    # document_different_exposure_rate_results(dataset='FreiHand')
    # document_different_motion_blur_results(dataset='FreiHand')
    # document_different_scaling_factor(dataset='FreiHand')

    # print("-------------------CMUhand-------------------")
    # document_baseline_results_naked_hands(dataset='CMUhand')
    # document_baseline_results_hands_with_objects(dataset='CMUhand')
    # document_finger_occlusion_results(dataset='CMUhand')
    # document_regional_occlusion_results(dataset='CMUhand')
    # document_level_occlusion_results(dataset='CMUhand')
    # document_different_exposure_rate_results(dataset='CMUhand')
    # document_different_motion_blur_results(dataset='CMUhand')
    # document_baseline_results_filtered_naked_hands()
    # document_different_scaling_factor(dataset='CMUhand')
    document_baseline_results_filtered_hands_with_objects()

    # print("-------------------HandDB-------------------")
    # document_baseline_results_naked_hands(dataset='HandDB')
    # document_finger_occlusion_results(dataset='HandDB')
    # document_regional_occlusion_results(dataset='HandDB')
    # document_level_occlusion_results(dataset='HandDB')