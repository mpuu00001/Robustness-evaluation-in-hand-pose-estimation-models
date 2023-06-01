"""
Author: Pu Muxin
Date: 31/10/2022
"""
from tab_functions import *


def document_evaluation_metric(input_dataframe, output_matrix_dir, output_result_dir, result_filename, type, threshold, is_hand_recognition,
                               group_by=None):
    """
    Document resulting evaluation_metric
    :param input_dataframe: a csv file contains prediction results from a hand landmark localisation model, each data must contains filename,
    the predicted iou value, mean keypoint error measured, 21 separated columns for 21 keypoint errors
    :param output_matrix_dir: the directory for storing the resulting confusion matrix
    :param output_result_dir: the directory for storing the resulting evaluation_metrices
    :param result_filename: the filename of the result
    :param type: the way or type of occlusion
    :param threshold: the error threshold used to determinate a correct prediction
    :param is_hand_recognition: if the current computation is for a hand recognition
    :param group_by: the group by factor of the output result
    """

    if group_by != 'keypoint':
        # Compute confusion matrix:
        if is_hand_recognition:
            confusion_matrix = form_confusion_matrix_for_hand_recognition_results(input_dataframe, 'iou', threshold, True)
        else:
            confusion_matrix = form_confusion_matrix_for_hand_landmark_localisation_results(input_dataframe, ['iou', 'mean_keypoint_error'],
                                                                                            threshold)
            confusion_matrix['label'] = confusion_matrix.index

        # Write confusion matrix
        title = f'result for {type} occluded hands:'
        write_csv_data(output_matrix_dir + result_filename, [title], 'a')
        write_csv_data(output_matrix_dir + result_filename, ['counts', 'label'], 'a')
        for i in range(len(confusion_matrix)):
            write_csv_data(output_matrix_dir + result_filename, confusion_matrix.iloc[i], 'a')
        write_csv_data(output_matrix_dir + result_filename, '\n', 'a')

        # Compute evaluation metrics
        precision, recall, f1_score = compute_evaluation_metric(confusion_matrix)
        result = [f'{type}', precision, recall, f1_score]

        # Write evaluation metric
        write_csv_data(output_result_dir + result_filename, result, 'a')

        # Display results
        print(title)
        print(confusion_matrix)
        print('precision: ' + str(precision) + '\n' + 'recall: ' + str(recall) + 'f1_score: ' + str(f1_score))
        print()

    else:
        # Compute confusion matrix:
        confusion_matrix = form_confusion_matrix_for_hand_landmark_localisation_results(input_dataframe, ['iou', 'mean_keypoint_error'],
                                                                                        threshold, 'keypoint')
        result = tabulate_evaluation_metric_by_keypoint(confusion_matrix)
        # Write evaluation metrics
        title = f'result for {type} occluded hands'
        for i in range(len(result)):
            write_csv_data(output_result_dir + result_filename, [f'{type}'] + result.iloc[i].tolist(), 'a')

        # Display results
        print(title)
        print(result)
        print()


def document_preparation(main_output_dir, result_filename, result_header):
    # Hand recognition analysis
    output_hand_recognition_result_path = main_output_dir + 'overall/hand_recognition/'
    output_hand_recognition_matrix_path = main_output_dir + 'confusion_matrixes/hand_recognition/'
    write_csv_data(output_hand_recognition_matrix_path + result_filename, '', 'w')
    write_csv_data(output_hand_recognition_result_path + result_filename, result_header, 'w')

    # Hand landmark localisation analysis (Use euclidean distances of 10 mm as error threshold)
    output_hand_landmark_localisation_result_path = main_output_dir + 'overall/hand_landmark_localisation/'
    output_hand_landmark_localisation_matrix_path = main_output_dir + 'confusion_matrixes/hand_landmark_localisation/'
    write_csv_data(output_hand_landmark_localisation_matrix_path + result_filename, '', 'w')
    write_csv_data(output_hand_landmark_localisation_result_path + result_filename, result_header, 'w')

    # Hand landmark localisation analysis group by keypoint
    output_hand_landmark_localisation_result_by_keypoint_path = main_output_dir + 'overall/hand_landmark_localisation_by_keypoint/'
    result_by_keypoint_header = [result_header[0]] + ['keypoint'] + result_header[1:]
    write_csv_data(output_hand_landmark_localisation_result_by_keypoint_path + result_filename, result_by_keypoint_header, 'w')


def document_single_occlusion_results(source_dir, main_output_dir, result_filename, result_header, type, combine=False, dataset='FreiHand', size=244):
    # Read data
    dataframe = pd.read_csv(source_dir)

    # Write the header of each result file
    if not combine:
        document_preparation(main_output_dir, result_filename, result_header)

    # Hand recognition analysis
    print("Document results for hand recognition:")
    output_hand_recognition_result_path = main_output_dir + 'overall/hand_recognition/'
    output_hand_recognition_matrix_path = main_output_dir + 'confusion_matrixes/hand_recognition/'
    hand_recognition_threshold = [-1, 0.5]
    document_evaluation_metric(input_dataframe=dataframe, output_matrix_dir=output_hand_recognition_matrix_path,
                               output_result_dir=output_hand_recognition_result_path, result_filename=result_filename, type=type,
                               threshold=hand_recognition_threshold, is_hand_recognition=True)

    print("Document results for hand localisation:")
    # Hand landmark localisation analysis (Use euclidean distances of 10 mm as error threshold)
    output_hand_landmark_localisation_result_path = main_output_dir + 'overall/hand_landmark_localisation/'
    output_hand_landmark_localisation_matrix_path = main_output_dir + 'confusion_matrixes/hand_landmark_localisation/'
    if dataset == 'FreiHand' or dataset == 'CMUhand':
        hand_localisation_threshold = [-1, 37.795]
    elif dataset == 'HandDB':
        hand_localisation_threshold = [-1, 57.002]
    else:
        raise Exception('Error: Cannot find the given dataset')
    if size != 244:
        hand_localisation_threshold[1] = hand_localisation_threshold[1] / 244 * size
        print(hand_localisation_threshold)

    document_evaluation_metric(input_dataframe=dataframe, output_matrix_dir=output_hand_landmark_localisation_matrix_path,
                               output_result_dir=output_hand_landmark_localisation_result_path, result_filename=result_filename,
                               type=type, threshold=hand_localisation_threshold, is_hand_recognition=False)

    print("Document results for hand localisation by keypoint:")
    # Hand landmark localisation analysis group by keypoint
    output_hand_landmark_localisation_result_by_keypoint_path = main_output_dir + 'overall/hand_landmark_localisation_by_keypoint/'
    document_evaluation_metric(input_dataframe=dataframe, output_matrix_dir=None,
                               output_result_dir=output_hand_landmark_localisation_result_by_keypoint_path,
                               result_filename=result_filename, type=type, threshold=hand_localisation_threshold,
                               is_hand_recognition=False, group_by='keypoint')


def document_baseline_results_naked_hands(dataset='FreiHand'):
    source_path = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/baseline/raw/naked_hands_prediction_result.csv'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/baseline/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'naked_hands_results.csv'

    document_preparation(main_output_dir, result_filename, result_header)

    document_single_occlusion_results(source_dir=source_path, main_output_dir=main_output_dir,
                                      result_filename=result_filename, result_header=result_header,
                                      type='naked_hands', combine=True, dataset=dataset)


def document_baseline_results_filtered_naked_hands():
    source_path = f'/Users/muxin/PyCharm/MediaPipe/result/CMUhand/baseline/raw/filtered_naked_hands_prediction_result.csv'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/CMUhand/baseline/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'filtered_naked_hands_results.csv'

    document_preparation(main_output_dir, result_filename, result_header)

    document_single_occlusion_results(source_dir=source_path, main_output_dir=main_output_dir,
                                      result_filename=result_filename, result_header=result_header,
                                      type='filtered_naked_hands', combine=True, dataset='CMUhand')


def document_baseline_results_filtered_hands_with_objects():
    source_path = f'/Users/muxin/PyCharm/MediaPipe/result/CMUhand/baseline/raw/filtered_hands_with_objects_prediction_result.csv'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/CMUhand/baseline/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'filtered_hands_with_objects_results.csv'

    document_preparation(main_output_dir, result_filename, result_header)

    document_single_occlusion_results(source_dir=source_path, main_output_dir=main_output_dir,
                                      result_filename=result_filename, result_header=result_header,
                                      type='filtered_hands_with_objects_results', combine=True, dataset='CMUhand')

def document_baseline_results_hands_with_objects(dataset='FreiHand'):
    source_path = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/baseline/raw/hands_with_objects_prediction_result.csv'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/baseline/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'hands_with_objects_results.csv'

    document_preparation(main_output_dir, result_filename, result_header)

    document_single_occlusion_results(source_dir=source_path, main_output_dir=main_output_dir,
                                      result_filename=result_filename, result_header=result_header,
                                      type='hands_with_objects', combine=True, dataset=dataset)


def document_level_occlusion_results(occluded_by='black_circle', dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/level_occlusion/raw/' + occluded_by + '/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/level_occlusion/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = f'{occluded_by}_occlusion_result.csv'
    num_levels = 21
    document_preparation(main_output_dir, result_filename, result_header)

    for i in range(1, num_levels+1):
        new_source_dir = source_dir + f'level{str(i)}_occlusion_prediction_result.csv'
        new_type = f'level {str(i)}'
        document_single_occlusion_results(source_dir=new_source_dir, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset)


def document_finger_occlusion_results(dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/finger_occlusion/raw/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/finger_occlusion/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'finger_occlusion_results.csv'
    document_preparation(main_output_dir, result_filename, result_header)

    finger_labels = ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky']
    for i in range(len(finger_labels)):
        new_source_dir = source_dir + f'{finger_labels[i]}_occluded_hands_prediction_result.csv'
        new_type = finger_labels[i]
        document_single_occlusion_results(source_dir=new_source_dir, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset)


def document_regional_occlusion_results(dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/regional_occlusion/raw/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/regional_occlusion/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'regional_occlusion_results.csv'
    document_preparation(main_output_dir, result_filename, result_header)

    region_labels = ['top_5', 'bottom_5']
    for i in range(len(region_labels)):
        new_source_path = source_dir + f'{region_labels[i]}_occluded_hands_prediction_result.csv'
        new_type = region_labels[i]
        document_single_occlusion_results(source_dir=new_source_path, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset)


def document_different_exposure_rate_results(dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/exposure_rate/raw/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/exposure_rate/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'different_exposure_rate_results.csv'
    document_preparation(main_output_dir, result_filename, result_header)

    labels = ['overexposed', 'strongly_overexposed', 'strongly_underexposed', 'underexposed']

    for i in range(len(labels)):
        new_source_path = source_dir + f'{labels[i]}_hands_prediction_result.csv'
        new_type = labels[i]
        document_single_occlusion_results(source_dir=new_source_path, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset)


def document_different_motion_blur_results(dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/motion_blur/raw/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/motion_blur/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'different_motion_blur_results.csv'
    document_preparation(main_output_dir, result_filename, result_header)

    labels = ['diagonal', 'horizontal', 'vertical']
    for i in range(len(labels)):
        new_source_path = source_dir + f'{labels[i]}_motion_blurred_hands_prediction_result.csv'
        new_type = labels[i]
        document_single_occlusion_results(source_dir=new_source_path, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset)


def document_different_scaling_factor(dataset='FreiHand'):
    source_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/scale/raw/'
    main_output_dir = f'/Users/muxin/PyCharm/MediaPipe/result/{dataset}/scale/'

    result_header = ['test_case', 'precision', 'recall', 'f1_score']
    result_filename = 'different_scaling_factor_results.csv'
    document_preparation(main_output_dir, result_filename, result_header)

    labels = ['122', '488']
    for i in range(len(labels)):
        new_source_path = source_dir + f'{labels[i]}_sized_hands_prediction_result.csv'
        new_type = labels[i]
        document_single_occlusion_results(source_dir=new_source_path, main_output_dir=main_output_dir,
                                          result_filename=result_filename, result_header=result_header,
                                          type=new_type, combine=True, dataset=dataset, size=int(labels[i]))
