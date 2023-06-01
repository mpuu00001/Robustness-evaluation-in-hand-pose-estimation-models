"""
Author: Pu Muxin
Date: 10/10/2022
"""
from functions import *
from os import listdir
from matplotlib import pyplot as plt

mp_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)


def single_hand_pose_estimation(image):
    # Perform hands landmarks detection
    results = mp_model.process(image)
    return results


def document_dataset_results(input_dir, output_path, dataset='FreiHand', size=244):
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

    # Prepare for result documentation
    output_csv_path = output_path + 'prediction_result.csv'
    write_csv_data(output_csv_path, format_csv_result(), "w")
    print("Writing result in folder: " + output_csv_path)

    # output_json_path = output_folder + 'predicted_coordinates.json'
    # write_json_data(output_json_path, {'predicted_data': []}, 'w')
    # feed_size = (244, 244)
    # Process prediction
    for filename in listdir(input_dir):
        if filename[-3:] == 'jpg':
            try:
                # Initialise result data
                iou, coordinates_variances, = -1, [-1] * 21
                # json_result = {'filename': filename, 'predicted_coordinates': []}
                print("Process file " + filename)
                # Read the image
                image_path = input_dir + f'/' + filename
                image = cv2.imread(image_path)
                # Record the original size
                # ori_size = image.shape[:-1]
                # Resize the image to be 244 x 244
                # image = cv2.resize(image, feed_size, interpolation=cv2.INTER_AREA)
                # Convert the BGR image to RGB before processing.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Perform hands landmarks detection
                results = single_hand_pose_estimation(image)

                # Get ground-truth hand landmark 2D coordinates
                if dataset == 'FreiHand':
                    index = get_image_index(filename)
                    actual_coordinates = projectPoints(xyz_array[index], k_array[index]).astype(np.int32)
                elif dataset == 'HandDB':
                    ground_truth = read_json_data(ground_truth_dir + f'/{filename[:-4]}.json')
                    actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in ground_truth['hand_pts']]
                elif dataset == 'CMUhand':
                    ground_truth = read_json_data(ground_truth_dir + f'/{filename[:-4]}.json')
                    actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in ground_truth['hand_pts']]
                else:
                    raise Exception('Error: Cannot find the given dataset')

                if results.multi_hand_landmarks:
                    # Project predicted 3D coordinates into 2D image space
                    predicted_coordinates = get_predicted_hand_landmarks(results, image)
                    if size != 244:
                        predicted_coordinates_unified = get_predicted_hand_landmarks(results, image)
                        predicted_coordinates = [[point[0] / size * 244, point[1] / size * 244] for point in
                                                 predicted_coordinates_unified]

                    # json_result['predicted_coordinates'] = predicted_coordinates

                    # Compute the area of both the prediction and ground-truth rectangles
                    # predicted_bbox = get_bbox_coordinates_mediapipe(results, image)
                    predicted_bbox = get_bbox_coordinates(predicted_coordinates)
                    actual_bbox = get_bbox_coordinates(actual_coordinates)

                    # compute intersection over union between the prediction and ground-truth rectangles
                    iou = compute_intersection_over_union(predicted_bbox, actual_bbox)

                    # Compute euclidean distances between the prediction and ground-truth hand landmarks
                    coordinates_variances = compute_euclidean_distances(predicted_coordinates, actual_coordinates)

                    # Annotated image and draw boundary box
                    # annotate_image(results, image, actual_coordinates)

                csv_result = tabulate_csv_result(filename, iou, coordinates_variances)
                # print(csv_result)
                write_csv_data(output_csv_path, csv_result, "a")
                # write_json_data(output_json_path, json_result, 'r+', 'predicted_data')
            except Exception as e:
                error_report_data = [filename, e]
                write_csv_data('error_report.csv', error_report_data, 'a')
            # break


def annotate_image(results, image, actual_coordinates):
    # Annotated image and draw boundary box
    annotated_image = draw_predicted_hand_landmarks(results, image.copy())
    annotated_image = draw_bbox(results, annotated_image.copy(), 0, (0, 255, 0), True)
    # annotated_image = draw_bbox(results, annotated_image.copy(), 0, (255, 0, 0), False, actual_coordinates)
    plt.imshow(annotated_image)
    plt.show()


def test_on_naked_hands(dataset='FreiHand'):
    print("-------------------naked_hands-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/original/naked_hands'
    output_path = f'result/{dataset}/baseline/raw/naked_hands_'
    document_dataset_results(input_dir=input_dir,
                             output_path=output_path, dataset=dataset)


def test_on_hands_with_objects(dataset='FreiHand'):
    print("-------------------hands_with_objects-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/original/hands_with_objects'
    output_path = f'result/{dataset}/baseline/raw/hands_with_objects_'
    document_dataset_results(input_dir=input_dir,
                             output_path=output_path, dataset=dataset)


def test_on_finger_occlusion(dataset='FreiHand'):
    print("-------------------finger_occlusion-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/finger_occlusion'
    output_path = f'result/{dataset}/finger_occlusion/raw'

    fingers = ['index_finger', 'middle_finger', 'pinky', 'ring_finger', 'thumb']

    for i in range(len(fingers)):
        new_input_dir = input_dir + f'/{fingers[i]}'
        new_output_path = output_path + f'/{fingers[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_regional_occlusion(dataset='FreiHand'):
    print("-------------------regional_occlusion-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/regional_occlusion'
    output_path = f'result/{dataset}/regional_occlusion/raw'

    category = ['bottom_5', 'top_5']

    for i in range(len(category)):
        new_input_dir = input_dir + f'/{category[i]}/'
        new_output_path = output_path + f'/{category[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_level_occlusion(occluded_by='black_circle', dataset='FreiHand'):
    print("-------------------level_occlusion-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/level_occlusion'
    output_path = f'result/{dataset}/level_occlusion/raw'

    for i in range(1, 22):
        new_input_dir = input_dir + f'/{occluded_by}/level{i}'
        new_output_path = output_path + f'/{occluded_by}/level{i}_occlusion_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_exposure_rate(dataset='FreiHand'):
    print("-------------------exposure_rate-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/exposure_rate'
    output_path = f'result/{dataset}/exposure_rate/raw'

    category = ['overexposed', 'strongly_overexposed', 'strongly_underexposed', 'underexposed']

    for i in range(len(category)):
        new_input_dir = input_dir + f'/{category[i]}/'
        new_output_path = output_path + f'/{category[i]}_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_motion_blur(dataset='FreiHand'):
    print("-------------------motion_blur-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/motion_blur'
    output_path = f'result/{dataset}/motion_blur/raw'

    category = ['diagonal', 'horizontal', 'vertical']

    for i in range(len(category)):
        new_input_dir = input_dir + f'/{category[i]}/'
        new_output_path = output_path + f'/{category[i]}_motion_blurred_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_scaling_factor(dataset='FreiHand'):
    print("-------------------scaling_factor-------------------")
    input_dir = f'/Users/muxin/PyCharm/{dataset}/transformed/scale'
    output_path = f'result/{dataset}/scale/raw'

    category = ['122', '488']

    for i in range(len(category)):
        new_input_dir = input_dir + f'/{category[i]}/'
        new_output_path = output_path + f'/{category[i]}_sized_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset, size=int(category[i]))


if __name__ == '__main__':
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')

    # print("-------------------FreiHand-------------------")
    # test_on_naked_hands(dataset='FreiHand')
    # test_on_hands_with_objects(dataset='FreiHand')
    # test_on_finger_occlusion(dataset='FreiHand')
    # test_on_regional_occlusion(dataset='FreiHand')
    # test_on_level_occlusion(dataset='FreiHand')
    # test_on_different_exposure_rate(dataset='FreiHand')
    # test_on_different_motion_blur(dataset='FreiHand')
    test_on_different_scaling_factor(dataset='FreiHand')

    # print("-------------------CMUhand-------------------")
    # # test_on_naked_hands(dataset='CMUhand')
    # test_on_finger_occlusion(dataset='CMUhand')
    # test_on_regional_occlusion(dataset='CMUhand')
    # test_on_level_occlusion(dataset='CMUhand')
    # test_on_different_exposure_rate(dataset='CMUhand')
    # test_on_different_motion_blur(dataset='CMUhand')
    # test_on_hands_with_objects(dataset='CMUhand')
    test_on_different_scaling_factor(dataset='CMUhand')

    # print("-------------------HandDB-------------------")
    # test_on_naked_hands(dataset='HandDB')
    # test_on_finger_occlusion(dataset='HandDB')
    # test_on_regional_occlusion(dataset='HandDB')
    # test_on_level_occlusion(dataset='HandDB')
