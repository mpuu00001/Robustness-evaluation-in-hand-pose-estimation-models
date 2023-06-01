import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from model.cpm_limb import CPMHandLimb
from PIL import Image, ImageDraw
from os import listdir, path
import sys

sys.path.insert(0, '../')
from functions import *

cuda = torch.cuda.is_available()
device_id = [0]
torch.cuda.set_device(device_id[0])
model = CPMHandLimb(outc=21, lshc=7, pretrained=True)
if cuda:
    model = model.cuda()
    model = nn.DataParallel(model, device_id)
state_dict = torch.load('best_model.pth', map_location='cuda:0')
model.load_state_dict(state_dict)


def load_image(img_path):
    ori_im = Image.open(img_path)
    ori_w, ori_h = ori_im.size
    im = ori_im.resize((368, 368))
    image = transforms.ToTensor()(im)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image)  # (C,H,W)
    image = image.unsqueeze(0)  # (1,C,H,W)
    return ori_im, image, ori_w, ori_h


def get_image_coordinate(pred_map, ori_w, ori_h):
    """
    decode heatmap of one image to coordinates
    :param pred_map: Tensor  CPU     size:(1, 21, 46, 46)
    :return:
    label_list: Type:list, Length:21,  element: [x,y]
    """
    pred_map = pred_map.squeeze(0)
    label_list = []
    for k in range(21):
        tmp_pre = np.asarray(pred_map[k, :, :])  # 2D array  size:(46,46)
        corr = np.where(tmp_pre == np.max(tmp_pre))  # coordinate of keypoints in 46 * 46 scale

        # get coordinate of keypoints in origin image scale
        x = int(corr[1][0] * (int(ori_w) / 46.0))
        y = int(corr[0][0] * (int(ori_h) / 46.0))
        label_list.append([x, y])
    return label_list


def draw_point(points, im):
    i = 0
    draw = ImageDraw.Draw(im)

    for point in points:
        x = point[0]
        y = point[1]

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im


def single_hand_pose_estimation(img_path='images/sample.jpg', save_path='images/sample_out_new.jpg'):
    with torch.no_grad():
        ori_im, img, ori_w, ori_h = load_image(img_path)
        if cuda:
            img = img.cuda()  # # Tensor size:(1,3,368,368)
        _, cm_pred = model(img)
        # limb_pred (FloatTensor.cuda) size:(bz,3,C,46,46)
        # cm_pred   (FloatTensor.cuda) size:(bz,3,21,46,46)

        coordinates = get_image_coordinate(cm_pred[:, -1].cpu(), ori_w, ori_h)
        # Type: list,   Length:21,      element:[x,y]
        # ori_im = draw_point(coordinates, ori_im)
        # print('save output to ', save_path)
        # ori_im.save(save_path)
        return coordinates


def document_dataset_results(input_dir, output_path, dataset='FreiHand', size=244):
    if dataset == 'FreiHand':
        dataset_dir = r'D:\PuMuxin\FreiHand'
        xyz_array = read_json_data(f'{dataset_dir}\\training_xyz.json')
        k_array = read_json_data(f'{dataset_dir}\\training_k.json')
        ground_truth_dir = None
    elif dataset == 'HandDB':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'D:\PuMuxin\HandDB\xyz'
    elif dataset == 'CMUhand':
        xyz_array, k_array, = [], []
        ground_truth_dir = r'D:\PuMuxin\CMUhand\xyz'
    else:
        raise Exception('Error: Cannot find the given dataset')

    # Prepare for result documentation
    output_csv_path = output_path + 'prediction_result.csv'
    write_csv_data(output_csv_path, format_csv_result(), "w")
    print("Writing result in folder: " + output_csv_path)

    for filename in listdir(input_dir):  # , result_dir):
        print("Process file " + filename)
        try:
            if filename[-3:] == 'jpg':
                # Initialise result data
                iou, coordinates_variances, pre_coordinates = -1, [-1] * 21, []
                # Process estimation
                pre_coordinates = single_hand_pose_estimation(img_path=f'{input_dir}\\{filename}')
                if len(pre_coordinates) == 21:
                    if size != 244:
                        pre_coordinates = [[point[0] / size * 244, point[1] / size * 244] for point in pre_coordinates]

                    # Get ground-truth hand landmark 2D coordinates
                    if dataset == 'FreiHand':
                        index = get_image_index(filename)
                        actual_coordinates = projectPoints(xyz_array[index], k_array[index]).astype(np.int32)
                    elif dataset == 'HandDB':
                        ground_truth = read_json_data(ground_truth_dir + '\\' + f'{filename[:-4]}.json')
                        actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in
                                              ground_truth['hand_pts']]
                    elif dataset == 'CMUhand':
                        ground_truth = read_json_data(ground_truth_dir + '\\' + f'{filename[:-4]}.json')
                        actual_coordinates = [[coordinate[i] for i in range(2)] for coordinate in
                                              ground_truth['hand_pts']]
                    else:
                        raise Exception('Error: Cannot find the given dataset')
                    actual_bbox = get_bbox_coordinates(actual_coordinates)

                    # Compute the intersection of union for the prediction and ground-truth rectangles
                    predicted_bbox = get_bbox_coordinates(pre_coordinates)
                    iou = compute_intersection_over_union(predicted_bbox, actual_bbox)
                    # Compute euclidean distances between prediction and baseline
                    coordinates_variances = compute_euclidean_distances(pre_coordinates, actual_coordinates)
                csv_result = tabulate_csv_result(filename, iou, coordinates_variances)
                write_csv_data(output_csv_path, csv_result, "a")
                # print('csv_result:' + str(csv_result))
        except Exception as e:
            error_report_data = [filename, e]
            write_csv_data('error_report.csv', error_report_data, 'a')
        # break


def test_on_naked_hands(dataset='FreiHand'):
    print("-------------------naked_hands-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\original\\naked_hands'
    output_path = f'result\\{dataset}\\baseline\\raw\\naked_hands_'

    document_dataset_results(input_dir=input_dir,
                             output_path=output_path,
                             dataset=dataset)


def test_on_hands_with_objects(dataset='FreiHand'):
    print("-------------------hands_with_objects-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\original\\hands_with_objects'
    output_path = f'result\\{dataset}\\baseline\\raw\\hands_with_objects_'
    document_dataset_results(input_dir=input_dir,
                             output_path=output_path)


def test_on_finger_occlusion(dataset='FreiHand'):
    print("-------------------finger_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\finger_occlusion'
    output_path = f'result\\{dataset}\\finger_occlusion\\raw'

    fingers = ['index_finger', 'middle_finger', 'pinky', 'ring_finger', 'thumb']

    for i in range(len(fingers)):
        new_input_dir = input_dir + f'\\{fingers[i]}'
        new_output_path = output_path + f'\\{fingers[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_regional_occlusion(dataset='FreiHand'):
    print("-------------------regional_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\regional_occlusion'
    output_path = f'result\\{dataset}\\regional_occlusion\\raw'

    category = ['bottom_5', 'top_5']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_occluded_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_level_occlusion(occluded_by='black_circle', dataset='FreiHand'):
    print("-------------------level_occlusion-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\level_occlusion'
    output_path = f'result\\{dataset}\\level_occlusion\\raw'

    for i in range(1, 22):
        new_input_dir = input_dir + f'\\{occluded_by}\\level{i}'
        new_output_path = output_path + f'\\{occluded_by}\\level{i}_occlusion_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_exposure_rate(dataset='FreiHand'):
    print("-------------------exposure_rate-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\exposure_rate'
    output_path = f'result\\{dataset}\\exposure_rate\\raw'

    category = ['strongly_underexposed', 'underexposed', 'overexposed', 'strongly_overexposed']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_motion_blur(dataset='FreiHand'):
    print("-------------------motion_blur-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\motion_blur'
    output_path = f'result\\{dataset}\\motion_blur\\raw'

    category = ['diagonal', 'horizontal', 'vertical']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_motion_blurred_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset)


def test_on_different_scaling_factor(dataset='FreiHand'):
    print("-------------------scaling_factorr-------------------")
    input_dir = f'D:\\PuMuxin\\{dataset}\\transformed\\scale'
    output_path = f'result\\{dataset}\\scale\\raw'

    category = ['122', '488']

    for i in range(len(category)):
        new_input_dir = input_dir + f'\\{category[i]}'
        new_output_path = output_path + f'\\{category[i]}_sized_hands_'
        document_dataset_results(input_dir=new_input_dir, output_path=new_output_path, dataset=dataset,
                                 size=int(category[i]))


if __name__ == "__main__":
    write_csv_data('error_report.csv', ["filename", "exception"], 'w')

    print("-------------------FreiHand-------------------")
    # test_on_hands_with_objects()
    # test_on_naked_hands(dataset='FreiHand')
    # test_on_finger_occlusion(dataset='FreiHand')
    # test_on_regional_occlusion(dataset='FreiHand')
    # test_on_level_occlusion(dataset='FreiHand')
    # test_on_different_motion_blur(dataset='FreiHand')
    # test_on_different_exposure_rate(dataset='FreiHand')
    test_on_different_scaling_factor(dataset='FreiHand')

    print("-------------------CMUhand-------------------")
    # test_on_naked_hands(dataset='CMUhand')
    # test_on_finger_occlusion(dataset='CMUhand')
    # test_on_regional_occlusion(dataset='CMUhand')
    # test_on_level_occlusion(dataset='CMUhand')
    # test_on_hands_with_objects(dataset='CMUhand')
    # test_on_different_motion_blur(dataset='CMUhand')
    # test_on_different_exposure_rate(dataset='CMUhand')
    test_on_different_scaling_factor(dataset='CMUhand')

    # print("-------------------HandDB-------------------")
    # test_on_naked_hands(dataset='HandDB')
    # test_on_finger_occlusion(dataset='HandDB')
    # test_on_regional_occlusion(dataset='HandDB')
    # test_on_level_occlusion(dataset='HandDB')
