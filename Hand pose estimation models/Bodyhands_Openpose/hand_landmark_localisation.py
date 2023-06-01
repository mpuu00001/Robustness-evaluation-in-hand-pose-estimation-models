"""
Author: Pu Muxin
Date: 29/11/2022
"""
import sys
import os

# import openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath('__file__'))
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/./bin/python/openpose/Release')
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/./x64/Release;' + dir_path + '/./bin;'
    import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
        'script in the right folder?')
    raise e

params = dict()
params["model_folder"] = "openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

# Starting OpenPose
op_wrapper = op.WrapperPython()
op_wrapper.configure(params)
op_wrapper.start()


def extend_hand_box(box, image_height, image_width, extend):
    x_min, y_min, x_max, y_max = box
    for i in range(extend):
        if x_min == 0 and x_max < image_width:
            x_max += 1
            extend += 1
        elif x_min > 0 and x_max == image_width:
            x_min -= 1
            extend += 1
        elif x_min <= 0 and x_max >= image_width:
            pass
        else:
            x_min, x_max = x_min - 1, x_max + 1

        if y_min == 0 and y_max < image_height:
            y_max += 1
            extend += 1
        elif y_min > 0 and y_max == image_height:
            y_min -= 1
            extend += 1
        elif y_min <= 0 and y_max >= image_height:
            pass
        else:
            y_min, y_max = y_min - 1, y_max + 1

    x_min = 0 if x_min < 0 else x_min
    y_min = 0 if y_min < 0 else y_min
    x_max = image_width if x_max > image_width else x_max
    y_max = image_height if y_max > image_height else y_max

    return x_min, y_min, x_max, y_max


def find_bounded_square(box):
    x_min, y_min, x_max, y_max = box
    x = (x_max - x_min) / 2 + x_min
    y = (y_max - y_min) / 2 + y_min
    extend = max(x_max - x_min, y_max - y_min) / 2
    new_x_min, new_y_min = x - extend, y - extend
    new_x_max, new_y_max = x + extend, y + extend
    return new_x_min, new_y_min, new_x_max, new_y_max


def convert_to_op_rectangle(box, image_height, image_width, extend=50):
    bounded_square = find_bounded_square(box)
    x_min, y_min, x_max, y_max = extend_hand_box(bounded_square, image_height, image_width, extend)
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    length = max(width, height)
    hand_rectangle = op.Rectangle(x_min, y_min, length, length)
    return hand_rectangle


def get_predicted_keypoint_coordinates(keypoint):
    coordinates = []
    for hand_list in keypoint:
        for hand in hand_list:
            this_hand = []
            for i in range(21):
                this_hand.append(list(hand[i, :2]))
            coordinates.append(this_hand)
    return coordinates


def localise_hand_landmark(image, hand_boxes):
    # Convert hand_boxes to openpose hands rectangles
    height, width = image.shape[:2]
    hand_rectangles = [
        [convert_to_op_rectangle(box, height, width, 30), convert_to_op_rectangle(box, height, width, 30)]
        for box in hand_boxes]

    # Create new datum
    datum = op.Datum()
    datum.cvInputData = image
    datum.handRectangles = hand_rectangles

    # Reformat results
    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
    keypoint = datum.handKeypoints
    keypoint_coordinates = get_predicted_keypoint_coordinates(keypoint)
    return keypoint_coordinates
