"""
Author: Pu Muxin
Date: 10/30/2022
"""

import pandas as pd
import csv
import sklearn.metrics


def form_confusion_matrix_for_hand_recognition_results(df, colum, threshold, is_hand_recognition):
    new_df = df[["filename", colum]]
    new_df['hand_recognise_result'] = new_df.apply(
        lambda row: compute_prediction_result(row, colum, threshold, is_hand_recognition), axis=1)
    confusion_matrix = new_df['hand_recognise_result'].value_counts().rename_axis('label').reset_index(name='counts')
    return confusion_matrix


def form_confusion_matrix_for_hand_landmark_localisation_results(df, droped_colums, threshold, group_by=None):
    new_df = df.drop(droped_colums, axis=1)
    new_df = tabulate_hand_localisation_result(new_df, 'keypoint_error', 'keypoint_prediction_result', threshold)
    if group_by == 'keypoints':
        confusion_matrix = group_confusion_matrix_by_keypoint(new_df, 'keypoint_prediction_result')
    else:
        confusion_matrix = new_df.drop(['filename'], axis=1).stack().value_counts().to_frame()
        confusion_matrix.rename({0: 'counts'}, axis=1, inplace=True)
    return confusion_matrix


def compute_prediction_result(row, column, threshold, is_hand_recognise):
    if float(row[column]) > threshold[1]:
        return 'TP' if is_hand_recognise else 'FP'
    elif float(row[column]) <= threshold[1] and float(row[column]) > threshold[0]:
        return 'FP' if is_hand_recognise else 'TP'
    else:
        return 'FN'


def compute_binary_prediction_result(value, threshold):
    if float(value) <= threshold[1] and float(value) > threshold[0]:
        return True
    else:
        return False


def precision_recall_curve_from(y_true, pred_value, thresholds):
    precisions = []
    recalls = []
    for threshold in thresholds:
        y_pred = [1 if compute_binary_prediction_result(value, [-1, threshold]) else 0 for value in pred_value]
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='binary')
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='binary')
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def compute_evaluation_metric(confusion_matrix):
    TP, FN, FP = 0, 0, 0
    if 'TP' in set(confusion_matrix['label']):
        TP_index = confusion_matrix.index[confusion_matrix['label'] == 'TP'].tolist()[0]
        TP = confusion_matrix['counts'][TP_index]
    if 'FN' in set(confusion_matrix['label']):
        FN_index = confusion_matrix.index[confusion_matrix['label'] == 'FN'].tolist()[0]
        FN = confusion_matrix['counts'][FN_index]
    if 'FP' in set(confusion_matrix['label']):
        FP_index = confusion_matrix.index[confusion_matrix['label'] == 'FP'].tolist()[0]
        FP = confusion_matrix['counts'][FP_index]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def group_confusion_matrix_by_keypoint(source_df, col_prefix):
    confusion_matrices = []
    for i in range(21):
        column = col_prefix + "_" + str(i)
        new_confusion_matrix = source_df[column].value_counts().rename_axis('label').reset_index(name='counts')
        confusion_matrices.append(new_confusion_matrix)
    return confusion_matrices


def tabulate_hand_localisation_result(source_df, col_prefix, new_col_prefix, threshold):
    result_df = source_df[['filename']]
    for i in range(21):
        column = col_prefix + "_" + str(i)
        new_column = new_col_prefix + "_" + str(i)
        result_df[new_column] = source_df.apply(lambda row: compute_prediction_result(row, column, threshold, False),
                                                axis=1)
    return result_df


def tabulate_evaluation_metric_by_keypoint(confusion_matrices):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(21):
        precision, recall, f1_score = compute_evaluation_metric(confusion_matrices[i])
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    result = {
        'keypoint': range(21),
        'precision': precisions,
        'recall': recalls,
        'f1_score': f1_scores
    }
    return pd.DataFrame(result)


def write_to_csv(dataframe, filename):
    dataframe.to_csv(filename, encoding='utf-8')


def write_csv_data(file_path, new_data, mode):
    with open(file_path, mode) as file:
        writer = csv.writer(file)
        writer.writerow(new_data)