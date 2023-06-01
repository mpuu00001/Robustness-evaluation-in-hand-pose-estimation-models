import sys
sys.path.insert(0, '../')
from functions import *
import csv


index_file = open(r'D:\PuMuxin\filtered_hands_with_objects_prediction_result.csv', 'r')
index_content = list(csv.reader(index_file, delimiter=","))
index_file.close()

content_file = open(r'D:\PuMuxin\Bodyhands_Openpose\result\CMUhand\baseline\raw\hands_with_objects_prediction_result.csv', 'r')
file_content = list(csv.reader(content_file, delimiter=","))
content_file.close()

new_file = r'D:\PuMuxin\Bodyhands_Openpose\result\CMUhand\baseline\raw\filtered_hands_with_objects_prediction_result.csv'
for row in file_content:
    if row != []:
        for index in index_content:
            if index != []:
                if row[0] == index[0]:
                    write_csv_data(new_file, row, 'a')