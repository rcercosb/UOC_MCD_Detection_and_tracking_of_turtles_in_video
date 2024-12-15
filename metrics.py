"""
Calculates the metrics of the detections and the metrics of the speeds of the detections.

Saves the metrics of the detections in a JSON file (boxes_metrics.json).

The metrics of the detections are the True Positives, the False Positives, the False Negatives, the Precision and the Recall.

Saves the metrics of the speeds of the detections in JSON files (VIDEOFILENAME_speeds_metrics.json).

The metrics of the speeds of the detections are the minimum, the maximum, the mean, the median and the total.

The files are saved inside three folders: ./detections/frame_by_frame/metrics/
"""


import os
import pandas as pd
import collections
import torch
from ultralytics.utils.metrics import bbox_iou

DATA_FOLDER = "./detections/frame_by_frame/data/"
LABELS_FOLDER = "D:/Thesis_Data_Set/Bahamas/Labels/"

METRICS_FOLDER = "./detections/frame_by_frame/metrics/"

IOU_THRESHOLD = 0.5

def create_output_folders():
    """
    Creates the output folders (if they don't exist)
    """

    if os.path.exists(METRICS_FOLDER) == False:
        os.makedirs(METRICS_FOLDER)

def get_boxes_speeds_filenames():
    """
    Gets the filenames of the boxes and speeds files

    Returns:
        boxes_filenames (list): A list of the boxes' filenames
        speeds_filenames (list): A list of the speeds' filenames
    """

    filenames = os.listdir(DATA_FOLDER)

    boxes_filenames = []
    speeds_filenames = []

    for filename in filenames:
        if "boxes.csv" in filename:
            boxes_filenames.append(filename)
        elif "speeds.csv" in filename:
            speeds_filenames.append(filename)
    
    return boxes_filenames, speeds_filenames

def get_data_dfs(filenames):
    """
    Gets a Pandas dataframe from each boxes or speeds file

    Parameters:
        filenames (list): A list of filenames

    Return:
        df_list (list): A list of dataframes
    """

    df_list = []

    for filename in filenames:
        df = pd.read_csv(DATA_FOLDER + filename)
        
        # filename: "DCW20180927_boxes.csv" -> filename.split("_")[0] -> "DCW20180927"
        df.columns.name = filename.split("_")[0]

        df_list.append(df)

    return df_list

def get_labels_df(folder_name):
    """
    Gets a Pandas dataframe from each labels file

    Parameters:
        folder_name (str): The name of the folder / video

    Returns:
        Pandas dataframe: The labels dataframe
    """

    filenames = os.listdir(LABELS_FOLDER + folder_name)

    # Reverse the order: from smallest (e.g., 630_0.txt) to biggest (e.g., 5790_0.txt)
    filenames.sort(reverse = True)

    df_list = []
    
    for filename in filenames:
        df = pd.read_csv(LABELS_FOLDER + folder_name  + "/" + filename, sep = " ", header = None)
        
        df.columns = ["frame", "x1", "y1", "x2", "y2"]

        df_list.append(df)

    # ignore_index = True: replaces the dataframes indices with 0 to n-1
    return pd.concat(df_list, ignore_index = True)

def reshape_coordinates(xyxy_list):
    """
    Reshapes the coordinates

    Parameters:
        xyxy_list (list): A list of coordinates (detections' frames shape: 3840 px X 2160 px)

    Returns:
        xyxy_list (list): A list of coordinates (labels' frames shape: 1280 (1/3) px X 720 (1/3) px)
    """

    for i in range(len(xyxy_list)):
        xyxy_list[i][0] = round(xyxy_list[i][0] / 3)
        xyxy_list[i][1] = round(xyxy_list[i][1] / 3)
        xyxy_list[i][2] = round(xyxy_list[i][2] / 3)
        xyxy_list[i][3] = round(xyxy_list[i][3] / 3)

    return xyxy_list

def calculate_IoU(labels, detections):
    """
    Calculates the IoU (Intersection over Union)

    Parameters:
        labels (list): A list of coordinates (labels)
        detections (list): A list of coordinates (detections)

    Returns:
        tp (int): The number of True Positives
        fp (int): The number of False Positives
    """

    tp = 0
    fp = 0

    iou_list = []

    for label in labels:
        for detection in detections:
            label_tensor = torch.tensor([label], dtype = torch.float)
            detection_tensor = torch.tensor([detection], dtype = torch.float)
    
            iou_list.append(bbox_iou(label_tensor, detection_tensor))

    # From most to least
    iou_list.sort(reverse = True)

    # The number of labels is the maximum number of True Positives
    # Check if the len(labels) largest IoU's pass the threshold
    for iou in iou_list[:len(labels)]:
        if iou >= IOU_THRESHOLD:
            tp += 1
        
        # If they aren't equal, they have already been counted by fn_counter or fp_counter
        elif iou < IOU_THRESHOLD and len(labels) == len(detections):
            fp += 1
        
    return tp, fp
    
def boxes_metrics(boxes_df_list):
    """
    Writes a JSON file with the TP, FP, FN, Precision and Recall of every dataframe

    Parameters:
        boxes_df_list (list): A list of boxes dataframes
    """

    df_summary = pd.DataFrame()
    
    df_summary.index = ["TP", "FP", "FN", "Precision", "Recall"]

    for boxes_df in boxes_df_list:
        labels_df = get_labels_df(boxes_df.columns.name)
        
        labels_frames_list = labels_df["frame"].tolist() # LEFT SET
        boxes_frames_list = boxes_df["frame"].tolist() # RIGHT SET

        tp = 0

        # The frames present in labels, but not in boxes.
        # The frames with undetected objects. The False Negatives.
        fn_counter = collections.Counter(labels_frames_list) # A set() can't have duplicates (multiple objects in the same frame)
        fn_counter.subtract(boxes_frames_list) # LEFT JOIN EXCLUDING INNER JOIN
        
        fn = len(list(fn_counter.elements())) # .total(): sums all the counts, including negative ones

        # The frames present in boxes, but not in labels.
        # The frames with detected nonexistent objects. The False Positives.
        fp_counter = collections.Counter(boxes_frames_list)
        fp_counter.subtract(labels_frames_list) # RIGHT JOIN EXCLUDING INNER JOIN
        
        fp = len(list(fp_counter.elements()))

        IoU_frames = set(boxes_frames_list) & set(labels_frames_list) # INNER JOIN

        for frame in IoU_frames:
            frame_labels = labels_df[labels_df["frame"] == frame]
            frame_detections = boxes_df[boxes_df["frame"] == frame]

            # Get a list of the coordinates
            frame_labels_xyxy = frame_labels[["x1", "y1", "x2", "y2"]].values.tolist()
            frame_detections_xyxy = frame_detections[["x1", "y1", "x2", "y2"]].values.tolist()

            frame_detections_xyxy = reshape_coordinates(frame_detections_xyxy)

            tp_delta, fp_delta = calculate_IoU(frame_labels_xyxy, frame_detections_xyxy)
            
            tp += tp_delta
            fp += fp_delta

        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)

        df_summary[boxes_df.columns.name] = [tp, fp, fn, precision, recall]

    # orient = "table": JSON string format
    # indent = 4: the number of whitespaces used to indent each record
    df_summary.to_json(METRICS_FOLDER + "boxes.json", orient = "table", indent = 4)

def speeds_metrics(speeds_df_list):
    """
    Writes a JSON file for each speed (preprocess, inference and postprocess) with the min, max, mean, median and sum of every dataframe

    Parameters:
        speeds_df_list (list): A list of speeds dataframes
    """
 
    # ["frame", "preprocess", "inference", "postprocess"]
    column_names = list(speeds_df_list[0].columns)
    
    # ["preprocess", "inference", "postprocess"]
    for column_name in column_names[1:]:
        df_summary = pd.DataFrame()

        df_summary.index = ["Min", "Max", "Mean", "Median", "Total"]

        for speeds_df in speeds_df_list:
            speed_min = round(speeds_df[column_name].min(), 2)
            speed_max = round(speeds_df[column_name].max(), 2)
            speed_mean = round(speeds_df[column_name].mean(), 2)
            speed_median = round(speeds_df[column_name].median(), 2)
            speed_sum = round(speeds_df[column_name].sum(), 2)

            df_summary[speeds_df.columns.name] = [speed_min, speed_max, speed_mean, speed_median, speed_sum]

        df_summary.to_json(METRICS_FOLDER + column_name + ".json", orient = "table", indent = 4)


if __name__ == "__main__":
    create_output_folders()

    boxes_filenames, speeds_filenames = get_boxes_speeds_filenames()

    boxes_df_list = get_data_dfs(boxes_filenames)
    speeds_df_list = get_data_dfs(speeds_filenames)

    boxes_metrics(boxes_df_list)
    speeds_metrics(speeds_df_list)
