"""
Uses a Yolov8 model (with DELT's weights and biases) to infer all the videos in a folder.

Saves the detections as lines in CSV files (VIDEOFILENAME_boxes.csv).

The lines of the CSV files have the frame number, the bounding box coordinates, the confidence and the class (0.0) of the detection.

Saves the speeds of the detections as lines in CSV files (VIDEOFILENAME_speeds.csv).

The lines of the CSV files have the frame number and the preprocess, inference and postprocess speeds.

The files are saved inside three folders: ./detections/frame_by_frame/data/
"""


from ultralytics import YOLO
import os
import csv

INPUT_PATHNAME = "D:/Thesis_Data_Set/Bahamas/Videos/"
OUTPUT_PATHNAME = "./detections/frame_by_frame/data/"


def create_output_folders():
    """
    Creates the output folders (if they don't exist)
    """

    if os.path.exists(OUTPUT_PATHNAME) == False:
        os.makedirs(OUTPUT_PATHNAME)

def inference(filename):
    """
    Infers the video and returns the detections and the speeds of the detections
    
    Parameters:
        filename (str): The name of the video to be inferred
    
    Returns:
        frame_xyxy_conf_class_list (list): A list of the detections
        frame_pre_inf_post_list (list): A list of the speeds of the detections
    """

    # Load DELT's base model
    model = YOLO("DELT_v1.pt")

    # stream = True: returns a generator that only stores the result of the current frame in memory (avoids running out of RAM)
    results = model.predict(INPUT_PATHNAME + filename, stream = True)

    frame_num = 0
    
    frame_xyxy_conf_class_list = []
    frame_pre_inf_post_list = []

    for result in results:
        pre_inf_post = list(result.speed.values())

        pre_inf_post.insert(0, frame_num)

        pre_inf_post[1] = round(pre_inf_post[1], 2) # Preprocess
        pre_inf_post[2] = round(pre_inf_post[2], 2) # Inference
        pre_inf_post[3] = round(pre_inf_post[3], 2) # Postprocess

        frame_pre_inf_post_list.append(pre_inf_post)

        # If object/s detected
        if len(result.boxes.data) > 0:
            # From tensor to list of lists. One for each object detected: [[x1, y1, x2, y2, conf, class], [...], ...]
            xyxy_conf_class_list = result.boxes.data.tolist()

            for xyxy_conf_class in xyxy_conf_class_list:
                xyxy_conf_class.insert(0, frame_num)

                # Round to nearest integer (pixels' coordinates are integers)
                xyxy_conf_class[1] = round(xyxy_conf_class[1]) # x1
                xyxy_conf_class[2] = round(xyxy_conf_class[2]) # y1
                xyxy_conf_class[3] = round(xyxy_conf_class[3]) # x2
                xyxy_conf_class[4] = round(xyxy_conf_class[4]) # y2
                
                xyxy_conf_class[5] = round(xyxy_conf_class[5], 4) # conf

                frame_xyxy_conf_class_list.append(xyxy_conf_class)

        # Frame number: 0, 1, 2, ...
        frame_num += 1

    return frame_xyxy_conf_class_list, frame_pre_inf_post_list

def write_boxes_csv(filename, frame_xyxy_conf_class_list):
    """
    Writes a CSV file with the frame number, bounding box coordinates, confidence and class of every detection

    Parameters:
        filename (str): The name of the CSV file
        frame_xyxy_conf_class_list (list): A list of detections
    """

    write_csv(filename, frame_xyxy_conf_class_list, False)

def write_speeds_csv(filename, frame_pre_inf_post_list):
    """
    Writes a CSV file with the frame number and the preprocess, inference and postprocess speeds

    Parameters:
        filename (str): The name of the CSV file
        frame_pre_inf_post_list (list): A list of the speeds of the detections
    """

    write_csv(filename, frame_pre_inf_post_list, True)

def write_csv(filename, data_list, boxes_or_speeds):
    """
    Writes a CSV file

    Parameters:
        filename (str): The name of the CSV file
        data_list (list): The data to be written
        boxes_or_speeds (bool): The flag that tells which type of data will be written
    """

    if not(boxes_or_speeds):
        type_of_file = "boxes"
        fields = ["frame", "x1", "y1", "x2", "y2", "conf", "class"]
    else:
        type_of_file = "speeds"
        fields = ["frame", "preprocess", "inference", "postprocess"]

    # filename: "DCW20180927.MOV" -> filename.split(".")[0] -> "DCW20180927"
    with open(OUTPUT_PATHNAME + filename.split(".")[0] + "_" + type_of_file + ".csv", "w", newline = "") as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(fields)
        csv_writer.writerows(data_list)


if __name__ == "__main__":
    create_output_folders()

    filenames = os.listdir(INPUT_PATHNAME)

    for filename in filenames:
        boxes_list, speeds_list = inference(filename)

        write_boxes_csv(filename, boxes_list)
        
        write_speeds_csv(filename, speeds_list)