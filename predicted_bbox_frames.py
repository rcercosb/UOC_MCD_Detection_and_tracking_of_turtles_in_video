"""
Uses a Yolov8 model (with DELT's weights and biases) to infer a video.

Saves the detections as annotated frames (frame_number.jpg) and as lines in a TXT file (detections.txt).

The annotated frames show the bounding boxes, the confidences and the class names (turtle) of the detected objects.

The lines of the TXT file have the frame number, the bounding box coordinates, the confidence and the class (0.0) of the detection.

The files are stored inside a folder with the inferred video's name. In turn, this folder is inside two folders: ./detections/annotated_frames/
"""

from ultralytics import YOLO
import cv2
import os

INPUT_PATHNAME = "D:/Thesis_Data_Set/Bahamas/Videos/"
VIDEO_FILENAME = "HSC20190406.MOV"

OUTPUT_PATHNAME = "./detections/annotated_frames/" + VIDEO_FILENAME.split(".")[0] + "/"
TXT_FILENAME = "boxes.txt"


if os.path.exists(OUTPUT_PATHNAME) == False:
    os.makedirs(OUTPUT_PATHNAME)

txt_file = open(OUTPUT_PATHNAME + TXT_FILENAME, "w")

video_capture = cv2.VideoCapture(INPUT_PATHNAME + VIDEO_FILENAME)

frame_num = 0

# Load DELF's base model
model = YOLO("best.pt")

while video_capture.isOpened():
    success, frame = video_capture.read()

    # Frame read
    if success:
        results = model.predict(frame)

        # First and only element of the list
        result = results[0]

        # Object found
        if len(result.boxes.data) > 0:
            frame_pathname = OUTPUT_PATHNAME + str(frame_num) + ".jpg"
            
            annotated_frame = result.plot()

            cv2.imwrite(frame_pathname, annotated_frame)

            # From tensor to list of lists. One for each object detected: [[x1, y1, x2, y2, conf, class], [...], ...]
            boxes_list = result.boxes.data.tolist()
            
            for box_list in boxes_list:
                box_str_list = [str(round(box_list[0])),
                                 str(round(box_list[1])),
                                 str(round(box_list[2])),
                                 str(round(box_list[3])),
                                 str(round(box_list[4], 4)),
                                 str(box_list[5])]

                # frame_num x1 y1 x2 y2 conf class
                # frame_num x1 y1 x2 y2 conf class
                # ...
                txt_file.write(str(frame_num) + " " + ' '.join(box_str_list) + "\n")

        # Frame number: 0, 1, 2, ...
        frame_num += 1

    # End of video
    else:
        break

video_capture.release()

txt_file.close()