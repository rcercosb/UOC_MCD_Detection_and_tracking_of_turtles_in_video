"""
Saves a given subset of video frames from a given video.

For every given frame number, the original frame is saved.

When the coordinates are provided, the frame with the bounding boxes it's also saved.

The files are stored inside the same folder as this code file.
"""


import cv2

INPUT_PATHNAME = "D:/Thesis_Data_Set/Bahamas/Videos/"
VIDEO_FILENAME = "HSC20190406.MOV"


def reshape_coordinates(bbox):
    """
    Reshapes the coordinates

    Parameters:
        bbox (list): A list of coordinates (labels' frames shape: 1280 px X 720 px)

    Returns:
        bbox (list): A list of coordinates (detections' frames shape: 3840 (3x) px X 2160 (3x) px)
    """

    bbox[0] = bbox[0] * 3
    bbox[1] = bbox[1] * 3
    bbox[2] = bbox[2] * 3
    bbox[3] = bbox[3] * 3

    return bbox

def write_frames(labels_dict):
    """
    Writes the original frames or with and without the bounding boxes

    Parameters:
        labels_dict (dict): A dictionary with the frame numbers as the keys and None or the bounding boxes as the values
    """
    
    video_capture = cv2.VideoCapture(INPUT_PATHNAME + VIDEO_FILENAME)

    frame_num = 0

    # yellow
    bgr = (0, 255, 255) 

    frames_nums_list = list(labels_dict.keys())
    frames_nums_list.sort()

    while video_capture.isOpened():
        success, frame = video_capture.read()

        # Frame read
        if success:
            # Frame found
            if frame_num in frames_nums_list:
                cv2.imwrite("./" + str(frame_num) + ".jpg", frame)

                bboxes = labels_dict[frame_num]

                if bboxes != None:
                    for bbox in bboxes:
                        bbox_re = reshape_coordinates(bbox)

                        cv2.rectangle(frame, (bbox_re[0], bbox_re[1]), (bbox_re[2], bbox_re[3]), bgr, 4)

                    cv2.imwrite("./" + str(frame_num) + "_bbox.jpg", frame)

            # All frames found
            elif frame_num > frames_nums_list[-1]:
                break

            # Frame number: 0, 1, 2, ...
            frame_num += 1

        # End of video
        else:
            break

    video_capture.release()


if __name__ == "__main__":
    labels_dict = {1126: [[623, 101, 647, 125]]}

    write_frames(labels_dict)