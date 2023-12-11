import numpy as np
import pandas as pd
import json
import os

# path to videos
PATH_to_videos =  "dataset/task1_videos_mp4/input/"
# path to csv file
CSV_PATH = "dataset/train.csv"

# create csv file for custom dataset as follows

"""
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
"""

def create_csv_for_kinetics(path_to_videos, path_to_csv):
    """
    create csv file for custom dataset as follows
    the csv file is:
    ```
    path_to_video_1 label_1
    path_to_video_2 label_2
    ...
    path_to_video_N label_N
    ```
    """
    # get all video files
    video_files = os.listdir(path_to_videos)
    # create empty list
    video_files_with_labels = []
    # iterate over all video files
    for video_file_idx, video_file in enumerate(video_files):
        if not '.mp4' in video_file:
            continue
        # get label
        label = video_file.split(".")[0]
        # append video file and label to list
        # full video path
        video_file = os.path.join(path_to_videos, video_file)
        video_files_with_labels.append([video_file, video_file_idx])
    # create dataframe
    df = pd.DataFrame(video_files_with_labels)
    # save dataframe as csv file
    df.to_csv(path_to_csv, index=False, sep=" ", header=False)

if __name__ == "__main__":
    create_csv_for_kinetics(VIDEO_PATH, CSV_PATH)

