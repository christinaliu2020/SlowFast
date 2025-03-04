import numpy as np
import pandas as pd
import json
import os

root_Path = "dataset/"
PATH_to_videos =  "dataset/task1_videos_mp4/input/"

# results paths
PATH_to_classInd = root_Path + "classInd.json"
PATH_to_train = root_Path + "train.csv"
PATH_to_test = root_Path + "test.csv"
PATH_to_val = root_Path + "val.csv"

# get all video files in PATH_to_videos
all_files = os.listdir(PATH_to_videos)
all_files = [f for f in all_files if f.endswith(".mp4")]

# resulting entries
train = []
for f in all_files:
    train.append([PATH_to_videos + f, 0])

# to csv file
df = pd.DataFrame(train, columns=["video", "label"], index=None)
df.to_csv(PATH_to_train, index=False, header=False, sep=" ")
df.to_csv(PATH_to_test, index=False, header=False, sep=" ")
df.to_csv(PATH_to_val, index=False, header=False, sep=" ")

# get classInd
classInd = {"rnd0": 0}
# save resulting dict as json
with open(PATH_to_classInd, "w") as fp:
    json.dump(classInd, fp)
