fake = False

import numpy as np
import pandas as pd
import json
import os

# path to videos
# PATH_to_txt_files = "./data/ucfTrainTestlist/classInd.txt"
root_path = "/root/"
PATH_to_annotations = root_path + "ucfTrainTestlist/"
PATH_to_videos = root_path + "UCF-101/"
PATH_to_txt_files = root_path + "ucfTrainTestlist/classInd.txt"

# results paths
PATH_to_classInd = PATH_to_videos + "classInd.json"
PATH_to_train = PATH_to_videos + "train.csv"
PATH_to_test = PATH_to_videos + "test.csv"
PATH_to_val = PATH_to_videos + "val.csv"

# read txt
with open(PATH_to_txt_files, "r") as f:
    lines = f.readlines()

classinds = {}
for line in lines:
    line = line.split("\n")[0]
    print(line)
    line = line.split(" ")
    id = line[0]
    name = line[1]
    if fake:
        classinds[name] = 0
    else:
        classinds[name] = int(id)-1

# save resulting dict as json
with open(PATH_to_classInd, "w") as fp:
    json.dump(classinds, fp)

# import subprocess
from subprocess import call

# # copy files using os
# call(
#     ["cp", "./data/ucfTrainTestlist/classids.json", "./data/ucf/classInd.json"]
# )

# read train csv file
train_csv = pd.read_csv(PATH_to_annotations + "trainlist01.txt", sep=" ", header=None)
train_csv.columns = ["path", "label"]
# substract from each label 1
train_csv["label"] = train_csv["label"] - 1
# save as csv
train_csv.to_csv(PATH_to_train, index=False, header=False, sep=" ")
train_csv.to_csv(PATH_to_val, index=False, header=False, sep=" ")


###
# # read test csv file
test_csv = pd.read_csv(PATH_to_annotations + "testlist01.txt", sep=" ", header=None)

# expand test_csv by another column
test_csv["label"] = 0

## go through each row and lookup the label in the classInd.json file
for row_idx, row in enumerate(test_csv.iterrows()):
    # get row value
    row = row[1].to_list()[0]
    video_name = row.split('/')[0]
    label = classinds[video_name]
    # save the label
    test_csv["label"][row_idx] = label


# save as csv
test_csv.to_csv(PATH_to_test, index=False, header=False, sep=" ")
#
# # read val csv file
# val_csv = pd.read_csv("./data/ucfTrainTestlist/testlist01.txt", sep=" ", header=None)
# val_csv.columns = ["path", "label"]
# # substract from each label 1
# val_csv["label"] = val_csv["label"] - 1
# # save as csv
# val_csv.to_csv("./data/ucf/val.csv", index=False, header=False, sep=" ")

# call(["cp", "./data/ucfTrainTestlist/trainlist01.txt", "./data/ucf/train.csv"])
# call(["cp", "./data/ucfTrainTestlist/testlist01.txt", "./data/ucf/test.csv"])
# call(["cp", "./data/ucfTrainTestlist/testlist01.txt", "./data/ucf/val.csv"])
# create csv file for custom dataset as follows
