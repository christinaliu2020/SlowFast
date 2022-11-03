import numpy as np
import pandas as pd
import json
import os

# path to videos
PATH_to_txt_files = "./data/ucfTrainTestlist/classInd.txt"

# read txt
with open(PATH_to_txt_files, "r") as f:
    lines = f.readlines()

res = {}
for line in lines:
    line = line.split("\n")[0]
    print(line)
    line = line.split(" ")
    id = line[0]
    name = line[1]
    res[name] = int(id)

# save resulting dict as json
with open("./data/ucfTrainTestlist/classids.json", "w") as fp:
    json.dump(res, fp)

# import subprocess
from subprocess import call

# copy files using os
call(
    ["cp", "./data/ucfTrainTestlist/classids.json", "./data/ucf/classInd.json"]
)
call(["cp", "./data/ucfTrainTestlist/trainlist01.txt", "./data/ucf/train.csv"])
call(["cp", "./data/ucfTrainTestlist/testlist01.txt", "./data/ucf/test.csv"])
call(["cp", "./data/ucfTrainTestlist/testlist01.txt", "./data/ucf/val.csv"])
# create csv file for custom dataset as follows
