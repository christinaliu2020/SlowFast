import numpy as np
import pandas as pd
import json
import os

# path to videos
PATH_to_txt_files = "./data/ucf/classInd.txt"

# read txt
with open(PATH_to_txt_files, 'r') as f:
    lines = f.readlines()

res = {}
for line in lines:
    line = line.split('\n')[0]
    print(line)
    line = line.split(' ')
    id = line[0]
    name = line[1]
    res[name] = id

# save resulting dict as json
with open('./data/ucf/classids.json', 'w') as fp:
    json.dump(res, fp)

# create csv file for custom dataset as follows

