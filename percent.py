#!/bin/env python
# from PIL import Image

import os
import sys
import random
import shutil

in_dir = sys.argv[1]
if in_dir[-1] != '/':
    in_dir += '/'
out_dir = sys.argv[2]
if out_dir[-1] != '/':
    out_dir += '/'
if len(sys.argv) > 3:
    percent = float(sys.argv[3])
    if percent >= 1:
        percent /= 100
else:
    percent = 0.01

if __name__ == "__main__":
    for _, _, files in os.walk(in_dir):
        for file in files:
            if (random.random() < percent):
                # pic = Image.open(os.path.join(in_dir, file))
                # pic.save(os.path.join(out_dir, file))
                shutil.copyfile(os.path.join(in_dir, file), os.path.join(out_dir, file))
