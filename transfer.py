#!/bin/env python
import os
import sys
import shutil

in_dir = sys.argv[1]
if in_dir[-1] != '/':
    in_dir += '/'
out_dir = sys.argv[2]
if out_dir[-1] != '/':
    out_dir += '/'

for _, dirs, _ in os.walk(in_dir):
    for dir in dirs:
        # print(dir)
        for _, _, files in os.walk(dir):
            for file in files:
                if 'jpg' in file or 'png' in file:
                    new_file = dir + '_' + file
                    shutil.copyfile(os.path.join(dir,file), os.path.join(out_dir, new_file))
