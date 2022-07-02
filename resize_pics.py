#!/bin/env python
from PIL import Image
import os
import sys

def resize_pic(file):
    with Image.open(file) as img:
        img = Image.open(file)
        size = img.size
        final_size = (64, 64)
        if size[0] == size[1]:
            img = img.resize(final_size)
        elif size[0] > size[1]:
            diff = size[0] - size[1]
            left = diff//2
            right = size[0] - left - (diff % 2)
            box = (left, 0, right, size[1])
            img = img.resize(final_size, box=box)
        else:
            diff = size[1] - size[0]
            top = diff // 2
            bottom = size[1] - top - (diff % 2)
            box = (0, top, size[0], bottom)
            img = img.resize(final_size, box=box)
        return img

in_dir = sys.argv[1]
if in_dir[-1] != '/':
    in_dir += '/'
out_dir = sys.argv[2]
if out_dir[-1] != '/':
    out_dir += '/'

if __name__ == "__main__":
    for _, _, files in os.walk(in_dir):
        for file in files:
            try:
                pic = resize_pic(in_dir + file)
                pic.save(out_dir + file)
            except:
                print(file, "failed")
