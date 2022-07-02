#!/bin/env python
from PIL import Image
import os
import sys

def resize_pic(file):
    with Image.open(file) as img:
        img = Image.open(file)
        blacken_white(img)
        img = pad_to_square(img)
        final_size = (64, 64)
        img = img.resize(final_size)
        return img

def blacken_white(img):
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            data = img.getpixel((i,j))
            if (data[0] == 255 and data[1] == 255 and data[2] == 255):
                img.putpixel((i,j), (0,0,0))

def pad_to_square(img):
    size = img.size
    if size[0] == size[1]:
        res = Image.new(img.mode, size, (0,0,0))
        res.paste(img, (0,0))
    elif size[0] > size[1]:
        diff = size[0] - size[1]
        top = diff//2
        # bottom = size[1] - top - (diff % 2)
        res = Image.new(img.mode, (size[0], size[0]), (0,0,0))
        res.paste(img, (0, top))
    else:
        diff = size[1] - size[0]
        left = diff // 2
        # right = size[0] - left - (diff % 2)
        res = Image.new(img.mode, (size[1], size[1]), (0,0,0))
        res.paste(img, (left, 0))
    return res

in_dir = sys.argv[1]
if in_dir[-1] != '/':
    in_dir += '/'
out_dir = sys.argv[2]
if out_dir[-1] != '/':
    out_dir += '/'

if __name__ == "__main__":
    for _, _, files in os.walk(in_dir):
        for file in files:
            pic = resize_pic(in_dir + file)
            pic.save(out_dir + file)
