#!/bin/env python
import sys
from PIL import Image


def extract_frame(img, width=4):
    frame = Image.new(img.mode, img.size, (0,0,0))
    for i in range(min(width, img.size[0])):
        for j in range(img.size[1]):
            data = img.getpixel((i,j))
            frame.putpixel((i,j), data)
    for i in range(max(img.size[0]-width, 0), img.size[0]):
        for j in range(img.size[1]):
            data = img.getpixel((i,j))
            frame.putpixel((i,j), data)
    for j in range(min(width, img.size[1])):
        for i in range(img.size[0]):
            data = img.getpixel((i,j))
            frame.putpixel((i,j), data)
    for j in range(max(img.size[1]-width, 0), img.size[1]):
        for i in range(img.size[0]):
            data = img.getpixel((i,j))
            frame.putpixel((i,j), data)
    return frame


in_file = sys.argv[1]
out_file = sys.argv[2]

if __name__ == "__main__":
    with Image.open(in_file) as img:
        frame = extract_frame(img)
        frame.save(out_file)
