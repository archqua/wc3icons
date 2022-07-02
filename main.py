#!/bin/env python
from pix import Pix
from dense_pix import DensePix
from util import darken_edge
import torch
from torchvision import transforms as tt
import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inp", help="input file", type=str)
parser.add_argument("-o", "--outp", help="output file", type=str)
parser.add_argument("-d", "--dense", help="use dense architecture", action="store_true")
parser.add_argument("-D", "--dropout", help="dropout", type=float, default=0.0)
parser.add_argument("-p", "--pretrained", help="use weights from training with simple pretrain", action="store_true")
loop_group = parser.add_mutually_exclusive_group()
loop_group.add_argument("-s", "--simple", help="simple train loop", action="store_true")
loop_group.add_argument("-c", "--cycle", help="cycle train loop", action="store_true")
loop_group.add_argument("-l", "--harmonic", help="harmonic train loop", action="store_true")
parser.add_argument("-e", "--extra_crop", help="extra crop factor", type=float, default=1.0)
parser.add_argument("--frame", help="frame", type=str, default="frame.png")


def crop_edges(img):
    size = img.size
    if size[0] == size[1]:
        img = img.resize(final_size)
    elif size[0] > size[1]:
        diff = size[0] - size[1]
        left = diff//2
        right = size[0] - left - (diff % 2)
        box = (left, 0, right, size[1])
        final_size = (size[1],) * 2
        img = img.resize(final_size, box=box)
    else:
        diff = size[1] - size[0]
        top = diff // 2
        bottom = size[1] - top - (diff % 2)
        box = (0, top, size[0], bottom)
        final_size = (size[0],) * 2
        img = img.resize(final_size, box=box)
    return img

def extra_crop(img, factor):
    if factor < 1.0:
        size = img.size
        size = (int(size[0] * factor), int(size[1] * factor))
        diff = img.size[0] - size[0]
        left = diff//2
        right = img.size[0] - left - (diff % 2)
        top = diff // 2
        bottom = img.size[1] - top - (diff % 2)
        box = (left, top, right, bottom)
        img = img.resize(size=size, box=box)
    return img

def get_model(which, dense, dropout, pretrained):
    if dense:
        fname = "dense_"
        model = DensePix.Transformator(width=16, density = 3 if which[0] else 2, random_dim=128, dropout=dropout)
    else:
        fname = "pix_"
        model = Pix.Transformator(width=16, latent_dim=128, random=16, dropout=dropout)
    if (which[0]):
        fname += "simple"
    elif (which[1]):
        fname += "cycle"
    elif (which[2]):
        fname += "harmonic"
    else:
        pass # TODO error
    if pretrained:
        fname += "_pretrained"
    fname += ".pt"
    model.train()
    model.load_state_dict(torch.load(os.path.join("weights", fname), map_location=torch.device('cpu')))
    return model

def main(arg):
    # TODO add operations with frame
    with Image.open(args.inp) as in_img:
        cropped = crop_edges(in_img)
        extra_cropped = extra_crop(cropped, args.extra_crop)
        img = extra_cropped.resize((56, 56))
        pic = tt.ToTensor()(img)[None,:]
        # print(pic.shape)
        darken_edge(pic)
        model = get_model(
            (args.simple, args.cycle, args.harmonic),
            args.dense,
            args.dropout,
            args.pretrained,
        )
        pic = model(pic)
        # TODO add frame
        out_img = tt.ToPILImage()(pic.squeeze())
        with Image.open(args.frame) as frame:
            real_out = Image.new(frame.mode, frame.size, (0,0,0))
            real_out.paste(frame, (0, 0))
            real_out.paste(out_img, (4, 4))
            real_out.save(args.outp)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
