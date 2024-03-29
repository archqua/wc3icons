# wc3icons
This repository is final project for 2022 spring DLS season.
The idea is to train generative adversarial network to
turn any picture into a Warcraft III icon.



# Table of contents
[Data](#Data)

[Preprocessing](#Preprocessing)

[Models](#Models)

[Training](#Training)

[Test](#Test)

[Results](#Results)
- [Identity](#Identity)
- [Simple](#Simple)
- [Cycle](#Cycle)
- [Harmonic](#Harmonic)

[Usage](#Usage)

[Conclusion](#Conclusion)



# Data
Most of the icons to train on are taken from
[here](https://www.moddb.com/games/warcraft-iii-frozen-throne/addons/wow-icons-for-war3).
These are not actual wc3 icons -- these are adjusted World of Warcraft icons.
The rest of icons are actual wc3 icons taken from
[here](https://wowpedia.fandom.com/wiki/Wowpedia:Warcraft_III_icons/Icon_list).
Only icons for buttons were eventually used, which turned out to be
unnecessary, but it doesn't seem to be a huge loss anyway.

There are 22529 of adjusted WoW icons and 1007 of wc3 icons, 23536 overall.

Non-wc3 pictures are taken from several datasets:
1. [fruits](https://www.kaggle.com/datasets/moltean/fruits)
2. [dogs and cats](https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip)
3. [monkeys](https://www.kaggle.com/datasets/slothkong/10-monkey-species?select=training)
4. [fashion](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
5. [buildings](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset?select=architectural-styles-dataset)
6. [weather](https://data.mendeley.com/datasets/4drtyfjtfy/1)
7. [food](https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset?select=training)
8. [animals](https://www.kaggle.com/datasets/virtualdvid/oregon-wildlife)
9. [faces](https://drive.google.com/file/d/1KWPc4Pa7u2TWekUvNu9rTSO0U2eOlZA9/view?usp=sharing)
    (small version of [this](https://github.com/NVlabs/ffhq-dataset))

Datasets were only used partially, there are
243 photos of fruits and vegetables, 5050 of dogs and cats,
1097 of monkeys, 4518 of fashion items, 3523 of buildings,
1127 of landscapes, 1500 of food and 1500 of non-food,
5272 of animals and finally 3143 of faces.
Which is 26973 photos overall.



# Preprocessing
Each non-icon was cropped, resized to $56 \times 56$ and shadowed at the edges.
Each icon was cropped to $56 \times 56$ to remove frame.
$64 \times 64$ datasets can be found
[here](https://drive.google.com/drive/folders/1Ndtzlsm8XtfXEW9EZAnAZvAsd8eGBwN9?usp=sharing),
the rest of preprocessing is done via [util.py](util.py).



# Models
Two models were tested. One is
[pix2pix-like](https://arxiv.org/abs/1611.07004) and can be found
in [pix.py](pix.py), the other one mixes pix2pix with
[dense-net](https://arxiv.org/abs/1608.06993v5)
and can be found in [dense\_pix.py](dense\_pix.py).

<figure>

<figcaption align="center">
<b>roughly pix architecture</b>
</figcaption>
<img src="pix_architecture.png" title="pix architecture"/>

</figure>

<figure>

<figcaption align="center">
<b>roughly dense architecture (density=4)</b>
</figcaption>
<img src="dense_architecture.png" title="dense architecture"/>

</figure>

Regarding density of dense architecture, in simple train loop (see below)
it equals 3, in cycle and harmonic train loops it's 2.

On last two layers upconvolutions are replaced with bilinear upsampling with
simple convolutions to mitigate regular grid-like artifacts.

To add randomness to output and make it possible to create several
different icons from one photo,
random vectors are appended to $N \times 1 \times 1$ feature maps.



# Training
There are three train loops, all can be found in [train\_util.py](train\_util.py).
Loss functions were chosen to be MSE, rather than BCE.

If you wish to know how much entropy was generated during training
(number of epochs, training time) or how to use
[util.py](util.py) and [train\_util.py](train\_util.py)
to train models, you can check out
[notebooks](https://drive.google.com/drive/folders/11y8_z9tHXHPIRienfLTCfpnVQB3NP8xs?usp=sharing).


## Simple train loop
First one is simple train loop:
1. make transformator generate icons from photos
2. train discriminator on real icons and fakes
3. make transformator generate icons from photos
4. train transformator


## Cycle train loop
Is inspired by [CycleGAN](https://arxiv.org/abs/1703.10593).

Train forward and backward transformator and discriminator
as in simple train loop with addition to cycle consistency loss.
Cycle consistency loss is MSE.


## Harmonic train loop
Is inspired by [HarmonicGAN](https://arxiv.org/abs/1902.09727).

Train forward and backward transformator and discriminator
as in simple train loop with addition to harmonic loss.
Unlike in the original, cycle consistency loss is ignored here.
The intention here is to preserve trained model from
"willing" to preserve pictures unchanged.
To calculate harmonic loss, each picture was split into 16 cells
($4 \times 4$).
The implementation of harmonic loss can be found in [harmonic\_loss.py](harmonic\_loss.py).

## Pretraining
In some cases, in order to initialize weights, transformators were for 1 epoch
(pre)trained in autoencoder manner with L1 loss function.
Pretraining of transformators was done on the opposites of their actual domains
(photo-to-icon transformator was pretrained in icon-to-icon mode)
to prevent transformators from preserving images to much.


## Parameters
Inspired by [DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html),
learning rates were chosen to be
$10^{-4}$ for transformators and $10^{-5}$ for discriminators,
betas were chosen to be $(0.5,\,0.999)$.


## GAN trickery
Labels (real and fake) were flipped (which is seemingly useless in case of
MSE loss, but MSE loss is later addition, so label flip is kind of legacy).

Labels were sampled from uniform distributions
$\mathcal{U}\left(0, 0.15\right)$ and $\mathcal{U}\left(0.85, 1\right)$.

When training discriminator, transformator weights were randomly chosen
from last 7 epochs.

When transformator loss exceeded $0.8$, it was given extra iterations
(not more than 3) to train.



# Test
The following pictures were used for testing:

<img src="illustration/test_orig/busya.jpg" width="400px"/>

[link](https://i.pinimg.com/736x/90/0a/b7/900ab76cf0c3b2fe8683e0e2039beb00.jpg)

<img src="illustration/test_orig/floppa.jpg"/>

[link](https://www.pinterest.com/pin/strategic-covering-women-model-redhead-freckles-face-curly-hair--483714816208425154/)

<img src="illustration/test_orig/ginger.jpg" width="400px"/>

<img src="illustration/test_orig/horny.jpg" width="400px"/>

[link](https://www.wallpaperbetter.com/wallpaper/810/134/410/ka-52-helicopter-russian-1080P-wallpaper.jpg)

<img src="illustration/test_orig/ka_52.jpg" width="400px"/>

<img src="illustration/test_orig/ll.jpg" width="400px"/>

[link](https://i.pinimg.com/736x/df/ef/52/dfef52bc718c0e35ded5ad5eb80da4bb.jpg)

<img src="illustration/test_orig/mike_wazowski.jpg" width="400px"/>

<img src="illustration/test_orig/pig.jpg" width="400px"/>

<img src="illustration/test_orig/sigma.jpg" width="400px"/>

[link](https://cdn.substack.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https://bucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com/public/images/68ce3fc6-7790-4875-986f-03af22d56395_640x640.jpeg)

<img src="illustration/test_orig/triplechad.jpg" width="400px"/>



# Results
## Identity
This is for comparisson.
<table><tbody>
<tr>
<td><img src="illustration/identity/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/identity/horny.png" width="128px"/></td>
<td><img src="illustration/identity/busya.png" width="128px"/></td>
<td><img src="illustration/identity/ll.png" width="128px"/></td>
<td><img src="illustration/identity/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/identity/sigma.png" width="128px"/></td>
<td><img src="illustration/identity/ginger.png" width="128px"/></td>
<td><img src="illustration/identity/ka_52.png" width="128px"/></td>
<td><img src="illustration/identity/pig.png" width="128px"/></td>
<td><img src="illustration/identity/floppa.png" width="128px"/></td>
</tr>
</tbody></table>


## Simple

### Pix

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/pix/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/pix/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/pix/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/pix/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/simple/pix/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/pix/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/pix/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/pix/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/pix/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/simple/pix/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

### Dense

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/dense/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/dense/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/dense/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/dense/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/simple/dense/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/dense/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/dense/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/simple/dense/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/simple/dense/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/simple/dense/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>


## Cycle

### Pix

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/pix/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/pix/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/pix/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/pix/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/pix/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/pix/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/pix/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/pix/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/pix/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

### Dense

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/dense/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/dense/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/dense/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/dense/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/dense/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/dense/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/cycle/dense/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/cycle/dense/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/cycle/dense/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>


## Harmonic

### Pix

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/pix/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/pix/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/pix/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/pix/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/pix/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/pix/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/pix/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/pix/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/pix/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

### Dense

#### Pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/dense/pretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/dense/pretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/dense/pretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/dense/pretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/pretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

#### No pretrain

##### 50% dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/dense/nopretrain/train/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/dense/nopretrain/train/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/train/floppa.png" width="128px"/></td>
</tr>
</tbody></table>

##### No dropout
<table><tbody>
<tr>
<td><img src="illustration/harmonic/dense/nopretrain/eval/mike_wazowski.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/horny.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/busya.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/ll.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/triplechad.png" width="128px"/></td>
</tr><tr>
<td><img src="illustration/harmonic/dense/nopretrain/eval/sigma.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/ginger.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/ka_52.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/pig.png" width="128px"/></td>
<td><img src="illustration/harmonic/dense/nopretrain/eval/floppa.png" width="128px"/></td>
</tr>
</tbody></table>



# Usage
```
git clone https://github.com/archqua/wc3icons.git
cd wc3icons
unzip path/to/weights.zip
./main.py -h
```

You can download weights
[here](https://drive.google.com/file/d/13JEiokUl_T-g1T5piqjnkCJWNZUHZbeV/view?usp=sharing).


## Dependencies
Pytorch, torchvision, pillow.



# Conclusion
The goal of creating nice wc3 icons from photos is not achieved.
Sometimes results kinda look somewhat good, but this is probably
due to the fact that ruining image a bit sometimes makes it look
more like wc3 icon.

Sometimes results look at least funny though.







