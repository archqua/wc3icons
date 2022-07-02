# wc3icons
This repository is final project for 2022 spring DLS season.
The idea is to train generative adversarial network to
turn any picture into Warcraft III icon.



# Data
Most of the icons to train on are taken from here(TODO insert link).
These are not actual wc3 icons -- these are adjusted World of Warcraft icons.
The rest of icons are actual wc3 icons taken from here(TODO insert link).
Only icons for buttons were eventually used, which turned out to be
unnecessary, but it doesn't seem to be huge loss anyway.

There are 22529 of adjusted WoW icons and 1007 of wc3 icons, 23536 overall.

Non-wc3 pictures are taken from several datasets(TODO insert links):
1. fruits
2. dogs and cats
3. monkeys
4. fashion
5. buildings
6. weather
7. food
8. animals
<!-- 9. birds -->
9. faces

Datasets were only used partially, there are
243 photos of fruits and vegetables, 5050 of dogs and cats,
1097 of monkeys, 4518 of fashion items, 3523 of buildings,
1127 of landscapes, 1500 of food and 1500 of non-food,
5272 of animals and finally 3143 of faces.
Which is 26973 photos overall.



# Preprocessing
Each non-icon was cropped, resized to $56 \times 56$ and shadowed at the edges.
TODO links to archives.



# Models
Two models were tested. One is pix2pix-like(TODO insert link) and can be found
in pix.py, the other one mixes pix2pix with dense-net(TODO insert link)
and can be found in dense\_pix.py.



# Training
There are three train loops, all can be found in train\_util.py.
Loss functions were chosen to be MSE, rather than BCE.


## Simple train loop
First one is simple train loop:
1. make transformator generate icons from photos
2. train discriminator on real icons and fakes
3. make transformator generate icons from photos
4. train transformator on real icons and fakes


## Cycle train loop
Is inspired by CycleGAN(TODO insert link).

Train forward and backward transformator and discriminator
as in simple train loop with addition to cycle consistency loss.
Cycle consistency loss is MSE.


## Harmonic train loop
Is inspired by HarmonicGAN(TODO insert link).

Train forward and backward transformator and discriminator
as in simple train loop with addition to harmonic loss.
Unlike the original, cycle consistency loss is ignored here.
The intention here is to preserve trained model from
"willing" to preserve pictures unchanged.
To calculate harmonic loss, each picture was split into 16 cells
($4 \times 4$).
The implementation of harmonic loss can be found in harmonic\_loss.py.

## Pretraining
In some cases, in order to initialize weights, transformators were for 1 epoch
(pre)trained in autoencoder manner with L1 loss function.


## Parameters
Inspired by DCGAN(TODO insert link), learning rates were chosen to be
$10^{-4}$ for transformators and $10^{-5}$ for discriminators,
betas were chosen to be $(TODO)$.


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
[](illustration/test\_orig/busya.jpg)
[](illustration/test\_orig/floppa.jpg)
[](illustration/test\_orig/ginger.jpg)
[](illustration/test\_orig/horny.jpg)
[](illustration/test\_orig/ka\_52.jpg)
[](illustration/test\_orig/ll.jpg)
[](illustration/test\_orig/mike\_wazowski.jpg)
[](illustration/test\_orig/pig.jpg)
[](illustration/test\_orig/sigma.jpg)
[](illustration/test\_orig/triplechad.jpg)


# Results
TODO

