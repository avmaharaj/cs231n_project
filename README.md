#Combatting Adversarial Examples With Discretization of Pixels


Code and analysis from CS231N project on combatting adversarial examples by discretization of input pixels. 

The basic idea of the project was to reduce the dimensionality of the input images, by discretizing/clamping pixel intensities. We found no significant drop in the accuracy of a variety of both shallow (5 Layer) and deep ConvNets (GoogleNet, CaffeNet) when input pixels were reduced to just 16 allowed intensities (instead of the default 255 allowed in standard image encodings)

This repo contains the trained caffe models, training logs, taining and deployment prototxt files. Also included are python scripts for processing data for input, and ipython notebooks analyzing the data, and used for generating plots for my final report (pdf included). Some links in Ipython notebooks will need to be modified accordingly. 

