#!/usr/bin/env python

"""
  Apply Multi-Scale Retinex to an image
  Usage:
    First, activate conda environment: conda activate py312

    #Create conda environment:
        conda create -n py312 python=3.12 pip
        pip install numpy
        pip install scipy
        pip install jupyter notebook
        pip install opencv-contrib-python
        
#Examples: 
#  msretinex.py '/home/joseluis/Dropbox/images/lena.png' testmsr.png
#  msretinex.py '/home/joseluis/Dropbox/2023.underwater_style_transfer/data/GoodBad/high-haze/vivo_20181014_175959M.jpg' testmsr.png

#Comments: very slow compared with the C++ version, even if using FFT Gaussian convolution

"""

import sys

import argparse

parser = argparse.ArgumentParser(description="Multi-Scale Retinex")

# parser.add_argument("namein")
# parser.add_argument("nameout")

parser.add_argument("-l", "--lowscale", help="Lower scale", default=15)
parser.add_argument("-m", "--medscale", help="Middle scale", default=80)
parser.add_argument("-g", "--highscale", help="Higher scale", default=250)

parser.add_argument("-s", "--darkprc", help="Simplest color balance parameter (percentage of dark values set to 0)", default=1)
parser.add_argument("-t", "--brightprc", help="Simplest color balance parameter (percentage of bright values set to 255)", default=1)

parser.add_argument("-i", "--ongray", help="Apply on intensity", action="store_true")

args = parser.parse_args()

# namein = args.namein
# nameout = args.nameout


lowscale = args.lowscale
medscale = args.medscale
highscale = args.highscale
s1 = args.darkprc
s2 = args.brightprc
onGray = args.ongray


print("lowscale=%2.2f  medscale=%2.2f  highscale=%2.2f  s1=%2.2f  s2=%2.2f  onGray=%s"%(lowscale, medscale, highscale, s1, s2, str(onGray)))


import os
import cv2
import math
import numpy as np

#Use FFT convolution
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
import scipy
def gaussianblur_fft(im, sigma):
    
    smooth = scipy.ndimage.gaussian_filter(im, sigma) 

    return smooth
    
   
    
#Multi-Scale Retinex for one channel image
def msretinex(im, lowscale, medscale, highscale):

    #convolution with Gaussian
    smoothLow = gaussianblur_fft(im, lowscale)
    smoothMed = gaussianblur_fft(im, medscale)
    smoothHigh = gaussianblur_fft(im, highscale)
        
    #add 1 in log computation to prevent log(0)
    outLow = np.log(im+1) - np.log(smoothLow+1)
    outMed = np.log(im+1) - np.log(smoothMed+1)
    outHigh = np.log(im+1) - np.log(smoothHigh+1)
    
    out = (outLow + outMed + outHigh)/3
    
    return out

#Simplest Color Balance
def scb(im, s1, s2):

    [h,w] = im.shape #image dimensions
    
    imvec = im.flatten() #convert to 1D array
    indices = imvec.argsort() #get indices of sorted values (increasing order)
    
    ibottom = int(math.floor(s1 * w*h / 100)) #position of brightest of dark values to be mapped to 0
    itop = min( int(math.ceil((100-s2) * w*h / 100)), w*h-1) #position of darkest of bright values to be mapped to 255 (maximum index: w*h-1)
    
    dark = imvec[indices[ibottom]]
    bright = imvec[indices[itop]]
    
    #map input values from [dark, bright] to [0, 255], clip rest of values
    imvec = 255 * (imvec-dark) / (bright-dark) 
    for ind in range(1, ibottom):
       imvec[indices[ind]] = 0
    for ind in range(itop, w*h):
        imvec[indices[ind]] = 255 
    
    return imvec.reshape((h, w))
    

#Recover color from grayscale: RGBout = RGBin * outGray / inGray, therefore gray(RGBout) = outGray
def color_from_grayscale(r, g, b, gray, outgray):

    factors = outgray/gray
    factors = np.clip(factors, None, 3) #clip maximum value to 3
    
    #limit values of factors to prevent saturation in any channel
    maxchannel = cv2.max(cv2.max(r, g), b)
    maxchannel[maxchannel == 0] = 1 #if value is 0, replace by 1, to prevent division by zero in next command
    maxfactors = 255 / maxchannel
    factors = cv2.min(factors, maxfactors)   
    
    #apply factors to each channel
    outr = r * factors
    outg = g * factors
    outb = b * factors
    
    #check saturated values
    

    return outr, outg, outb


#MAIN Algorithm

def mainRetinex(img):

    src = img.copy()
    im = src.astype('float64')
    b, g, r = cv2.split(im)
    rgbsum = r+g+b

    if not onGray:
        outr = msretinex(r, lowscale, medscale, highscale)
        outg = msretinex(g, lowscale, medscale, highscale)
        outb = msretinex(b, lowscale, medscale, highscale)
        
        #color restoration
        outr = outr * np.log(125 * r/rgbsum + 1)
        outg = outg * np.log(125 * g/rgbsum + 1)
        outb = outb * np.log(125 * b/rgbsum + 1)
        
        #simplest color balance
        outr = scb(outr, s1, s2)
        outg = scb(outg, s1, s2)
        outb = scb(outb, s1, s2)
        
    else:
        outgray = msretinex(rgbsum/3, lowscale, medscale, highscale)
        outgray = scb(outgray, s1, s2)
        outr, outg, outb = color_from_grayscale(r, g, b, rgbsum/3, outgray)
        
    #save result 
    res = np.empty(im.shape,im.dtype);
    res[:, :, 0] = outb
    res[:, :, 1] = outg
    res[:, :, 2] = outr

    return res


