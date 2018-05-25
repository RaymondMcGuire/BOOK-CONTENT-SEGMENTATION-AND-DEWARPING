# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:55:21 2018

@author: raymondmg
"""
import os
import numpy as np
import PIL.Image
import random

HEIGHT = 800
WIDTH = 600


imgPathName = "/image/"
imgLabelName = "/label/" 

imgOutPathName = "/t_image/"
imgOutLabelName = "/t_label/" 

imgValOutPathName = "/v_image/"
imgValOutLabelName = "/v_label/" 

def read_dataset(data_dir):
    img_dir = data_dir + imgPathName
    label_dir = data_dir + imgLabelName
    img_array = []
    label_array = []
    for file in os.listdir(img_dir):  
        img_array.append(img_dir+file)
    for file in os.listdir(label_dir):  
        label_array.append(label_dir+file)
    if len(img_array) == len(label_array):
        perm = np.arange(len(img_array))
        np.random.shuffle(perm)
        
        img_array = np.array(img_array)[perm]
        label_array = np.array(label_array)[perm]
        return img_array[0:len(img_array)],label_array[0:len(img_array)]
    else:
        print("label number not match the image number!")
        return img_array,label_array

def identity_dataset(data_dir,imArray,labArray,trainPercent = 0.8):
    number = len(imArray)
    trainNumber = int(number * trainPercent)
    for i in range(number):
        imgName = imArray[i]
        labName = labArray[i]
        image = PIL.Image.open(imgName)
        label = PIL.Image.open(labName)
        width, height = image.size
        if width < height:
            #resize
            resize_image = image.resize((WIDTH,HEIGHT))
            resize_label = label.resize((WIDTH,HEIGHT))
        else:
            #reshape img shape to width>height,then resize
            newImage = np.asarray(image)
            outImage = np.zeros((width,height,3))
            newLabel = np.asarray(label)
            outLabel = np.zeros((width,height))
            for h in range(height):
                for w in range(width):
                    outImage[w,h] = newImage[h,w]
                    outLabel[w,h] = newLabel[h,w]
            outImage = PIL.Image.fromarray(np.uint8(outImage))
            outLabel = PIL.Image.fromarray(np.uint32(outLabel))
            resize_image = outImage.resize((WIDTH,HEIGHT))
            resize_label = outLabel.resize((WIDTH,HEIGHT))
        oImgName = imgName.split("/")[-1]
        oLabName = labName.split("/")[-1]
        
        if i <=trainNumber:
            resize_image.save(data_dir + imgOutPathName +oImgName)
            resize_label.save(data_dir + imgOutLabelName +oLabName)
        else:
            resize_image.save(data_dir + imgValOutPathName +oImgName)
            resize_label.save(data_dir + imgValOutLabelName +oLabName)

img_array,label_array = read_dataset("./data")
identity_dataset("./data",img_array,label_array)