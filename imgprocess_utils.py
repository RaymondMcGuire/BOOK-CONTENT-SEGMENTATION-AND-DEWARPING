# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:30:59 2017

@author: raymondmg
"""
import cv2
import numpy as np
import random


def saturate_cast(val):
    if val < 0:
        val = 0
    elif val > 255:
        val = 255
    return val

def addweight_img2black(img,percentage):
    blackimg=img.copy()   
    height =img.shape[0]
    width  =img.shape[1]
    for j in range(0,height):
        for i in range(0,width):
            blackimg[j,i]=0
   
    img2black = cv2.addWeighted(img,percentage,blackimg,(1-percentage),0)   
    return img2black  

def getrandom(min_val,max_val):
    return random.randint(min_val, max_val)


def getrandomlightcolor(n):
    
    color_dataset = []
    color_dataset.append([109, 174, 210])
    color_dataset.append([47, 219, 255])
    color_dataset.append([238, 255, 255])
    
    index = getrandom(0, n-1)
    
    return color_dataset[index]

def draw_rectangle(img,left,right,color,thickness):
    cv2.rectangle(img,left,right,color,thickness)

def connectcomponent(thresholdimg):
    n_labels, label_image = cv2.connectedComponents(thresholdimg)
    print("connect component processed, labels num:"+ str(n_labels))
    return  n_labels, label_image


def thresholdprocess(img,minsize,maxsize):
    #gray2binary
    ret,threshtobinary = cv2.threshold(img,minsize,maxsize,cv2.THRESH_BINARY)
    return threshtobinary

def getkernel(row,col):
    return np.ones((row,col),np.uint8)

def readimage(path):
    return cv2.imread(path)

def readgrayimage(path):
    return cv2.imread(path,0)

def resizeimage(img,width,height,flag=cv2.INTER_CUBIC):
    return cv2.resize(img,(width,height),interpolation=flag)

def saveimage(img,path,flag):
    if flag:
        cv2.imwrite(path,img)
        
def convbgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

