# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:12:58 2018

@author: raymondmg
"""
import os
import numpy as np
import PIL.Image
import cv2

imgPathName = "image/"
imgLabelName = "label/" 

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
        return img_array,label_array
    else:
        print("label number not match the image number!")
        return img_array,label_array

def visualize(image):
    vis_img = np.zeros((image.shape[0],image.shape[1],3))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                vis_img[i,j] = (255,0,0) 
            elif image[i][j] == 1:
                vis_img[i,j] = (255,255,0)
            elif image[i][j] == 2:
                vis_img[i,j] = (0,255,255) 
    return vis_img
        
class BatchDataSet:

    def __init__(self, train_image_list,train_label_list,resize_flag=True,resize_size=(500,500)):
        self.train_image_list = train_image_list
        self.train_label_list = train_label_list
        self.resize_flag = resize_flag
        self.resize_size = resize_size
        self.read_images()
        self.batch_offset = 0
        self.epochs_completed = 0

    def read_images(self):
        self.images = np.array([self.transform(filename) for filename in self.train_image_list])
        self.annotations = np.array([np.expand_dims(self.transform(filename),axis=3) for filename in self.train_label_list])

    def transform(self, filename):
        image =  PIL.Image.open(filename)
        if self.resize_flag and self.resize_size:
            resize_image = image.resize(self.resize_size)
        else:
            resize_image = image
        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
    def get_all_batch(self):
        return self.images, self.annotations
