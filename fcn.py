# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:32:59 2018

@author: raymondmg
"""
import scipy.io
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class FCN:
    def __init__(self,vgg16_npy_path,keep_prob,class_num = 151):
        model_data = scipy.io.loadmat(vgg16_npy_path)
        mean = model_data['normalization'][0][0][0]
        self.mean_pixel = np.mean(mean, axis=(0, 1))
        self.weights = np.squeeze(model_data['layers'])
        self.keep_prob = keep_prob
        self.class_num = class_num
        print("npy file loaded")
        
    def conv2d_transpose_strided(self, x, W, b, output_shape=None, stride = 2):
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)
        
    def weight_variable(self,shape, stddev=0.02, name=None):
        initial = tf.truncated_normal(shape, stddev=stddev)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)
        
    def get_variable(self,weights, name):
        init = tf.constant_initializer(weights, dtype=tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
        return var
    
    def bias_variable(self,shape, name=None):
        initial = tf.constant(0.0, shape=shape)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)
    
    def conv2d(self,x, W, bias):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(conv, bias)

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def avg_pool(self,x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu
    
    def vgg_net(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
    
        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                kernels = self.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = self.get_variable(bias.reshape(-1), name=name + "_b")
                current = self.conv2d(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = self.avg_pool(current)
            net[name] = current
    
        return net
        
    def build_network(self,image):
        processed_image = image-self.mean_pixel
        image_net = self.vgg_net(self.weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        self.pool5 = self.max_pool(conv_final_layer)

        #fully convolutional
        self.fc6 = self.conv2d(self.pool5,self.weight_variable([7, 7, 512, 4096], name="w6"),self.bias_variable([4096],name="b6"))
        self.relu6 = tf.nn.relu(self.fc6, name="relu6")
        self.dropout6 = tf.nn.dropout(self.relu6, keep_prob=self.keep_prob)
        
        self.fc7 = self.conv2d(self.dropout6,self.weight_variable([1, 1, 4096, 4096], name="w7"),self.bias_variable([4096],name="b7"))
        self.relu7 = tf.nn.relu(self.fc7, name="relu7")
        self.dropout7 = tf.nn.dropout(self.relu7, keep_prob=self.keep_prob)
        
        self.fc8 = self.conv2d(self.dropout7,self.weight_variable([1, 1, 4096, self.class_num], name="w8"),self.bias_variable([self.class_num],name="b8"))
        
        #upscale
        self.pool3 = image_net["pool3"]
        self.pool4 = image_net["pool4"]
        
        self.deconv1 = self.conv2d_transpose_strided(self.fc8,self.weight_variable([4, 4, self.pool4.get_shape()[3].value, self.class_num], name="dw1"),self.bias_variable([self.pool4.get_shape()[3].value],name="db1"),output_shape=tf.shape(self.pool4))
        self.fuse1 = tf.add(self.deconv1, self.pool4, name="fuse1")
        
        self.deconv2 = self.conv2d_transpose_strided(self.fuse1,self.weight_variable([4, 4, self.pool3.get_shape()[3].value, self.pool4.get_shape()[3].value], name="dw2"),self.bias_variable([self.pool3.get_shape()[3].value],name="db2"),output_shape=tf.shape(self.pool3))
        self.fuse2 = tf.add(self.deconv2, self.pool3, name="fuse2")
        
        shape = tf.shape(image)
        self.deconv3 = self.conv2d_transpose_strided(self.fuse2,self.weight_variable([16, 16, self.class_num, self.pool3.get_shape()[3].value], name="dw3"),self.bias_variable([self.class_num],name="db3"),output_shape=tf.stack([shape[0], shape[1], shape[2], self.class_num]),stride=8)
        
        annotation_pred = tf.argmax(self.deconv3, dimension=3, name="prediction")
        
        return tf.expand_dims(annotation_pred, dim=3), self.deconv3