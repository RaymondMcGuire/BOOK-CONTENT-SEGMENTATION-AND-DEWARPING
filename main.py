# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:55:35 2018

@author: raymondmg
"""
import tensorflow as tf
import numpy as np
import datetime
import fcn
import data_process as dp
import scipy.misc as misc
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "./logs/", "path to logs directory")
tf.flags.DEFINE_string("vgg_dir", "./pre_data/imagenet-vgg-verydeep-19.mat", "path to vgg directory")
tf.flags.DEFINE_string("data_train_dir", "./data_book/", "path to train dataset")
tf.flags.DEFINE_string("data_valid_dir", "./data_book_valid/", "path to valid dataset")
tf.flags.DEFINE_float("learning_rate", "1e-6", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = 20000
NUM_OF_CLASSESS = 3
IMAGE_SIZE_HEIGHT = 400
IMAGE_SIZE_WIDTH = 300

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def save_image(image, save_dir, name):
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

def main(argv=None):
    
    #param
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None,  IMAGE_SIZE_HEIGHT,IMAGE_SIZE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None,IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 1], name="annotation")

    fcn_net= fcn.FCN(FLAGS.vgg_dir,keep_probability,NUM_OF_CLASSESS)
    pred_annotation, logits = fcn_net.build_network(image)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy")))
    
    tf.summary.scalar("entropy", loss)
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)
    summary_op = tf.summary.merge_all()
    img_array,label_array = dp.read_dataset(FLAGS.data_train_dir)
    img_valid_array,label_valid_array = dp.read_dataset(FLAGS.data_valid_dir)
    
    #setting
    if FLAGS.mode == 'train':
        train_dataset_reader = dp.BatchDataSet(img_array,label_array,True,(IMAGE_SIZE_WIDTH,IMAGE_SIZE_HEIGHT))
    elif FLAGS.mode == 'test':
        test_dataset_reader = dp.BatchDataSet(img_valid_array,label_valid_array,True,(IMAGE_SIZE_WIDTH,IMAGE_SIZE_HEIGHT))
    
    validation_dataset_reader = dp.BatchDataSet(img_valid_array, label_valid_array,True,(IMAGE_SIZE_WIDTH,IMAGE_SIZE_HEIGHT))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")


    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            #print(train_images.shape,train_annotations.shape)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)
    elif FLAGS.mode == "test":
        test_images, test_annotations = test_dataset_reader.get_random_batch(3)
        pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations,
                                                    keep_probability: 1.0})
        test_annotations = np.squeeze(test_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(3):
            save_image(test_images[itr].astype(np.uint8), FLAGS.pred_dir, name="inp_" + str(itr))
            save_image(dp.visualize(test_annotations[itr].astype(np.uint8)), FLAGS.pred_dir, name="gt_" + str(itr))
            save_image(dp.visualize(pred[itr].astype(np.uint8)), FLAGS.visual_dir, name="visual_" + str(itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()