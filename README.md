# BOOK-CONTENT-SEGMENTATION-AND-DEWARPING
OverView:Using FCN to segment the book's content and background, then dewarping the pages.

Last Updated Code:2018.05.25
Continuing......

First Step:
# BOOK-CONTENT-SEGMENTATION
Using FCN(fully convolution network) to segment the image into 3 parts(left page,right page and background).

DataSet Created By Lin YangBin
Now we have 500 images and labeled images for book pages. 

Using GTX 1070 8GB Trained this network for 8 hours. loss value ->0.01

TODO:Data Augment and Dewarp Algorithm.

# USING BOOK-CONTENT-SEGMENTATION
1.You should download the trained VGG model from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat, put this model into the folder "./pre_data".

2.Prepared the dataset for training and validation.

3.cmd->cd to your path->python main,py->running(first training your model use this network, then change the flag.mode to "test" mode or "visual" mode to identify the results).
