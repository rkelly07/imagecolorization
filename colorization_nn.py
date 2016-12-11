#Attempt at conv nn as defined by http://richzhang.github.io/colorization/
from tflearn import *
import tensorflow as tf
import numpy as np

X = np.load('flowersbw.npy')
Y = np.load('flowers.npy')
#Y = tf.placeholder(tf.float32, [None, 256,256,3])
#Can use dropout to throw away some data randomly during training to prevent over-fitting
#network = dropout(network, 0.5)

# Building deep neural network, input is black and white photo
input_layer = input_data(shape=[None, 256, 256])
input_layer = tf.reshape(input_layer, [-1,256,256,1])

#conv1 256x256x1
network = conv_2d(input_layer, 64, 3, activation='relu')

network = conv_2d(network, 64, 3, activation='relu', strides=2)
network = batch_normalization(network)

#conv2 128x128x64
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu', strides=2)
network = batch_normalization(network)

#conv3 64x64x128
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu', strides=2)
network = batch_normalization(network)

#conv4 32x32x256
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = batch_normalization(network)

#conv5 32x32x512
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = batch_normalization(network)

#conv6 32x32x512
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = batch_normalization(network)

#conv7 32x32x512
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = batch_normalization(network)

#conv8 32x32x512
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d_transpose(network, 256 , 3, [64,64,256], activation='relu', strides = 2)

#conv8 64x64x256
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d_transpose(network, 128 , 3, [128,128,128], activation='relu', strides = 2)

#conv8 128x128x128
network = conv_2d(network, 6, 3, activation='relu')
network = conv_2d_transpose(network, 3 , 3, [256,256,3], activation='relu', strides=2)
loss = losses.L2(Y-network)

network = regression(network, optimizer='Adam',
                     loss=loss,
                     learning_rate=0.0001)

model = DNN(network, checkpoint_path='saveFiles/colorizer',
                    max_checkpoints=2, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")
#model = DNN(network)

#model.load('Colorize')

model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=1000,
          snapshot_epoch=False, run_id='flowers')

model.save('Colorize')

#Probability Distribution?
