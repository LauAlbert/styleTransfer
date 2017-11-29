# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:16:10 2017

@author: Albert-Desktop
"""


from PIL import Image
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.misc import imsave

# Set image height and width
height = 512
width = 512
numberOfChannels = 3

# WEIGHTS For Total Variance/ Total Loss
# mess with there weights for different result
content_weight = 0.025 #Alpha
style_weight = 5.0 #Beta

# Get and resize image the height and width
# MAIN IMAGE
content_path = "D:\\Downloads\\image.jpg"
content_img = Image.open(content_path).resize((width, height))

# STYLE IMAGE
style_path = "D:\\Downloads\\style4.jpg"
style_img = Image.open(style_path).resize((width, height))

# Preprocess image function
def preprocessImage(img):
    img = np.asarray(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    img[:,:,:,0] -= 103.939
    img[:,:,:,1] -= 116.779
    img[:,:,:,2] -= 123.68
    img = img[:,:,:,::-1]
    return img

# Undo the prepocess for to combine image
def undoPreprocessImage(tensor, index):
    img = tensor[index].copy()
    img = img[:,:,::-1]
    img[:,:,0] += 103.939
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Preprocess image to work with VGG19
content_image_array = preprocessImage(content_img)

style_image_array = preprocessImage(style_img)

# Setup placeholder for input to VGG16/VGG19 model
input_tensor = tf.placeholder(tf.float32, shape=(1, height, width, numberOfChannels))

# GET VGG19 model
base_model = tf.keras.applications.VGG19(include_top=False, pooling='avg', weights="imagenet", input_tensor=input_tensor)

# layer use in content transfer
layerName = 'block4_conv4'

# layer use in style transfer
feature_layers = ['block1_conv1', 
                  'block2_conv2',
                  'block3_conv1',
                  'block4_conv1',
                  'block5_conv1']



# Calculate the content loss
def content_loss(sess, model, content_image, layerName):
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    layerRef = layers[layerName]
    value = sess.run(layerRef, feed_dict={input_tensor: content_image})
    with model.graph.as_default():
        value = tf.constant(value)
        return tf.reduce_sum(tf.square(layerRef-value)) / 2.0


# Calculate the style loss
def style_loss(sess, model, style_image, feature_layers):
    def gram_matrix(tensor, numberOfFilters):
        features = tf.reshape(tensor, shape=(-1, numberOfFilters))
        return tf.matmul(tf.transpose(features), features)
    
    def style_loss_helper(style_feature, transfer_feature):
        # compute style loss for each feature layer
        numberOfFilters = style_feature.shape[3]
        imageHeightWidth = style_feature.shape[1] * style_feature.shape[2]
        A = gram_matrix(style_feature, numberOfFilters)
        G = gram_matrix(transfer_feature, numberOfFilters)
        return tf.reduce_sum(tf.square(G-A)) / (4.0 * (numberOfFilters**2) * (imageHeightWidth**2))
    
    loss = 0.0
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    with model.graph.as_default():
        # Loop though all feature layers and caculate 
        for feature_layer in feature_layers:
            layerRef = layers[feature_layer]
    
            value = sess.run(layerRef, feed_dict={input_tensor: style_image})
            # compute mean square error of feature layers
            loss += 1/len(feature_layers) * style_loss_helper(value, layerRef)
    return loss
   

# generate noise image
input_image = np.random.rand(*content_image_array.shape) + 128


# Amount of iteration
numberOfSteps = 40

# learning rate/ is not use in my code becuase the takes too long to minimize total loss
learningRate = 10


# Create tensorflow session
sess = tf.Session()

# Initial Variables
sess.run(tf.global_variables_initializer())

# Content Loss
contentLoss = content_loss(sess, base_model, content_image_array, layerName)

# Style Loss
styleLoss = style_loss(sess, base_model, style_image_array, feature_layers)

# Total loss
totalLoss =  content_weight * contentLoss + style_weight * styleLoss

#sess.run(base_model.input.assign(input_image))

gradient = tf.gradients(ys=totalLoss, xs=[base_model.input])


for step in range(numberOfSteps):
    # minimize the loss of the total loss
    grad = sess.run(gradient, feed_dict={input_tensor: input_image})[0]
    
    # Update the image
    step_size_scaled = 20 / (np.std(grad) + 1e-8)
    input_image -= grad * step_size_scaled
#    input_image = np.clip(input_image, 0.0, 255.0)
    
    # Every 10 display the image
    if((step+1)%1 == 0):
        print(step)
#        print(sess.run(totalLoss))
        i = input_image
        result = undoPreprocessImage(i, 0)
        print("unpreproccess")
        plt.imshow(i[0])
        plt.show()
        plt.imshow(result)
        plt.show()
imsave('test1.png', i[0])
imsave('test.png', result)

sess.close()