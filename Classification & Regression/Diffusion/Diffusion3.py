import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from keras import layers
from keras.models import Model

from load_smallnorb import load_smallnorb

# //////////////////////////////////////////////////////////////////////////////////

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("Diff2_saved"):
    os.mkdir('Diff2_saved')

# Specify the names of the save files
save_name = os.path.join('Diff2_saved', 'smallnorb_diffusion_')
unet_save_name = save_name + 'unet.h5'

# //////////////////////////////////////////////////////////////////////////////////
# Data importing

(train_images, _), (test_images, _) = load_smallnorb()

# Image sized reduced for computational efficiency
train_images = train_images[:, 16:80, 16:80, 0:1] / 255.0
test_images = test_images[:, 16:80, 16:80, 0:1] / 255.0

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////
# Data preprocessing - (3 methods of noise application)


# Function to add noise to image in specified quantity, each layer applies greater noise to a base image,
# no relation between layers
def gen_noise(arr, num_levels):
    outputs = arr
    noise = np.random.normal(
        loc=0.0, scale=1.0, size=arr.shape
)
    for val in range(num_levels-1):
        next_arr = arr + ((val+1) * (1 / num_levels) * noise)
        outputs = np.concatenate((outputs, next_arr), axis=0)
    inputs = np.concatenate((outputs[arr.shape[0]:], noise), axis=0)
    return inputs, outputs


# Function to add same noise element wise per filter - noise steps identical for each image
def gen_noise2(arr, num_levels):
    outputs = arr
    arr_len = arr.shape[0]
    for _ in range(num_levels):
        noise = np.random.normal(size=arr[0].shape)
        outputs = np.concatenate((outputs, outputs[-arr_len:]), axis=0)
        for i in range(arr_len + 1):
            outputs[-(i+1)] = outputs[-(i+1)] + 0.15 * noise
    inputs = outputs[arr_len:]
    outputs = outputs[:-arr_len]
    return inputs, outputs


# Function to generate training data and targets through sequential random noising
def gen_noise3(arr, num_levels):
    outputs = arr
    arr_len = arr.shape[0]
    for _ in range(num_levels-1):
        noise = np.random.normal(size=arr.shape)
        curr_arr = outputs[-arr_len:] + 0.01 * noise
        outputs = np.concatenate((outputs, curr_arr), axis=0)
    noise = np.random.normal(size=arr.shape)
    new_arr = outputs[-arr_len:] + 0.01 * noise
    inputs = np.concatenate((outputs[arr_len:], new_arr), axis=0)
    return inputs, outputs


# Function to display 2 sets of 10 images
def display(arr1, arr2):
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(arr1[:10], arr2[:10])):
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(image1, cmap="gray")
        ax.axis("off")

        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(image2, cmap="gray")
        ax.axis("off")

    plt.show()

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////
# Model block definitions


# Define central bottleneck block also used in extending blocks
def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu")(x)
    return x


# Define downsample block
def downsample_block(x, n_filters):
    conv_feature = double_conv_block(x, n_filters)
    pool = layers.MaxPool2D(2)(conv_feature)
    return conv_feature, pool


# Define upsample block
def upsample_block(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = double_conv_block(x, n_filters)
    return x


# /////////////////////////////////////////////////////////////////////
# Model creation and implementation


# Create unet model
def build_unet_model():
    inputs = layers.Input(shape=(64, 64, 1))

    feature1, down_block1 = downsample_block(inputs, 64)
    feature2, down_block2 = downsample_block(down_block1, 128)

    bottleneck = double_conv_block(down_block2, 256)

    up_block1 = upsample_block(bottleneck, feature2, 128)
    up_block2 = upsample_block(up_block1, feature1, 64)

    outputs = layers.Conv2D(1, 3, padding="same")(up_block2)
    model = keras.Model(inputs, outputs)
    return model


unet_model = build_unet_model()
unet_model.compile(optimizer="adam", loss="mse", metrics="mae")

unet_model.summary()

# Number of noise layers for model to be trained on
NOISE_SPLITS = 1000
training_data, training_label = gen_noise3(train_images, NOISE_SPLITS)

# Show the first 10 images under steps 0 and 1, then steps 3 and 4
display(training_label[:10], training_data[:10])
skip = 2 * train_images.shape[0]
display(training_label[skip:(skip + 10)], training_data[skip:(skip + 10)])

# Show the final 2 steps for 10 images
display(training_label[-train_images.shape[0]:], training_data[-train_images.shape[0]:])

BATCH_SIZE = 128
EPOCHS = 50
train_info = unet_model.fit(
            x=training_data,
            y=training_label,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True)

# Show model passed random noise every so often layer-split interval
predictions = np.random.normal(size=(10, 64, 64, 1))
for rep in range(NOISE_SPLITS):
    prev = predictions
    predictions = unet_model.predict(predictions)
    if (rep % 100) == 0:
        display(prev, predictions)
