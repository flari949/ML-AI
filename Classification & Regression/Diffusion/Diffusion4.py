import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

from load_smallnorb import load_smallnorb

# //////////////////////////////////////////////////////////////////////////////////

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("Diff2_saved"):
    os.mkdir('Diff2_saved')

# Specify the names of the save files
save_name = os.path.join('Diff2_saved', 'smallnorb_diffusion_final_')
unet_save_name = save_name + 'unet.h5'

# //////////////////////////////////////////////////////////////////////////////////
# Data preparing

(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Crop and normalise image values - 64x64 and [-1, 1] respectively
train_images = train_images[:, :, :, 0:1] / 255


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


# Define the diffusion model architecture
def diffusion_model(input_shape, num_levels):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Add num_levels noise diffusion steps
    for i in range(num_levels):
        x = add_noise(x, i + 1)

    feature1, down_block1 = downsample_block(x, 32)
    feature2, down_block2 = downsample_block(down_block1, 64)
    bottleneck = double_conv_block(down_block2, 128)
    up_block2 = upsample_block(bottleneck, feature2, 64)
    up_block3 = upsample_block(up_block2, feature1, 32)

    # Add a final convolutional layer to generate the output image
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="linear")(up_block3)

    # Define the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


# Define the function to add noise to an image
def add_noise(image, level):
    # Define the standard deviation of the Gaussian noise
    sigma = np.sqrt(level) / (2 * np.sqrt(2 * np.log(10)))

    # Generate the Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), stddev=sigma)

    # Add the noise to the image
    noisy_image = image + noise

    return noisy_image


# Resize images to 64x64
x_train = train_images
x_train = tf.image.resize(x_train, [64, 64])

# Define the input shape of the diffusion model
input_shape = (64, 64, 1)

# Define the number of noise diffusion levels
num_levels = 15

# Create the diffusion model
model = diffusion_model(input_shape, num_levels)

# Compile the model
model.compile(optimizer="adam", loss="mse")

training = False

if training:
    # Train the model on the SmallNORB dataset
    model.fit(x_train, x_train, epochs=20, batch_size=32)

    # Save model
    model.save(unet_save_name)

else:
    model = keras.models.load_model(unet_save_name)

# Generate a new image from random noise
noise = tf.random.normal(shape=(1, 64, 64, 1))
generated_image = model.predict(noise)

# Display the generated image
plt.imshow(generated_image[0, :, :, 0], cmap="gray")
plt.show()
