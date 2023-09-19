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
# Data preprocessing

(train_images, _), (test_images, _) = load_smallnorb()

train_images = train_images[:, :, :, 0:1] / 255.0
test_images = test_images[:, :, :, 0:1] / 255.0


# Function to create an array of random noise
def add_noise(arr, level):
    noise_arr = arr + level * np.random.normal(
        loc=0, scale=1.0, size=arr.shape
    )
    return noise_arr


# Function to display 2 sets of images
def display(arr1, arr2):
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(arr1, arr2)):
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(image1, cmap="gray")
        ax.axis("off")

        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(image2, cmap="gray")
        ax.axis("off")

    plt.show()


# Initial noise
noisy_test = add_noise(test_images, 1)

# Model definition

input = layers.Input(shape=(96, 96, 1))

# Downsampling and pooling
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Upsampling and realising skip connections
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)

x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

model = Model(input, x)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()


EPOCHS = 30
BATCH_SIZE = 128

training = False
noise_values = [0.1, 0.4, 0.7, 1.0]

# Model training
if training:
    for val in noise_values:
        noisy_train = add_noise(train_images, val)
        print("Training noise value : ", val)
        model.fit(
            x=noisy_train,
            y=train_images,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(noisy_test, test_images)
        )

        model.save(unet_save_name)
        predictions = model.predict(test_images[:10])
        display(noisy_test[:10], predictions)

else:
    model = keras.models.load_model(unet_save_name)
    test = np.random.normal(loc=0.0, scale=1, size=(10, 96, 96, 1))
    random_predictions = model.predict(test)
    display(test, random_predictions)
