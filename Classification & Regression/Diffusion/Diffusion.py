import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from load_smallnorb import load_smallnorb

# //////////////////////////////////////////////////////////////////////////////////

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("Diff_saved"):
    os.mkdir('Diff_saved')

# Specify the names of the save files
save_name = os.path.join('Diff_saved', 'smallnorb_diffusion')
gen_net_save_name = save_name + 'gen_net.h5'
disc_net_save_name = save_name + 'disc_net.h5'

# //////////////////////////////////////////////////////////////////////////////////

(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Data preprocessing
train_images = train_images[:, :, :, 0:1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 24300
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Function to create model that trains to generate training data from noise,
# training based on response from discriminator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(24 * 24 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((24, 24, 256)))
    assert model.output_shape == (None, 24, 24, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 24, 24, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 48, 48, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, 1)

    return model


generator = make_generator_model()


# Function to create model that trains to distinguish between real training data and fake
# emulated data created by the generator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[96, 96, 1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()

# Define loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Define the loss returned by the discriminator model
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Define the loss returned by the generator model
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Define model optimizers and learning rates
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


# May not be needed depending on tf version ->
@tf.function
# Function to define training steps for generator and discriminator models
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Training functions to train generator and discriminator model simultaneously
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

    generate_and_save_images(generator, epochs, seed)
    generator.save(gen_net_save_name)
    discriminator.save(disc_net_save_name)


# Function to print and save images showing diffusion progress
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # Commented out as will override existing images
    # img_save_name = save_name + 'epoch_{:03d}.png'.format(epoch)
    # plt.savefig(img_save_name)
    plt.show()


# Function to showcase saved model functionality on random noise
def show_model():
    gen_model = keras.models.load_model(gen_net_save_name)

    seed_val = tf.random.normal([16, 100])
    predictions = gen_model(seed_val, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show(block=True)


# --- Function calling ---

# train(train_dataset, EPOCHS)
show_model()
