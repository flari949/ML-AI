import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

import os
import pickle, gzip

from load_smallnorb import load_smallnorb

load_from_file = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
    os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'smallnorb_rwd%.1e_rdp%.1f_rbn%d_daug%d' % (
    0, 0.5, int(False), int(False)))
net_save_name = save_name + '_cnn1_net.h5'
checkpoint_save_name = save_name + '_cnn1_net.chk'
history_save_name = save_name + '_cnn1_net.hist'

(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Shuffle data between train and test sets
all_i = np.concatenate((train_images, test_images))
all_l = np.concatenate((train_labels, test_labels))
indices = np.random.permutation(len(all_i))
len1 = len(train_images)
train_images = all_i[indices[:len1]]
train_labels = all_l[indices[:len1]]
test_images = all_i[indices[len1:]]
test_labels = all_l[indices[len1:]]

# Initial shape images : [24300, 96, 96, 2]
# Initial shape labels : [24300, 5]

# Extend data through concatenating camera views
all_train_images = np.concatenate((train_images[:, :, :, 0:1], train_images[:, :, :, 1:2]))
all_train_labels = np.concatenate((train_labels[:, 2], train_labels[:, 2]))
# Reshaped images : [48600, 96, 96, 1]
# Reshaped labels : [48600, 5]

if load_from_file and os.path.isfile(net_save_name):
    # ***************************************************
    # * Loading previously trained neural network model *
    # ***************************************************

    # Load the model from file
    print("Loading neural network from %s..." % net_save_name)
    net = keras.models.load_model(net_save_name)

    # Load the training history - since it should have been created right after saving the model
    if os.path.isfile(history_save_name):
        with gzip.open(history_save_name) as f:
            history = pickle.load(f)
    else:
        history = []
else:

    # ----------------------------------------------------------------------------------------------
    # Data preprocessing

    # Determine proportional amount of data for validation
    n = len(all_train_images)
    n_valid = int(n * 0.33)

    # Create random permutation of list indices
    I = np.random.permutation(n)

    # Select the validation inputs and the corresponding labels
    x_val = all_train_images[I[:n_valid]]
    y_val = all_train_labels[I[:n_valid]]

    # Select the training input and the corresponding labels
    x_train = all_train_images[I[n_valid:]]
    y_train = all_train_labels[I[n_valid:]]

    # Augment the data
    data_gen = keras.preprocessing.image.ImageDataGenerator(
        zca_epsilon=1e-06,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True
    )

    data_gen.fit(x_train)
    data_train_aug = data_gen.flow(x_train, y_train)

    # ----------------------------------------------------------------------------------------------
    # Model declaration

    net = keras.models.Sequential()
    net.add(keras.Input(shape=(96, 96, 1)))
    net.add(layers.Rescaling(1. / 255))

    net.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='valid'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='valid'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Flatten())

    net.add(layers.Dense(128, activation='relu'))
    net.add(layers.Dense(512, activation='relu'))
    net.add(layers.Dropout(0.5))
    net.add(layers.Dense(5, activation='softmax'))

    net.summary()

    # Training callback to call on every epoch
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_name,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    net.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_info = net.fit(data_train_aug, validation_data=(x_val, y_val), epochs=300,
                         shuffle=True, callbacks=[model_checkpoint_callback])

    # Load the weights of the best model
    print("Loading best save weight from %s..." % checkpoint_save_name)
    net.load_weights(checkpoint_save_name)

    # Save the entire model to file
    print("Saving neural network to %s..." % net_save_name)
    net.save(net_save_name)

    # Save training history to file
    history = train_info.history
    with gzip.open(history_save_name, 'w') as f:
        pickle.dump(history, f)

    # Concatenate testing data for more in depth testing result
    full_test_set = np.concatenate((test_images[:, :, :, 0:1], test_images[:, :, :, 1:2]))
    full_test_labels = np.concatenate((test_labels[:, 2], test_labels[:, 2]))

    test_loss, test_acc = net.evaluate(full_test_set, full_test_labels)
    print(test_acc)
    print(test_loss)

    # Plot training and validation accuracy and loss progression over time
    acc = train_info.history['accuracy']
    val_acc = train_info.history['val_accuracy']
    plt.plot(acc, label="Training accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    loss = train_info.history['loss']
    val_loss = train_info.history['val_loss']
    plt.plot(loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



# *********************************************************
# * Training history *
# *********************************************************

if load_from_file:

    # Plot training and validation accuracy over the course of training
    if history != []:
        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************

    full_test_set = np.concatenate((test_images[:, :, :, 0:1], test_images[:, :, :, 1:2]))
    full_test_labels = np.concatenate((test_labels[:, 2], test_labels[:, 2]))

    loss_train, accuracy_train = net.evaluate(all_train_images, all_train_labels, verbose=0)
    loss_test, accuracy_test = net.evaluate(full_test_set, full_test_labels, verbose=0)

    print("Train accuracy (tf): %.4f" % accuracy_train)
    print("Test accuracy  (tf): %.4f" % accuracy_test)
