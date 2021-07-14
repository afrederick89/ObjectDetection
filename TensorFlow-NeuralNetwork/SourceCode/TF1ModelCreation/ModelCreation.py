import numpy as np
from os import listdir
from matplotlib import image
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import datetime


filepath = 'C:\\Users\\Ryan\\Desktop\\Fullerton Fall 2020\\Tuesday.Thursday - CPSC 481 - Artificial Intelligence\\Final\\'

# Generate lists for training/testing images and labels
train_images = list()
train_labels = list()
train_images_path = filepath + 'TrainingImages\\'

test_images = list()
test_labels = list()
test_images_path = filepath + 'TestingImages\\'

class_names = ['Iron Ore']

# Load images into training list
for filename in listdir(train_images_path):
    # load image
    img_data = image.imread(train_images_path + filename)
    # store loaded image
    train_images.append(img_data)

# Load labels into training list
for x in range(0, len(train_images)):
    train_labels.append(0)

# Convert to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Load images into testing list
for filename in listdir(test_images_path):
    # load image
    img_data = image.imread(test_images_path + filename)

    test_images.append(img_data)

# Load labels into testing data
for x in range(0, len(test_images)):
    test_labels.append(0)

# Convert to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Create model within session
with tf.Graph().as_default():
    with tf.Session() as sess:
        # model creation and compilation
        model = tf.keras.Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=(200,200,4)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(64))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        # Attempt to generate just config file to load in openCV
        tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModelConfig.pbtxt', as_text=True)

        log_dir = filepath + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Train Model
        model.fit(train_images, train_labels, epochs=3,callbacks=[tensorboard_callback])

        output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print(output_node_names)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)
        # Generates frozen graph
        with open('TF1Model_Frozen_Graph.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

        # Saves the current model in the saved model format with checkpoints and meta data
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, './tensorflowModel.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)