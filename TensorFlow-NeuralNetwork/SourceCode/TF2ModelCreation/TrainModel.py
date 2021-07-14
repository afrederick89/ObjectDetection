import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from matplotlib import image
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

# Basic TF Keras Model
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(200, 200, 4)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Add tensorboard callbacks for tensor board visualization
log_dir = filepath + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with training images and labels
model.fit(train_images, train_labels, epochs=3,callbacks=[tensorboard_callback])
# Test the model using testing images
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Create probability model to predict images within TensorFlow
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Print out 15 images and make predictions based on prediction model + testing images
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Frozen Graph Generation For OpenCV Loading
frozen_graph_filename = "TF2Model_Frozen_Graph"
frozen_out_path = filepath

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)
