from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

Logger = tf.get_logger()
Logger.setLevel(logging.ERROR)

Dataset, Metadata = tfds.load('mnist', as_supervised = True, with_info = True)
Train_dataset, Test_dataset = Dataset['train'], Dataset['test']

Class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

Num_train_examples = Metadata.splits['train'].num_examples
Num_test_examples = Metadata.splits['test'].num_examples

# Transform 0 to 255 number scale to binary
def Normalize(Img, Label): 
    Img = tf.cast(Img, tf.float32)
    Img /= 255
    return Img, Label

Train_dataset = Train_dataset.map(Normalize)
Test_dataset = Test_dataset.map(Normalize)

# Classification of neural network structure
Model = tf.keras.Sequential([ 
	tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
	tf.keras.layers.Dense(64, activation = tf.nn.relu),
	tf.keras.layers.Dense(64, activation = tf.nn.relu),
	tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# Used functions
Model.compile(
	optimizer = 'adam',
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy']
)

# Batch learning of 32 each
BATCHSIZE = 32
Train_dataset = Train_dataset.repeat().shuffle(Num_train_examples).batch(BATCHSIZE)
Test_dataset = Test_dataset.batch(BATCHSIZE)

# Learning process
Model.fit(
	Train_dataset, epochs = 5,
	steps_per_epoch = math.ceil(Num_train_examples / BATCHSIZE)
)

# Evaluate trained model against the test dataset
Test_loss, Test_accuracy = Model.evaluate(
	Test_dataset, steps = math.ceil(Num_test_examples / 32)
)

print("Tests accuracy results: ", Test_accuracy)

for Test_img, Test_label in Test_dataset.take(1):
	Test_img = Test_img.numpy()
	Test_label = Test_label.numpy()
	Predictions = Model.predict(Test_img)

def ImgPlot(i, Predictions_array, True_labels, Img):
	Predictions_array, true_label, Img = Predictions_array[i], True_labels[i], Img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(Img[..., 0], cmap = plt.cm.binary)

	Predicted_label = np.argmax(Predictions_array)
	if Predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("Prediction: {}".format(Class_names[Predicted_label]), color=color)

def ArrayPlotValue(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	Plot = plt.bar(range(10), predictions_array, color = "#888888")
	plt.ylim([0,1])
	predicted_label = np.argmax(predictions_array)

	Plot[predicted_label].set_color('red')
	Plot[true_label].set_color('green')

N_rows = 5
N_cols = 3
N_imgs = N_rows * N_cols

plt.figure(figsize = (2 * 2 * N_cols, 2 * N_rows))
for i in range(N_imgs):
	plt.subplot(N_rows, 2 * N_cols, 2 * i + 1)
	ImgPlot(i, Predictions, Test_label, Test_img)
	plt.subplot(N_rows, 2 * N_cols, 2 * i + 2)
	ArrayPlotValue(i, Predictions, Test_label)

plt.show()