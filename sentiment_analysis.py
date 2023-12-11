"""
date: 10th dec 2023
reference: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
sentiment analysis on IMDB dataset (50k), to classify review as positive or negative
finetuning a BERT model using Tensorflow

1. Load the IMDB dataset
2. Load a BERT model from TensorFlow Hub
3. Build your own model by combining BERT with a classifier
4. Train your own model, fine-tuning BERT as part of that
5. Save your model and use it to classify sentences

pip install -U "tensorflow-text"
pip install "tf-models-official"

"""

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt


# Download the IMDB dataset

# url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
# dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')
# dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
dataset_dir = "dataset/aclImdb"
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# remove unused folders to make it easier to load the data
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)

# create a validation set

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='training',
	seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
	train_dir,
	batch_size=batch_size,
	validation_split=0.2,
	subset='validation',
	seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
	test_dir,
	batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# EDA
for text_batch, label_batch in train_ds.take(1):
	for i in range(3):
		print(f'Review: {text_batch.numpy()[i]}')
		label = label_batch.numpy()[i]
		print(f'Label : {label} ({class_names[label]})')


bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


# Choose a BERT model to fine-tune from TensorFlow Hub
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

#try the preprocessing model on some text 
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')


# Using the BERT model
bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


# Define your model
def build_classifier_model():
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
	preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
	encoder_inputs = preprocessing_layer(text_input)
	encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
	outputs = encoder(encoder_inputs)
	net = outputs['pooled_output']
	net = tf.keras.layers.Dropout(0.1)(net)
	net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
	return tf.keras.Model(text_input, net)

# Let's check that the model runs with the output of the preprocessing model.
classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

# Let's take a look at the model's structure.
tf.keras.utils.plot_model(classifier_model)


#Model training

#Loss function
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Optimizer
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

#metric
metrics = tf.metrics.BinaryAccuracy()

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# compile the BERT model 
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Training
print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

print("train results: ", history.history)

# Evaluate the model
loss, accuracy = classifier_model.evaluate(test_ds)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


#Plot the accuracy and loss over time

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')


#save model
dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)


#inference

#load model
reloaded_model = tf.saved_model.load(saved_model_path)

def print_my_examples(inputs, results):
	result_for_printing = \
	[f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
	                     for i in range(len(inputs))]
	print(*result_for_printing, sep='\n')
	print()


examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]


reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)

