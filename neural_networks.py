import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('sarcasm1.json', 'r') as f:
    data = [json.loads(line) for line in f]
TRAIN_SIZE = 20000
VOCAB_SIZE = 30000
EPOCHS = 20

sentences = []
labels = []
urls = []
for d in data:
    urls.append(d['article_link'])
    labels.append(d['is_sarcastic'])
    sentences.append(d['headline'])
# sentences = ['a chair is a type of seat.',
#              'The Chair primary features are two pieces of a durable material, attached as back and seat to one '
#              'another at a 90° or slightly greater angle!',
#              'chairs may be made of wood, metal, or synthetic materials.']
#
# test_sentences = ['My chair is not moving',
#                   'My chair is made out of wood and some metal']
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
training_sen = sentences[0:TRAIN_SIZE]
testing_sen = sentences[TRAIN_SIZE:]
training_lab = labels[0:TRAIN_SIZE]
testing_lab = labels[TRAIN_SIZE:]

training_lab = np.asarray(training_lab)
testing_lab = np.asarray(testing_lab)

tokenizer.fit_on_texts(training_sen)
train_seq = tokenizer.texts_to_sequences(training_sen)
train_mat = pad_sequences(train_seq, padding='post')

tokenizer.fit_on_texts(testing_sen)
test_seq = tokenizer.texts_to_sequences(testing_sen)
test_mat = pad_sequences(test_seq, padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 3, input_length=VOCAB_SIZE),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_mat, training_lab, epochs=EPOCHS, validation_data=(test_mat, testing_lab),
                    verbose=2)

sen = ['granny is the most bird in the wild',
       'the weather is bright and sunny']
seq = tokenizer.texts_to_sequences(sen)
check_mat = pad_sequences(seq, padding='post')
print(model.predict(check_mat))
# tokenizer.fit_on_texts(sentences)
# word_ind = tokenizer.word_index
# seq = tokenizer.texts_to_sequences(sentences)
# test = tokenizer.texts_to_sequences(test_sentences)
# matrix1 = tf.ragged.constant(seq)
# matrix2 = tf.ragged.constant(test)
# matrix = tf.concat([matrix1, matrix2], axis=0)


