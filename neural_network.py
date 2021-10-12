__author__ = "Aurélien Bück-Kaeffer"
__version__ = "1.1"
__date__ = "12-10-2021"

#Credit to https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/ for most of this code

import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()
    input = input.replace("£", "A")

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    #filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(tokens)

def preprocess_data(data):
    #print(data)
    processed_inputs = tokenize_words(data)
    #processed_inputs = data
    chars = sorted(list(set(processed_inputs)))
    char_to_num = dict((c, i) for i, c in enumerate(chars))
    num_to_char = dict((i, c) for i, c in enumerate(chars))

    input_len = len(processed_inputs)
    vocab_len = len(chars)
    print ("Total number of characters:", input_len)
    print ("Total vocab:", vocab_len)

    seq_length = 100
    x_data = []
    y_data = []

    # loop through inputs, start at the beginning and go until we hit
    # the final character we can create a sequence out of
    for i in range(0, input_len - seq_length, 1):
        # Define input and output sequences
        # Input is the current character plus desired sequence length
        in_seq = processed_inputs[i:i + seq_length]

        # Out sequence is the initial character plus total sequence length
        out_seq = processed_inputs[i + seq_length]

        # We now convert list of characters to integers based on
        # previously and add the values to our lists
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])

    n_patterns = len(x_data)
    print ("Total Patterns:", n_patterns)

    X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)

    y = np_utils.to_categorical(y_data)

    return X, y, input_len, vocab_len, seq_length, n_patterns, num_to_char, char_to_num

def create_model(x_shape_1, x_shape_2, y_shape_1):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_shape_1, x_shape_2), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y_shape_1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train(model, server_name, X, y):
    filepath = "Models/{}.hdf5".format(server_name)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]
    model.fit(X, y, epochs=10, batch_size=256, callbacks=desired_callbacks)