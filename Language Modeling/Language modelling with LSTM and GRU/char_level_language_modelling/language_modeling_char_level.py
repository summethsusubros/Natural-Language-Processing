#Importing required libraries.
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.tokenize import regexp_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Embedding

"""STEP-1:Importing the dataset and preprocessing"""

#Loading the dataset.
sample = open("/content/text_data.txt", "r") 
s = sample.read() 
text = s.replace("\n", " ")

#Preprocessing the text data.
def text_preprocessing(text):
  #Converting the text to lower case and removing the begining and end spaces.
  preprocessed_text = text.strip().lower()
  #Tokenizing the text data.
  preprocessed_text = regexp_tokenize(preprocessed_text,'[a-zA-Z]+')
  #Detokenizing the tokens.
  preprocessed_text = TreebankWordDetokenizer().detokenize(preprocessed_text)
  return preprocessed_text

text=text_preprocessing(text)
text

len(text)

"""STEP-2:Formation of character sequence."""

#Character sequence formation.
sequence=[]
#Sequence length is set to 50.
length=50
for i in range(0,len(text)+1-length):
  seq=text[i:i+length]
  sequence.append(seq)
sequence

len(sequence)

"""STEP-3:Building encoding dictionary and encoding the character sequence."""

#Building encoder dictionary.
chars = sorted(list(set(text)))
encoder_dictionary = dict((c, i) for i, c in enumerate(chars))
chars

encoder_dictionary

vocab=len(encoder_dictionary)

#Encoding the character sequence.
encoded_sequence = []
for i in sequence:
  seq=[encoder_dictionary[j] for j in i]
  encoded_sequence.append(seq)  
encoded_sequence = np.array(encoded_sequence)

encoded_sequence

"""STEP-4:Train test split"""

#Last element of the encoded sequence is considered as the target value(y) of the preceding sequence(x).
x = encoded_sequence[:,:-1]
y = encoded_sequence[:,-1]

#One-hot encoder.
y = to_categorical(y, num_classes=vocab)

#Train test split.
X_tr, X_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=63)

"""Step-5:Building the SimpleRNN, LSTM, GRU models."""

#Define SimpleRNN model.
SimpleRNN_model = Sequential()
SimpleRNN_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
SimpleRNN_model.add(SimpleRNN(150, recurrent_dropout=0.1, dropout=0.1))
SimpleRNN_model.add(Dense(vocab, activation='softmax'))
print(SimpleRNN_model.summary())

#Compile the SimpleRNN model.
SimpleRNN_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
SimpleRNN_model.fit(X_tr, y_tr, epochs=3, verbose=2, validation_data=(X_val, y_val))

#Define LSTM model.
LSTM_model = Sequential()
LSTM_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
LSTM_model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
LSTM_model.add(Dense(vocab, activation='softmax'))
print(LSTM_model.summary())

#Compile the LSTM model.
LSTM_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
LSTM_model.fit(X_tr, y_tr, epochs=10, verbose=2, validation_data=(X_val, y_val))

#Define GRU model.
GRU_model = Sequential()
GRU_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
GRU_model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
GRU_model.add(Dense(vocab, activation='softmax'))
print(GRU_model.summary())

#Compile the GRU model.
GRU_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
GRU_model.fit(X_tr, y_tr, epochs=3, verbose=2, validation_data=(X_val, y_val))

"""STEP-6:Text sequence generation and tesing the models."""

#Generate a sequence of characters with a language model.
def generate_seq(model, encoder_dictionary, seqence_length, input_text, no_of_chars_to_be_generated):
	in_text = input_text
	#generate a fixed number of characters.
	for i in range(no_of_chars_to_be_generated):
		#Encode the characters as integers.
		encoded = [encoder_dictionary[char] for char in in_text]
		#Truncate sequences to a fixed length.
		encoded = pad_sequences([encoded], maxlen=seqence_length, truncating='pre')
		#Predict character.
		ypred = model.predict_classes(encoded, verbose=0)
		#Reverse map integer to character.
		out_char = ''
		for char, index in encoder_dictionary.items():
			if index == ypred:
				out_char = char
				break
		#Append to input.
		in_text += char
	return in_text


#SimpleRNN model testing.
inp = 'the'
print(generate_seq(SimpleRNN_model, encoder_dictionary,length-1,inp.lower(),20))

#LSTM model testing.
inp = 'the '
print(generate_seq(LSTM_model, encoder_dictionary,length-1,inp.lower(),20))

#GRU model testing.
inp = 'they'
print(generate_seq(GRU_model, encoder_dictionary,length-1,inp.lower(),20))
