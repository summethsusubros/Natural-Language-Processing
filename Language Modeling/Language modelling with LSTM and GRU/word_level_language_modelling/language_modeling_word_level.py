#Importing required libraries.
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
nltk.download('punkt')
from nltk.tokenize import regexp_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Embedding

"""STEP-1:Importing the dataset and preprocessing"""

#Loading the dataset.
sample = open("/content/219-0.txt", "r") 
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

len(text.split())

"""STEP-2:Formation of word sequence."""

#Word sequence formation.
sequence=[]
#Sequence length is set to 5.
length=5
for i in range(0,len(text.split())+1-length):
  seq=text.split()[i:i+length]
  sequence.append(seq)
sequence

"""STEP-3:Building encoding dictionary and encoding the word sequence."""

#Building encoder dictionary.
word = sorted(list(set(text.split())))
encoder_dictionary = dict((c, i) for i, c in enumerate(word))
word

encoder_dictionary

vocab=len(encoder_dictionary)

#Encoding the word sequence.
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
SimpleRNN_model.fit(X_tr, y_tr, epochs=20, verbose=2, validation_data=(X_val, y_val))

#Define LSTM model.
LSTM_model = Sequential()
LSTM_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
LSTM_model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
LSTM_model.add(Dense(vocab, activation='softmax'))
print(LSTM_model.summary())

#Compile the LSTM model.
LSTM_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
LSTM_model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))

#Define GRU model.
GRU_model = Sequential()
GRU_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
GRU_model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
GRU_model.add(Dense(vocab, activation='softmax'))
print(GRU_model.summary())

# compile the GRU model
GRU_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
GRU_model.fit(X_tr, y_tr, epochs=20, verbose=2, validation_data=(X_val, y_val))

"""STEP-6:Text sequence generation and testing the models."""

#Generate a sequence of words with a language model.
def generate_seq(model, encoder_dictionary, seqence_length, input_text, no_of_words_to_be_generated):
	in_text = input_text.lower().split()
	#Generate a fixed number of words.
	for i in range(no_of_words_to_be_generated):
		#Encode the words as integers.
		encoded = [encoder_dictionary[word] for word in in_text]
		#Truncate sequences to a fixed length.
		encoded = pad_sequences([encoded], maxlen=seqence_length, truncating='pre')
		#Predict words.
		ypred = model.predict_classes(encoded, verbose=0)
		#Reverse map integer to word
		out_word = []
		for word, index in encoder_dictionary.items():
			if index == ypred:
				out_word = word
				break
		#Append to input
		in_text.append(out_word)
	return ' '.join(tokens for tokens in in_text)

#SimpleRNN model testing.
inp = 'my love'
print(generate_seq(SimpleRNN_model, encoder_dictionary,length-1,inp,14))

#LSTM model testing.
inp = 'what'
print(generate_seq(LSTM_model, encoder_dictionary,length-1,inp,8))

#GRU model testing.
inp = 'my love'
print(generate_seq(GRU_model, encoder_dictionary,length-1,inp,8))