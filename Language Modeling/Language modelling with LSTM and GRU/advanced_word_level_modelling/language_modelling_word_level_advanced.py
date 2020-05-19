#Importing required libraries.
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
nltk.download('punkt')
from nltk.tokenize import regexp_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize 

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Embedding

#Dictionary for contraction mapping.
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

"""STEP-1:Importing the dataset and preprocessing"""

#Loading the dataset.
sample = open("/content/example.txt", "r") 
s = sample.read() 
  
text = s.replace("\n", " ")

text

#Removing the contraction words.
text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])    
text = re.sub(r"'s\b","",text)

#Tokenizing the sentence into sentences.
sentences=sent_tokenize(text)

#Forming a dataframe with sentences.
data={'sentences':sentences}
sentences_df=pd.DataFrame(data)
sentences_df

#Preprocessing the text data.
def preprocessor (sentence):
  #converting the text to lower case.
  sentence=sentence.map(lambda s : s.lower())
  #Tokenizing the text data.
  sentence=sentence.map(lambda s : regexp_tokenize(s,'[a-zA-Z]+'))
  #Detokenizing the tokens to fprm the text data.
  sentence=sentence.map(lambda s : TreebankWordDetokenizer().detokenize(s))
  return sentence

preprocessed_sentences = preprocessor(sentences_df['sentences'])
preprocessed_sentences

"""STEP-2:Formation of word sequence."""

#Removing sentences with words less than the sequence length.
#The sequence length is set to 5.
length=5
preprocessed_long_sentences=[]
for i in preprocessed_sentences:
  if(len(i.split())>length-1):
    preprocessed_long_sentences.append(i)
preprocessed_long_sentences

#Word sequence formation.
sequence=[]
for sent in preprocessed_long_sentences:
  for i in range(0,len(sent.split())+1-length):
    seq=sent.split()[i:i+length]
    sequence.append(seq)

sequence

"""STEP-3:Building encoding dictionary and encoding the word sequence."""

#Forming the a text only with the long sentences.
text=' '.join(i for i in preprocessed_long_sentences)
text

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
SimpleRNN_model.fit(X_tr, y_tr, epochs=50, verbose=2, validation_data=(X_val, y_val))

#Define LSTM model.
LSTM_model = Sequential()
LSTM_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
LSTM_model.add(LSTM(150, recurrent_dropout=0.1, dropout=0.1))
LSTM_model.add(Dense(vocab, activation='softmax'))
print(LSTM_model.summary())

#Compile the LSTM model.
LSTM_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
LSTM_model.fit(X_tr, y_tr, epochs=50, verbose=2, validation_data=(X_val, y_val))

#Define GRU model.
GRU_model = Sequential()
GRU_model.add(Embedding(vocab, 50, input_length=length-1, trainable=True))
GRU_model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
GRU_model.add(Dense(vocab, activation='softmax'))
print(GRU_model.summary())

# compile the GRU model
GRU_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
GRU_model.fit(X_tr, y_tr, epochs=50, verbose=2, validation_data=(X_val, y_val))

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
inp = 'since'
print(generate_seq(SimpleRNN_model, encoder_dictionary,length-1,inp,14))

#LSTM model testing.
inp = 'what'
print(generate_seq(LSTM_model, encoder_dictionary,length-1,inp,8))

#GRU model testing.
inp = 'my love'
print(generate_seq(GRU_model, encoder_dictionary,length-1,inp,8))