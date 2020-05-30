from attention import AttentionLayer

# Commented out IPython magic to ensure Python compatibility.
# Importing libraries
import string
import re
import numpy as np
from numpy import array, argmax 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Dense, LSTM, Embedding,TimeDistributed ,Input 
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
# %matplotlib inline
pd.set_option('display.max_colwidth', 200)

dataset=pd.read_table('/content/Machine-Translation/deu.txt',names=['english_text','german_text','link'])
dataset

len(dataset)

dataset.columns

dataset = dataset[['english_text', 'german_text']]

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

# Preprocessing english text
cleaned_english_text = dataset.english_text.map(lambda x : x.lower())
cleaned_english_text = cleaned_english_text.map(lambda x : ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in x.split(" ")]))
cleaned_english_text = cleaned_english_text.map(lambda x : re.sub(r"'s\b",'',x))
cleaned_english_text = cleaned_english_text.map(lambda x : re.sub(r'[^a-zA-Z]', ' ', x))
cleaned_english_text = cleaned_english_text.map(lambda x : ' '.join([w for w in x.split()])) #for removing multiple spaces together

print(cleaned_english_text)

# Preprocessing english text
table = str.maketrans(string.punctuation, ' '*len(string.punctuation), string.digits)
cleaned_german_text = dataset.german_text.map(lambda x : x.lower())
cleaned_german_text = cleaned_german_text.map(lambda x : x.translate(table))
cleaned_german_text = cleaned_german_text.map(lambda x : ' '.join([w for w in x.split()])) #for removing multiple spaces together
cleaned_german_text = cleaned_german_text.map(lambda x : 'starttoken ' + x + ' endtoken')

print(cleaned_german_text)

dataset['cleaned_english_text'] = cleaned_english_text
dataset['cleaned_german_text'] = cleaned_german_text

# Displaying first five lines from preprocessed dataset
for i in range(len(dataset)-10,len(dataset)):
  print('English :' ,cleaned_english_text[i] )
  print('German :' ,cleaned_german_text[i])
  print('\n')

# Finding the length of text
english_word_count = dataset['cleaned_english_text'].map(lambda x : len(x.split()))
dataset['english_word_count'] = english_word_count
english_word_count.max()

# Finding the length of heaadlines
german_word_count = dataset['cleaned_german_text'].map(lambda x : len(x.split()))
dataset['german_word_count'] = german_word_count
german_word_count.max()

english_length=101 #max length of english sentence
german_length=78   #max length of german sentence

# train test split
x_train, x_test, y_train, y_test = train_test_split(dataset['cleaned_english_text'], dataset['cleaned_german_text'], test_size=0.2,random_state=0)

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

x_train    =   x_tokenizer.texts_to_sequences(x_train) 
x_test   =   x_tokenizer.texts_to_sequences(x_test)

x_train    =   pad_sequences(x_train,  maxlen=english_length) 
x_test   =   pad_sequences(x_test, maxlen=english_length)

english_vocab_size   =  len(x_tokenizer.word_index) +1
english_vocab_size

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

y_train    =   y_tokenizer.texts_to_sequences(y_train) 
y_test   =   y_tokenizer.texts_to_sequences(y_test) 

y_train    =   pad_sequences(y_train, maxlen=german_length)
y_test   =   pad_sequences(y_test, maxlen=german_length)

german_vocab_size  =   len(y_tokenizer.word_index) +1
german_vocab_size

latent_dim = 101
embedding_dimension = 256
epochs =10
batch_size = 256

from keras import backend as K 
K.clear_session() 

encoder_inputs = Input(shape=(english_length,), name='encoder_inputs_layer')

encoder_embedding_layer= Embedding(english_vocab_size, embedding_dimension, mask_zero= True ,name='encoder_embedding_layer')
encoder_embedding = encoder_embedding_layer(encoder_inputs)

encoder_lstm_layer = LSTM(latent_dim , return_sequences = True ,return_state = True , name = 'encoder_lstm_layer')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm_layer(encoder_embedding)

decoder_inputs = Input(shape=(None,), name = 'decoder_inputs_layer')

decoder_embedding_layer= Embedding(german_vocab_size, embedding_dimension,mask_zero= True , name = 'decoder_embedding_layer')
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_lstm_layer = LSTM(latent_dim, return_sequences=True, return_state = True , name = 'decoder_lstm_layer' )
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm_layer(decoder_embedding ,initial_state = [encoder_state_h, encoder_state_c])

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

decoder_dense_layer = TimeDistributed(Dense(german_vocab_size, activation='softmax' , name = 'decoder_dense_layer'))
decoder_dense_layer_outputs = decoder_dense_layer(decoder_concat_input)

model = Model([encoder_inputs, decoder_inputs], decoder_dense_layer_outputs) 
model.summary(250)

# Model compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=epochs,callbacks=[es],batch_size=batch_size, validation_data=([x_test,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))

# epochs loss function
from matplotlib import pyplot 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() 
pyplot.show()

reverse_target_word_index = y_tokenizer.index_word 
reverse_source_word_index = x_tokenizer.index_word 
target_word_index = y_tokenizer.word_index

len(reverse_target_word_index)

# encoder inference
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, encoder_state_h, encoder_state_c])
encoder_model.summary(250)

# decoder inference
decoder_state_input_h = Input(shape=(latent_dim,), name= 'decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name= 'decoder_state_input_c')
decoder_hidden_state_input = Input(shape=(english_length,latent_dim), name= 'decoder_hidden_state_input')

decoder_embedding_2= decoder_embedding_layer(decoder_inputs)

decoder_outputs2, decoder_state_h2, decoder_state_c2 = decoder_lstm_layer(decoder_embedding_2, initial_state=[decoder_state_input_h, decoder_state_input_c])

attention_layer_inf = Attention(name='attention_layer_inf')
attention_out_inf, attention_states_inf = attention_layer_inf([decoder_hidden_state_input, decoder_outputs2])

decoder_inf_concat = concatenate(axis=-1, name='concat')([decoder_outputs2, attention_out_inf])

decoder_outputs2 = decoder_dense_layer(decoder_inf_concat)

decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [decoder_state_h2, decoder_state_c2])
decoder_model.summary(250)

def sequence_to_sentance(input_sequence):

    e_out, e_h, e_c = encoder_model.predict(input_sequence)

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_word_index['starttoken']

    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        predicted_token_index = np.argmax(output_tokens[:])
        predicted_token = reverse_target_word_index[predicted_token_index]

        if(predicted_token!='endtoken'):
            decoded_sentence += ' '+predicted_token

        if (predicted_token == 'endtoken' or len(decoded_sentence.split()) >= (german_length-1)):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = predicted_token_index

        e_h, e_c = h, c

    return decoded_sentence

def sequence_to_german(input_sequence):
    text=''
    for i in input_sequence:
      if((i!=0 and i!=target_word_index['starttoken']) and i!=target_word_index['endtoken']):
        text=text+reverse_target_word_index[i]+' '
    return text

def sequence_to_english(input_sequence):
    text=''
    for i in input_sequence:
      if(i!=0):
        text=text+reverse_source_word_index[i]+' '
    return text

for i in range(5,10):
  print("English:",sequence_to_english(x_test[i]))
  print("Original German:",sequence_to_german(y_test[i]))
  print("Predicted German:",sequence_to_sentance(x_test[i].reshape(1,english_length)))
  print("\n")

