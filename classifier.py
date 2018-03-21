import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.shape

test.shape

train.head()

X_train = train["comment_text"].fillna("Nan").values
X_test = test["comment_text"].fillna("Nan").values

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

y_train = train[labels].values

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
tokenizer.fit_on_texts(list(X_train))
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)
X_train_p = pad_sequences(X_train_tokenized,maxlen = 50,padding = "post",truncating = "post")
X_test_p = pad_sequences(X_test_tokenized,maxlen = 50,padding = "post",truncating = "post")

embeddings_index = dict()
data = open('glove.6B.300d.txt')
for line in data:
    values = line.split()
    word = values[0]
    coeff = np.asarray(values[1:])
    embeddings_index[word] = coeff
data.close()

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,300))
for word,i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers import CuDNNLSTM,Dense,MaxPooling1D,Conv1D,Dropout,BatchNormalization,GlobalMaxPooling1D,Bidirectional
from keras.layers import Embedding
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 300, weights=[embedding_matrix],input_length=50, trainable=True))
model.add(Bidirectional(CuDNNLSTM(300, return_sequences=True)))
model.add(Conv1D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='sigmoid'))

model.compile(optimizer=Adam(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])

from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X_train_p,y_train,test_size=0.20)

import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'

model.fit(X_tr,y_tr,batch_size=64,epochs=100,validation_data=(X_te,y_te))
