# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:51:40 2020

@author: shubh
"""
#Import necessary packages
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

#reate required list
words=[]
training_sentences = []
labels = []
classes = []
documents = []

#read intents.json file
data_file = open('intents.json').read()
intents = json.loads(data_file)


stop_words = set(stopwords.words('english')) 
import string
table = str.maketrans('', '', string.punctuation)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        # remove stopwords
        w = [word for word in w if not word in stop_words] 
        # remove punctuation marks
        w = [word.translate(table) for word in w]
        # convert all words to lower case
        w = [lemmatizer.lemmatize(word.lower()) for word in w]
        # convert all words to their roots
        w = [porter.stem(word) for word in w]
        words.extend(w)
        #increase training dataset to train deep learning model more accurately
        j=0
        for j in range(30):
            # adding documents
            documents.append((w, intent['tag']))
        
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# create class index dictionary
class_index = {classes[i] :  i for i in range(len(classes))}
class_index_inv = {i :  classes[i] for i in range(len(classes))}

#shuffle dataset to train model accurately
random.shuffle(documents)

#target data
output = [0]*12

# create train and target data
for i in range(len(documents)):
    training_sentences.append(documents[i][0])
    output_row = list(output)
    output_row[class_index.get(documents[i][1])]=1
    labels.append(output_row)


from tensorflow.keras.preprocessing.text import Tokenizer

#create word index for all the unique words in json file
tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
# vectorizing te sentences
sequences = tokenizer.texts_to_sequences(training_sentences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
#padding to have all vectors of same length
padded = pad_sequences(sequences , maxlen = 8 , truncating = 'post')

#defining deep neural network
model = Sequential([
        tensorflow.keras.layers.Embedding(100,64),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(64, return_sequences = True)),
        tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(32)),
        tensorflow.keras.layers.Dense(64,activation='relu'),
        tensorflow.keras.layers.Dense(len(class_index),activation='softmax')
        ])

#SGD optimizer is used
sgd = tensorflow.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(padded), np.array(labels), epochs=20, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)


#pickle.dump(words,open('.pkl','wb'))
#pickle.dump(classes,open('classes.pkl','wb'))


from tensorflow.keras.models import load_model
#load model
model = load_model('chatbot_model.h5')
import json
import random
#read json file
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#model.summary()
# message preprocessing and predicting response
def chatbot_response(msg):
    
    w = nltk.word_tokenize(msg)
    w = [word for word in w if not word in stop_words] 
    w = [word.translate(table) for word in w]
    w = [lemmatizer.lemmatize(word.lower()) for word in w]
    w = [porter.stem(word) for word in w]
    msgs = ""
    for i in range(len(w)):
        msgs = msgs + w[i]
        msgs = msgs + " "
    sequences = tokenizer.texts_to_sequences([msgs])
    msg = pad_sequences(sequences , maxlen = 8 , truncating = 'post')
    ERROR_THRESHOLD = 0.75
    res = model.predict(np.array(msg))[0]
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    tag = class_index_inv.get(results[0][0])
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result



import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# creating GUI for chatbot
base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="lightblue", height="8", width="50", font="Calibri",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=426)
ChatLog.place(x=6,y=6, height=426, width=370)
EntryBox.place(x=128, y=441, height=50, width=265)
SendButton.place(x=6, y=441, height=50)

base.mainloop()