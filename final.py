import streamlit as st
import pickle
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import regex as re


user_input = st.text_area("Enter the Kannada Text")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('my_model.h5')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

file1 = open('stopwords-kn.txt', 'r')
Lines = file1.readlines()

STOPWORDS = []
count = 0
# Strips the newline character
for line in Lines:
    STOPWORDS.append(line.strip())

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

user_input = clean_text(user_input)

seq = tokenizer.texts_to_sequences([user_input])
padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=250)
pred = model.predict(padded)
labels = ['entertainment', 'lifestyle', 'sports']
print(pred)

# print(pred, labels[np.argmax(pred)])

st.write('The text entered is:\n', user_input)

if pred is not None:
    st.write("Pecentage of the predicted class is:\n")
    st.write("entertainment",pred[0][0]*100)
    st.write("lifestyle",pred[0][1]*100)
    st.write("sports",pred[0][2]*100)
    st.title('The predicted class is:\n')
    st.title(labels[np.argmax(pred)])
    

