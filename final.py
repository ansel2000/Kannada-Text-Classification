import streamlit as st
import pickle
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf


user_input = st.text_area("Kannada Text", "Enter the kannada text")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('my_model.h5')

seq = tokenizer.texts_to_sequences([user_input])
padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=250)
pred = model.predict(padded)
labels = ['entertainment', 'lifestyle', 'sports']
print(pred)
# print(pred, labels[np.argmax(pred)])

st.write('The text entered is:\n', user_input)

if pred is not None:
    st.title('The predicted class is:\n')
    st.title(labels[np.argmax(pred)])