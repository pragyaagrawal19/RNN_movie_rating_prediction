import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key,value) in word_index.items()}
model= load_model('simple_rnn_imdb.h5')
model.summary()

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocessing_text(text):
    ## lowering the text
    word_list = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in word_list]
    padded_review= sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## create prediction function
def predict_review(text):
    padded_review = preprocessing_text(text)
    prediction = model.predict(padded_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
user_input = st.text_area("Enter your movie review here:")
if st.button("Predict Sentiment"):
    sentiment, score = predict_review(user_input)
    st.write(f"Sentiment: {sentiment}, Score: {score}")
else:
    st.write("Please enter a movie review and click the button to predict sentiment.")
