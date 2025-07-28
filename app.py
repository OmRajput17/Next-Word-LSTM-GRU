import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model

model =  load_model('hamlet-LSTMRNN.h5')

# load the tokenizer 

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict the next word

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len):]
    
    token_list = pad_sequences([token_list], maxlen = max_sequence_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Function to predict the sentence of Length N
def generate_sentence(model, tokenizer, text, max_sequence_len, max_words_to_generate):
    generated_text = text.strip()
    current_input_context = text.strip()
    index_to_word = tokenizer.index_word
    sentence_enders = {'.', '!', '?', '\n'}

    for _ in range(max_words_to_generate):
        token_list = tokenizer.texts_to_sequences([current_input_context])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len):]

        token_list_padded = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list_padded, verbose=0)[0]
        predicted_word_index = np.argmax(predicted_probs)
        next_word = index_to_word.get(predicted_word_index)

        if next_word is None:
            break

        if next_word in sentence_enders:
            generated_text += next_word
            break

        if generated_text and not generated_text.endswith((' ', '.', '!', '?')):
            generated_text += " "
        
        generated_text += next_word
        current_input_context = generated_text

    return generated_text

# Streamlit app

st.title("Next Word Prediction")

input_text = st.text_input("Enter the sequence of words", "To be or not to be")

if st.button("Predict Next word and Sentence"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(
        model=model, 
        tokenizer=tokenizer, 
        text=input_text, 
        max_sequence_len=max_sequence_len,
    )
    st.write(f"Next Word {next_word}")

    sentence = generate_sentence(
        model=model,
        tokenizer=tokenizer,
        text = input_text,
        max_sequence_len=max_sequence_len,
        max_words_to_generate=25,
    )
    st.write(f"Generated Sentence : {sentence}")
