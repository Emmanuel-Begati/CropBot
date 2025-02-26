import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
from datetime import datetime

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('crops.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('backend/words.pkl', 'rb'))
classes = pickle.load(open('backend/classes.pkl', 'rb'))
model = load_model('backend/chatbotmodel.h5')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    outputs = bert_model(inputs)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return embeddings

def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.20
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if tag in i['tags']:
            return random.choice(i['responses'])
    return "I don't have information about that. Could you please rephrase or ask about a different crop disease?"

def main_(message: str):
    ints = predict_class(message)
    if ints:
        return get_response(ints, intents)
    return "I don't have information about that. Could you please rephrase or ask about a different crop disease?"

# Set page config
st.set_page_config(
    page_title="Crop Bot",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS for white background and modern UI
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Crop Bot")
st.write("Ask me about crops and their diseases!")

# Chat interface
if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.text_input("Ask your question here:", "", key="input", placeholder="e.g., What is a cover crop?")
if st.button("Send"): 
    if user_input:
        response = main_(user_input)
        st.session_state['history'].append((user_input, response))

# Display chat history
for user_msg, bot_msg in st.session_state['history']:
    message(user_msg, is_user=True)
    message(bot_msg, is_user=False)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>\n<p>This is just a personal passion project. So use with caution. #Begati", unsafe_allow_html=True)

