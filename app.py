# import streamlit as st
# import speech_recognition as sr
# import pyttsx3
# import requests
# import json
# from gtts import gTTS
# import io
# import pickle
# from textblob import TextBlob
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import os

# # Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Initialize GROQ API
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# # Initialize HuggingFace API
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# # Initialize speech recognition
# recognizer = sr.Recognizer()

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load or create conversation history
# try:
#     with open('conversation_history.pkl', 'rb') as f:
#         conversation_history = pickle.load(f)
# except FileNotFoundError:
#     conversation_history = []

# # User profiles
# user_profiles = {}

# def transcribe_audio(audio_file):
#     try:
#         with sr.AudioFile(audio_file) as source:
#             audio = recognizer.record(source)
#         text = recognizer.recognize_google(audio, language="en-IN")
#         return text
#     except sr.UnknownValueError:
#         return "Sorry, I couldn't understand the audio."
#     except sr.RequestError:
#         return "Sorry, there was an error processing the audio."

# def text_to_speech(text, lang='en', tld='co.in'):
#     tts = gTTS(text=text, lang=lang, tld=tld)
#     fp = io.BytesIO()
#     tts.write_to_fp(fp)
#     fp.seek(0)
#     return fp

# def get_groq_response(messages):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "mixtral-8x7b-32768",
#         "messages": messages,
#         "max_tokens": 2048
#     }
#     try:
#         response = requests.post(GROQ_API_URL, headers=headers, json=data)
#         response.raise_for_status()
#         return response.json()['choices'][0]['message']['content']
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error communicating with GROQ API: {e}")
#         return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."

# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# def extract_intent(text):
#     tokens = word_tokenize(text.lower())
#     stop_words = set(stopwords.words('english'))
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
#     intent_keywords = {
#         'greeting': ['hello', 'hi', 'hey', 'greetings'],
#         'farewell': ['goodbye', 'bye', 'see you', 'farewell'],
#         'question': ['what', 'why', 'how', 'when', 'where', 'who'],
#         'request': ['can', 'could', 'would', 'please'],
#         'opinion': ['think', 'believe', 'feel', 'opinion'],
#     }
    
#     for intent, keywords in intent_keywords.items():
#         if any(keyword in tokens for keyword in keywords):
#             return intent
    
#     return 'general'

# def main():
#     st.set_page_config(page_title="Multimodal Chatbot", layout="wide")

#     st.title("Multimodal Chatbot")

#     # Sidebar for settings
#     st.sidebar.title("Settings")
#     language = st.sidebar.selectbox("Choose language", ["English", "Hindi", "Spanish", "French", "German"])
#     voice_type = st.sidebar.radio("Choose voice type", ["Male", "Female"])

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # User login
#     if "user" not in st.session_state:
#         user_name = st.text_input("Enter your name:")
#         if user_name:
#             st.session_state.user = user_name
#             user_profiles[user_name] = {"language": language, "voice_type": voice_type}
#             st.sidebar.success(f"Logged in as {user_name}")
#     else:
#         user_name = st.session_state.user
#         st.sidebar.success(f"Logged in as {user_name}")

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Audio input
#     audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
#     if audio_file:
#         if st.button("Transcribe Audio"):
#             transcription = transcribe_audio(audio_file)
#             st.session_state.messages.append({"role": "user", "content": transcription})
#             with st.chat_message("user"):
#                 st.markdown(transcription)

#     # Text input
#     if prompt := st.chat_input("What's on your mind?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#     # Generate response
#     if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             user_message = st.session_state.messages[-1]["content"]
#             sentiment = analyze_sentiment(user_message)
#             intent = extract_intent(user_message)
            
#             # Add sentiment and intent to the context
#             context_message = f"The user's message has a sentiment of {sentiment:.2f} and the intent seems to be {intent}. Please take this into account in your response."
#             messages_with_context = st.session_state.messages + [{"role": "system", "content": context_message}]
#             response = get_groq_response(messages_with_context)
            
#             st.markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})

#             # Text-to-speech
#             lang_code = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr', 'German': 'de'}
#             tld = 'co.in' if language == 'Hindi' else 'com'
#             audio_fp = text_to_speech(response, lang=lang_code[language], tld=tld)
#             st.audio(audio_fp, format='audio/mp3')

#     # Save conversation history
#     conversation_history.extend(st.session_state.messages)
#     with open('conversation_history.pkl', 'wb') as f:
#         pickle.dump(conversation_history, f)

#     # Option to clear conversation history
#     if st.button("Clear Conversation History"):
#         st.session_state.messages = []
#         conversation_history.clear()
#         with open('conversation_history.pkl', 'wb') as f:
#             pickle.dump(conversation_history, f)
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()


import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import json
from gtts import gTTS
import io
import pickle
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pyaudio

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load or create conversation history
try:
    with open('conversation_history.pkl', 'rb') as f:
        conversation_history = pickle.load(f)
except FileNotFoundError:
    conversation_history = []

# User profiles
user_profiles = {}

def transcribe_audio(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="en-IN")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error processing the audio."

def text_to_speech(text, lang='en', tld='co.in'):
    tts = gTTS(text=text, lang=lang, tld=tld)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def get_groq_response(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": messages,
        "max_tokens": 1000
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with GROQ API: {e}")
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def extract_intent(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    intent_keywords = {
        'greeting': ['hello', 'hi', 'hey', 'greetings'],
        'farewell': ['goodbye', 'bye', 'see you', 'farewell'],
        'question': ['what', 'why', 'how', 'when', 'where', 'who'],
        'request': ['can', 'could', 'would', 'please'],
        'opinion': ['think', 'believe', 'feel', 'opinion'],
    }
    
    for intent, keywords in intent_keywords.items():
        if any(keyword in tokens for keyword in keywords):
            return intent
    
    return 'general'

def main():
    st.set_page_config(page_title="Multimodal Chatbot", layout="wide")

    st.title("Multimodal Chatbot")

    # Sidebar for settings
    st.sidebar.title("Settings")
    language = st.sidebar.selectbox("Choose language", ["English", "Hindi", "Spanish", "French", "German"])
    voice_type = st.sidebar.radio("Choose voice type", ["Male", "Female"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User login
    if "user" not in st.session_state:
        user_name = st.text_input("Enter your name:")
        if user_name:
            st.session_state.user = user_name
            user_profiles[user_name] = {"language": language, "voice_type": voice_type}
            st.sidebar.success(f"Logged in as {user_name}")
    else:
        user_name = st.session_state.user
        st.sidebar.success(f"Logged in as {user_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Audio input from file
    audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
    if audio_file:
        if st.button("Transcribe Audio"):
            transcription = transcribe_audio(audio_file)
            st.session_state.messages.append({"role": "user", "content": transcription})
            with st.chat_message("user"):
                st.markdown(transcription)

    # Live audio input
    if st.button("Start Live Recording"):
        with st.spinner("Listening..."):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                    transcription = recognizer.recognize_google(audio, language="en-IN")
                    st.session_state.messages.append({"role": "user", "content": transcription})
                    with st.chat_message("user"):
                        st.markdown(transcription)
            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand the audio.")
            except sr.RequestError:
                st.error("Sorry, there was an error processing the audio.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Text input
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            user_message = st.session_state.messages[-1]["content"]
            sentiment = analyze_sentiment(user_message)
            intent = extract_intent(user_message)
            
            # Add sentiment and intent to the context
            context_message = f"The user's message has a sentiment of {sentiment:.2f} and the intent seems to be {intent}. Please take this into account in your response."
            messages_with_context = st.session_state.messages + [{"role": "system", "content": context_message}]
            response = get_groq_response(messages_with_context)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Text-to-speech
            lang_code = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr', 'German': 'de'}
            tld = 'co.in' if language == 'Hindi' else 'com'
            audio_fp = text_to_speech(response, lang=lang_code[language], tld=tld)
            st.audio(audio_fp, format='audio/mp3')

    # Save conversation history
    conversation_history.extend(st.session_state.messages)
    with open('conversation_history.pkl', 'wb') as f:
        pickle.dump(conversation_history, f)

    # Option to clear conversation history
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        conversation_history.clear()
        with open('conversation_history.pkl', 'wb') as f:
            pickle.dump(conversation_history, f)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
