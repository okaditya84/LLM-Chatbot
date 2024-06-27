# Multimodal Chatbot

## Overview
This repository contains the code for a multimodal chatbot capable of interacting with users through both voice and text inputs and providing responses in both formats. The chatbot utilizes speech-to-text (STT), text-to-speech (TTS), and natural language processing (NLP) technologies to engage in natural-sounding conversations and provide accurate and relevant responses.

## Features
- **Multimodal Input Capability:** Accepts user input in both voice and text formats.
- **Voice and Text Output:** Responds to user queries in both voice and text formats.
- **Conversational Flow:** Maintains context and understanding to engage in natural-sounding conversations.
- **Sentiment Analysis and Intent Extraction:** Analyzes the sentiment and extracts the intent of user messages to provide relevant responses.
- **User-Friendly Interface:** Simple and intuitive interface created with Streamlit.

## Technologies Used
- **Streamlit:** User interface creation.
- **SpeechRecognition:** Speech-to-text conversion.
- **pyttsx3 and gTTS:** Text-to-speech conversion.
- **Requests:** Interaction with the GROQ API for generating responses.
- **Pickle:** Saving and loading conversation history.
- **TextBlob:** Sentiment analysis.
- **NLTK:** Natural language processing.
- **Pyaudio:** Capturing live audio input.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multimodal-chatbot.git
    cd multimodal-chatbot
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows: `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables
Set up the following environment variables:
- `GROQ_API_KEY`: Your GROQ API key.
- `HUGGINGFACE_API_KEY`: Your HuggingFace API key (if used).

### Running the Chatbot
1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Usage
- **Text Input:** Type your message into the input box and press Enter.
- **Voice Input:** Click the "Start Listening" button to provide a live speech input or upload an audio file.
- **Clear History:** Click the "Clear Conversation History" button to clear the chat history.

## Project Structure
- multimodal-chatbot/
- ├── app.py # Main - application code
- ├── requirements.txt # List of required Python packages
- ├── conversation_history.pkl # File to save and load conversation history
- ├── README.md # Project documentation


## Challenges and Solutions
1. **Microphone Access Errors:** 
   - **Challenge:** Initial attempts to capture live audio input resulted in errors.
   - **Solution:** Implemented error handling for microphone initialization and proper resource management.

2. **Handling Multiple Languages:** 
   - **Challenge:** Implementing multilingual support required careful handling of text-to-speech conversion and user preferences.
   - **Solution:** Used `gTTS` for TTS and allowed users to select their preferred language and voice type.

3. **Maintaining Context:** 
   - **Challenge:** Keeping track of conversation history to maintain context.
   - **Solution:** Used `pickle` to save and load conversation history.

4. **API Communication Errors:** 
   - **Challenge:** Ensuring reliable communication with the GROQ API.
   - **Solution:** Implemented error handling and retry mechanisms.

## Future Work
- Improve intent detection and sentiment analysis.
- Integrate additional APIs for more diverse and accurate responses.
- Enhance the user interface with visual elements.
- Implement more advanced NLP techniques for better understanding and response generation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## Contact
For any questions or suggestions, please open an issue or contact me at jethaniaditya7@gmail.com.

---

Thank you for using the Multimodal Chatbot!
