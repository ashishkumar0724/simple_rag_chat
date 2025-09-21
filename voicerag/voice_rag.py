import re
import os
import streamlit as st
from vosk import Model, KaldiRecognizer
import wave
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directory to store uploaded audio files
audios_directory = 'voice-rag/audios/'

# Ensure the directory exists
os.makedirs(audios_directory, exist_ok=True)

# Initialize embeddings and language model with Deepseek-R1:1.5b
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:1.5b")

# Function to upload an audio file
def upload_audio(file):
    if not file.name.lower().endswith(".wav"):
        raise ValueError("Only mono, 16-bit, 16kHz WAV files are supported.")
    file_path = audios_directory + file.name
    try:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
    return file_path

# Function to transcribe audio using Vosk
def transcribe_audio(file_path):
    # Load the Vosk model (download it if not already available)
    model_path = "vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Vosk model not found at {model_path}. Download it from https://alphacephei.com/vosk/models"
        )

    # Load the Vosk model
    vosk_model = Model(model_path)
    recognizer = KaldiRecognizer(vosk_model, 16000)

    # Open the audio file
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        raise ValueError("Audio file must be mono, 16-bit, and 16kHz.")

    # Transcribe the audio
    transcription = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription.append(result.get("text", ""))
    final_result = json.loads(recognizer.FinalResult())
    transcription.append(final_result.get("text", ""))

    return " ".join(transcription).strip()

# Function to split text into smaller chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

# Function to index documents into the vector store
def index_docs(texts):
    vector_store.add_texts(texts)

# Function to retrieve relevant documents based on a query
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to answer a question using the retrieved documents
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return clean_text(chain.invoke({"question": question, "context": context}))

# Function to clean up the generated text
def clean_text(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# Streamlit UI
uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav"],  # Only support WAV files
    accept_multiple_files=False
)

if uploaded_file:
    try:
        file_path = upload_audio(uploaded_file)
        text = transcribe_audio(file_path)
        st.success("Transcription completed!")
        st.text_area("Transcribed Text", value=text, height=200)

        chunked_texts = split_text(text)
        index_docs(chunked_texts)

        question = st.chat_input("Ask a question about the audio content:")

        if question:
            st.chat_message("user").write(question)
            related_docs = retrieve_docs(question)
            answer = answer_question(question, related_docs)
            st.chat_message("assistant").write(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")