import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import uuid  # For generating unique session IDs

# Set page configuration FIRST
st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")

# Define the prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directory to store uploaded PDFs
pdfs_directory = 'chat-with-pdf/pdfs/'

# Ensure the directory exists
os.makedirs(pdfs_directory, exist_ok=True)

# Initialize embeddings and language model with the correct model name
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:1.5b")


# Function to upload a PDF file
def upload_pdf(file):
    file_path = pdfs_directory + file.name
    try:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


# Function to load a PDF file
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# Function to split text into smaller chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


# Function to index documents into the vector store
def index_docs(documents):
    vector_store.add_documents(documents)


# Function to retrieve relevant documents based on a query
def retrieve_docs(query):
    return vector_store.similarity_search(query)


# Function to answer a question using the retrieved documents
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# Initialize session state variables
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # Stores chat history for each session
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None


# Generate a unique session ID
def generate_session_id():
    return str(uuid.uuid4())[:8]  # Shortened UUID for readability


# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")

    # Option to create a new session
    if st.button("Create New Session"):
        session_id = generate_session_id()
        st.session_state.current_session_id = session_id
        st.session_state.sessions[session_id] = {"chat_history": []}
        st.rerun()  # Refresh the app to update the session

    # Display existing sessions
    st.subheader("Sessions")
    for session_id in st.session_state.sessions.keys():
        if st.button(f"Session: {session_id}", key=f"session_{session_id}"):
            st.session_state.current_session_id = session_id
            st.rerun()  # Refresh the app to load the selected session

    # Toggle Chat History Visibility
    if st.button("Toggle Chat History"):
        st.session_state.show_chat_history = not st.session_state.get("show_chat_history", False)

# Add a title and description
st.title("📚 Chat with PDF")
st.markdown("""
Welcome to **Chat with PDF**! Upload a PDF file, and ask questions about its content. The app uses advanced AI models to provide answers based on the document's context.
""")

# Display Current Session ID
if st.session_state.current_session_id:
    st.subheader(f"Current Session ID: {st.session_state.current_session_id}")
else:
    st.info("No active session. Create a new session from the sidebar.")

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    # Display a spinner while processing the file
    with st.spinner("Uploading and processing the PDF..."):
        upload_pdf(uploaded_file)
        file_path = pdfs_directory + uploaded_file.name

        # Check if the file exists
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
        else:
            documents = load_pdf(file_path)
            chunked_documents = split_text(documents)
            index_docs(chunked_documents)

    st.success("PDF processed successfully! You can now ask questions.")

    # Chat input and response handling
    question = st.chat_input("Ask a question about the PDF:")

    if question:
        # Save the question to chat history
        current_session = st.session_state.sessions[st.session_state.current_session_id]
        current_session["chat_history"].append({"role": "user", "message": question})
        st.chat_message("user").write(question)

        # Show a status indicator while the model is thinking
        with st.status("Retrieving relevant information and generating the answer...", expanded=True) as status:
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            status.update(label="Answer generated!", state="complete", expanded=False)

        # Save the answer to chat history
        current_session["chat_history"].append({"role": "assistant", "message": answer})
        st.chat_message("assistant").write(answer)

# Display Chat History (only if toggled in the sidebar)
if st.session_state.get("show_chat_history", False):
    st.subheader("Chat History")
    if st.session_state.current_session_id:
        current_session = st.session_state.sessions[st.session_state.current_session_id]
        chat_history = current_session["chat_history"]
        if chat_history:
            for entry in chat_history:
                role = entry["role"].capitalize()
                message = entry["message"]
                st.markdown(f"**{role}:** {message}")
        else:
            st.info("No chat history available for this session.")
    else:
        st.info("No active session. Create a new session from the sidebar.")
else:
    st.info("Please upload a PDF file to get started.")