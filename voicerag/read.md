# Chat with PDF
A simple Voice RAG (Retrieval-Augmented Generation) system using Deepseek, LangChain, and Streamlit to chat with audio files and answer complex questions about them.

You can watch the video on how it was built on my [YouTube](https://youtu.be/HT4a6A_wXdA).

# Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

```bash
ollama pull deepseek-r1:1.5b
```

Install the dependencies using pip:

```bash
pip install -r requirement.txt
```

# Run
Run the Streamlit app:

```bash
streamlit run voicechat.py
```