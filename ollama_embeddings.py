from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    return OllamaEmbeddings(model="llama3.1")
