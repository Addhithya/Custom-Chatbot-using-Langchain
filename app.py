from flask import Flask, request, jsonify
from langchain_data_extractor import extract_data
from pinecone_setup import PineconeStore
from ollama_embeddings import get_embeddings  # Importing the get_embeddings function
import requests

app = Flask(__name__)

# Define the URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Extract data from the URL and store them in chunks
texts = extract_data(url)

# Set up Pinecone vector store with extracted texts
vectorstore = PineconeStore()

# Define metadata for the document and chunks
metadata = {"id": "technical_courses", "source": url}
chunks = texts  # Assuming `texts` are already the chunks

# Generate embeddings for each chunk
embedding_model = get_embeddings()  # Initialize the embedding model

# Ensure we are only passing plain text to the embed_query method
vectors = [embedding_model.embed_query(text) for text in chunks if isinstance(text, str)]

# Save vectors to Pinecone
vectorstore.save_vectors(vectors, metadata, chunks)

# FastAPI endpoint for Ollama
ollama_host = "http://localhost:8000/generate"

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Please provide a message."}), 400
    
    # Prepare the payload for the FastAPI endpoint
    payload = {"prompt": user_message}
    
    # Send a POST request to the FastAPI Ollama endpoint
    response = requests.post(ollama_host, json=payload)

    if response.status_code == 200:
        return jsonify({"response": response.json()["text"]})
    else:
        return jsonify({"error": "Failed to get response from Ollama."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
