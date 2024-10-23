import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

class PineconeStore:
    def __init__(self, environment="us-east-1"):
        load_dotenv()  # Load environment variables from .env file
        
        # Load API key from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Create a Pinecone instance
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Define the index name
        self.index_name = "web-vector-store"

        # Check if the index exists, if not create it
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=environment)
            )

    def save_vectors(self, vectors, metadata, chunks):
        # Get the index
        index = self.pc.Index(self.index_name)

        # Iterate over the embeddings and save each one with unique metadata
        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": chunks[i]  # Add the text of the chunk here
            }
            # Upsert each vector with its corresponding metadata
            index.upsert(vectors=[(vector_id, vector, chunk_metadata)])

if __name__ == '__main__':
    vector_store = PineconeStore()
    # Example call to save vectors
    # vector_store.save_vectors(embedding, {"id": "doc_1", "source": "example.url"}, chunks)
