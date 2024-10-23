# langchain_data_extractor.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_data(url):
    # Create a WebBaseLoader instance with the URL directly
    loader = WebBaseLoader(url)
    
    # Extract content from the URL
    documents = loader.load()
    
    # Optional: Split the content into smaller chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    print(f"Loaded {len(texts)} chunks of text from the website.")
    
    return texts
