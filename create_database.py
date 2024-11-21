# Import necessary modules from langchain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Import modules for OpenAI API and environment variables
import openai 
from dotenv import load_dotenv
# Import modules for file and directory operations
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Define constants for paths
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")  # Debug print
    chunks = split_text(documents)
    print(f"Split into {len(chunks)} chunks.")  # Debug print
    save_to_chroma(chunks)

def load_documents():
    # Load documents from the specified directory
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print content and metadata of a sample chunk
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the existing database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new database from the document chunks
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")  # Debug print

if __name__ == "__main__":
    main()