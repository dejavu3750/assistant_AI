import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from uuid import uuid4
from models import Models

load_dotenv()

# Initialize the models
models = Models()
embeddings = models.embeddings
llm = models.models_ollama

# Intialize the document loader
data_folder = "./data/pdf"
chunk_size = 1000
chunk_overlap = 50

# Chroma vector store database
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db"
)

# Ingest a file
def ingest_file(file_path):
    # Skip for non-PDF files
    if not file_path.lower().endswith(".pdf"):
        return
    
    print(f"Ingesting file: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", "", " "]
    )

    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len(documents)} documents to the vector store.")
    vector_store.add_documents(documents, ids=uuids)
    print(f"Finished ingesting file: {file_path}")


def main_loop():    
    # Get all files in the data folder
    for file in os.listdir(data_folder):
        if not file.startswith("_"):
            file_path = os.path.join(data_folder, file)
            ingest_file(file_path)
            new_filename = "_" + file
            new_filename_path = os.path.join(data_folder, new_filename)
            os.rename(file_path, new_filename_path)
    
if __name__ == "__main__":
    main_loop()
