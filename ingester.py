import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from uuid import uuid4
from models import Models

# Initialize the models
models = Models()
embeddings = models.embeddings
llm = models.models_ollama

# Intialize the document loader
data_folder = "./data"
pdf_folder = os.path.join(data_folder, "pdf")
markdown_folder = os.path.join(data_folder, "markdown")
db_folder = "./db/chroma_langchain_db"
chunk_size = 1000
chunk_overlap = 50

# Chroma vector store database
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=db_folder
)

def get_document_loader(file_path: str):
    """
    Return appropriate loader based on file extension.
    """
    if file_path.lower().endswith('.pdf'):
        return PyPDFLoader(file_path)
    elif file_path.lower().endswith(('.md', '.markdown')):
        return UnstructuredMarkdownLoader(file_path)
    return None

# Ingest a file
def ingest_file(file_path):
    # Get appropriate loader
    loader = get_document_loader(file_path)
    if loader is None:
        print(f"Unsupported file type: {file_path}")
        return   
    
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

def process_folder(data_folder: str):
    """Process all files in a given folder"""
    if not os.path.exists(data_folder):
        print(f"Creating folder: {data_folder}")
        os.makedirs(data_folder)
        return
    
    for file in os.listdir(data_folder):        
        file_path = os.path.join(data_folder, file)
        ingest_file(file_path)        

def main_loop():        
    # Process bolt PDF and Markdown folders
    process_folder(pdf_folder)
    process_folder(markdown_folder)
    
if __name__ == "__main__":
    main_loop()
