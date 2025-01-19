from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        # ollama pull nomic-embed-text
        self.embeddings = OllamaEmbeddings(
            model = "nomic-embed-text"
        )

        # ollama pull llama3.1
        self.models_ollama = ChatOllama(
            model="llama3.1",
            temperature=0,
        )
            

