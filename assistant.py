
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from models import Models
import logging
from typing import Optional, Dict, Any


class Assistant:
    """
    A Retrieval-Augmented Generation system using LangChain and Ollama 
    """
    def __init__(self, collection_name: str = "documents", persist_dir: str = "./db/chroma_langchain_db"):
        """
        Initialize the RAG system.
        Args:
            collection_name (str): Name of the collection.
            persist_dir (str): Directory to persist the vector store.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize the models
            self.models = Models()
            self.embeddings = self.models.embeddings
            self.llm = self.models.models_ollama

            # Initialize the vector store
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )

            # Set up prompt template
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant. Answer the question based only the data provided."),
                    ("human", "Use the user question {input} to answer the question. Use only the {context} to answer the question.")
                ]
            )

            # Initialize retrieval chain
            self.setup_retrieval_chain(k=5)


        except Exception as e:
            self.logger.error(f"Error initializing the assistant: {e}")
            raise

    def setup_retrieval_chain(self, k: int = 10):
        """
        Setup the retrieval chain with specified number of documents to retrieve.
        Args:
            k (int): Number of documents to retrieve.
        """
        try:
            self.retriever = self.vector_store.as_retriever(
                kwargs={"k": k}
                # kwargs={ 
                # "k": k,                         # Final number of documents to retrieve
                # "score_threshold": 0.8,         # Minimum score threshold for retrieval
                # "fetch_k": 20"                  # Initial fetch size before filtering
                # }
            )
            combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
        except Exception as e:
            self.logger.error(f"Error setting up the retrieval chain: {e}")
            raise

    def query(self, input_text: str)-> Optional[Dict[str, Any]]:
        """
        Process a query through the RAG system.
        Args:
            input_text (str): The user query.
        Returns:
            Dictionary containing the answer and any additional information.
        """
        try: 
            if not input_text.strip():
                raise ValueError("Empty input text provided.")
            
            self.logger.info(f"Processing query: {input_text}")
            result = self.retrieval_chain.invoke({"input": input_text})
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {"error": str(e)}


def main():
    """
    Main function to run the assistant.
    """
    try: 
        assistant = Assistant()
        while True:
            query = input("User (or type 'q', 'quit', or 'exit' to quit): ")
            if query.lower() in ["q", "quit", "exit"]:
                print("Goodbye!")
                break

            result = assistant.query(query)
            if "error" in result:
                print("Error: ", result["error"])   
            else:
                print(f"Assistant: {result['answer']}\n")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    

if __name__ == "__main__":
    main()
   
