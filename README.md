# assistant_AI
Assistant AI by Ollama and langchain. Currently support for PDF and Markdown files only.

# Set up
After download the assistant_AI. Please install ollama LLM model "llama3.1" and embedding model "nomic-embed-text" first.

Install all packages in requirements.txt

There are two folders in "data". They are "markdown" and "pdf" folders. Please copy your PDF files or Markdown files into these folders.

# Run
Step1: Run ingester.py to create vector store database in the db folder.
Step2: Run assistant.py. You can ask any questions related to your provided files.



