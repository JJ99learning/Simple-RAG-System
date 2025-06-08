# Simple RAG System with Jina Embeddings and Groq LLM

This is a Retrieval-Augmented Generation (RAG) system that combines document retrieval with large language model generation. It uses Jina AI for embeddings and Groq for the language model.

## Features

- PDF document loading and processing
- Text chunking with configurable size and overlap
- Vector storage using Chroma
- Semantic search capabilities
- Custom embedding implementation using Jina AI

## Prerequisites

- Python 3.8+
- Jina AI API key
- Groq API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/JJ99learning/Simple-RAG-System
cd Simple-RAG-System
```

2. Install required packages:

```bash
pip install langchain langchain-community langchain-groq chromadb pypdf python-dotenv requests
```

3. Create a `.env` file in the root directory with your API keys:

```
JINA_API_KEY=your_jina_api_key
GROQ_API_KEY=your_groq_api_key
```

## Project Structure

```
.
├── main.py              # Main RAG system implementation
├── embedder.py          # Custom Jina embedder implementation
├── data/               # Directory for your documents
│   └── your_pdf.pdf    # Your PDF documents
└── chromaDB/           # Vector database storage
```

## Components

### 1. Embedder (embedder.py)

- Custom implementation using Jina AI's embedding API
- Supports both document and query embedding
- Compatible with LangChain and Chroma interfaces

### 2. Main System (main.py)

- Document loading and processing
- Text chunking with RecursiveCharacterTextSplitter
- Vector storage using Chroma
- RAG chain implementation
- Interactive Q&A interface

## Usage

1. Place your PDF documents in the `data/` directory

2. Run the system:

```bash
python main.py
```

3. Enter your questions about the document when prompted

## Configuration

### Text Chunking

You can adjust the chunking parameters in `main.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Maximum characters per chunk
    chunk_overlap=200,  # Overlap between chunks
)
```

### RAG Prompt

The prompt template can be modified in `main.py`:

```python
template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the information provided."""
```

## How It Works

1. **Document Processing**:

   - Loads PDF documents
   - Splits text into manageable chunks
   - Creates embeddings for each chunk

2. **Vector Storage**:

   - Stores document chunks and their embeddings in Chroma
   - Enables efficient similarity search

3. **Query Processing**:
   - Takes user questions
   - Retrieves relevant document chunks
   - Generates answers using the Groq LLM

## Customization

- Modify the embedder implementation in `embedder.py`
- Adjust chunking parameters for different document types
- Customize the prompt template for different use cases
- Add support for different document types
