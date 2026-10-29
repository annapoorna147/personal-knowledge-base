# Personal Knowledge Base AI

An AI-powered personal knowledge assistant that lets you upload PDF and
TXT documents, search their contents using semantic retrieval, and ask
questions with answers grounded in retrieved passages.

## Features

- PDF and TXT document ingestion
- Text extraction and overlapping chunking
- Semantic embeddings using Sentence Transformers
- Persistent vector storage with ChromaDB
- Question answering using the Gemini API
- Source-aware answers with document and page references
- Streamlit web interface

## Architecture

1. Upload a PDF or TXT document.
2. Extract and split its text into chunks.
3. Convert chunks into embeddings with `all-MiniLM-L6-v2`.
4. Store chunks and metadata in ChromaDB.
5. Embed a user's question and retrieve relevant passages.
6. Send the passages and question to Gemini.
7. Display the answer and retrieved sources.

## Tech Stack

- Python
- Streamlit
- Google Gemini API (`google-genai`)
- ChromaDB
- Sentence Transformers
- pypdf
- python-dotenv

## Requirements

- Python 3.11 or newer recommended
- A Gemini API key

## Setup

Clone the repository:

```bash
git clone https://github.com/annapoorna147/personal-knowledge-base.git
cd personal-knowledge-base
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Never commit your `.env` file or share your API key.

Start the application:

```bash
streamlit run app.py
```

## Project Structure

```text
personal-knowledge-base/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── answer_generator.py
│   ├── document_processor.py
│   └── vector_store.py
├── tests/
└── data/
```

## Current Limitations

- Retrieval quality depends on document text extraction and chunking.
- Scanned PDFs may require OCR support.
- Gemini answers require network access and a working API key.
- ChromaDB uses local persistent storage by default.

## Security

Do not upload API keys, private documents, virtual environments, or
the local ChromaDB database to a public repository.

## Future Improvements

- Automated tests for document processing and retrieval
- Improved document management and duplicate detection
- Better error handling and observability
- OCR and additional document formats
- Deployment with appropriate persistent storage
