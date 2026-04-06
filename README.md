# PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and have a streaming conversation about its contents. Built with LangChain, FAISS, and Claude Opus 4.6.

---

## Features

- **PDF Upload** — supports any text-based PDF (reports, papers, manuals, books)
- **Automatic chunking** — splits text into overlapping chunks for accurate retrieval
- **Local embeddings** — uses `sentence-transformers` (no API cost, runs on CPU)
- **FAISS vector search** — fast in-memory similarity search over embedded chunks
- **Streaming responses** — Claude answers stream token-by-token in real time
- **Multi-turn memory** — conversation history is preserved across questions
- **Page attribution** — retrieved chunks include the source page number

---

## Project Structure

```
PDF Chatbot/
├── app.py              # Streamlit UI and chat logic
├── rag.py              # RAG pipeline (load, chunk, embed, retrieve)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## Tech Stack

| Component | Library / Model |
|---|---|
| UI | Streamlit |
| PDF parsing | pypdf |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS (in-memory) |
| LLM | Claude Opus 4.6 via Anthropic API |
| Env management | python-dotenv |

---

## Setup

### 1. Clone / open the project

```bash
cd "PDF Chatbot"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` model (~90 MB). It is cached locally after that.

### 3. Set your API key

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. Open the app in your browser
2. Click **Browse files** in the sidebar and upload a PDF
3. Wait for indexing (a few seconds depending on PDF size)
4. Type a question in the chat box and press Enter
5. Claude streams an answer grounded in the PDF content
6. Continue asking follow-up questions — the chat history is maintained

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INDEXING                             │
│                                                             │
│  PDF file                                                   │
│     │                                                       │
│     ▼                                                       │
│  load_pdf()          pypdf — extracts text page by page     │
│     │                                                       │
│     ▼                                                       │
│  chunk_documents()   RecursiveCharacterTextSplitter         │
│                      chunk_size=1000, overlap=200           │
│     │                                                       │
│     ▼                                                       │
│  build_vectorstore() HuggingFaceEmbeddings (MiniLM-L6-v2)  │
│                      → FAISS in-memory index                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERYING                             │
│                                                             │
│  User question                                              │
│     │                                                       │
│     ▼                                                       │
│  retrieve_context()  FAISS similarity search → top-4 chunks │
│     │                                                       │
│     ▼                                                       │
│  Claude Opus 4.6     system prompt + context + question     │
│                      → streamed answer                      │
│     │                                                       │
│     ▼                                                       │
│  Streamlit UI        token-by-token display + chat history  │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

All tunable constants are at the top of [rag.py](rag.py):

| Constant | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

Claude model and `max_tokens` are set in [app.py](app.py).

---

## Limitations

- **In-memory only** — the FAISS index is lost when the app restarts; re-upload the PDF to re-index
- **Text PDFs only** — scanned/image PDFs without embedded text will yield no results
- **Single document** — uploading a new PDF replaces the previous index and clears chat history
- **No persistent storage** — conversation history exists only for the current session

---

## Dependencies

```
anthropic>=0.40.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
faiss-cpu>=1.7.4
pypdf>=4.0.0
streamlit>=1.40.0
python-dotenv>=1.0.0
sentence-transformers>=3.0.0
```
