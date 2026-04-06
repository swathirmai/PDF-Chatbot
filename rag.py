"""
RAG pipeline: PDF loading, chunking, embedding, and retrieval.
"""

import io
from typing import Optional

import pypdf
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Cache embeddings model across calls (expensive to reload)
_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def load_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """Extract text from a PDF and return one Document per page."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": filename},
                )
            )
    return pages


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed chunks and store in a FAISS index."""
    return FAISS.from_documents(chunks, _get_embeddings())


def process_pdf(file_bytes: bytes, filename: str) -> tuple[FAISS, int]:
    """Full pipeline: load → chunk → embed. Returns (vectorstore, chunk_count)."""
    pages = load_pdf(file_bytes, filename)
    if not pages:
        raise ValueError("No extractable text found in this PDF.")
    chunks = chunk_documents(pages)
    vectorstore = build_vectorstore(chunks)
    return vectorstore, len(chunks)


def retrieve_context(vectorstore: FAISS, query: str, k: int = TOP_K) -> str:
    """Return the top-k most relevant chunks joined as a single string."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )
