"""
PDF Chatbot — Streamlit UI with RAG + Claude streaming.
"""

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from rag import process_pdf, retrieve_context

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided PDF document context. If the answer is not in the context, say so "
    "clearly rather than guessing. Be concise and accurate."
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot")
st.caption("Upload a PDF, then ask anything about it.")

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": str, "content": str}]
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_name:
        with st.spinner("Reading & indexing PDF…"):
            try:
                file_bytes = uploaded.read()
                vectorstore, n_chunks = process_pdf(file_bytes, uploaded.name)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded.name
                st.session_state.messages = []
                st.success(f"✅ **{uploaded.name}**\n\n{n_chunks} chunks indexed.")
            except Exception as exc:
                st.error(f"Failed to process PDF: {exc}")

    if st.session_state.pdf_name:
        st.info(f"Active: **{st.session_state.pdf_name}**")
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Powered by LangChain · FAISS · Claude claude-opus-4-6")

# ── Guard: require PDF ────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your PDF…")
if not question:
    st.stop()

# Display user message
st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

# Retrieve relevant chunks
context = retrieve_context(st.session_state.vectorstore, question)

# Build the API message list:
# - History (previous turns, without injected context — they reference earlier answers)
# - Current user turn (with injected context)
api_messages = []
for msg in st.session_state.messages[:-1]:  # exclude the current question
    api_messages.append({"role": msg["role"], "content": msg["content"]})

api_messages.append({
    "role": "user",
    "content": (
        f"Use the following excerpts from the PDF to answer my question.\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}"
    ),
})

# Stream Claude's response
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

with st.chat_message("assistant"):
    placeholder = st.empty()
    full_response = ""

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=api_messages,
    ) as stream:
        for delta in stream.text_stream:
            full_response += delta
            placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)

st.session_state.messages.append({"role": "assistant", "content": full_response})
