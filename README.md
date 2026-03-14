🔍 RAG Search

A Retrieval-Augmented Generation (RAG) powered search application built in Python, featuring both a command-line interface and an interactive Streamlit web UI.

📖 Overview

RAG Search combines the power of semantic document retrieval with large language model generation to answer queries grounded in your own data. Instead of relying solely on a model's training knowledge, the system first retrieves the most relevant document chunks from a local data store and then passes them as context to the LLM - producing accurate, source-backed responses.

✨ Features

1. Semantic Search - Embeds documents and queries into vector space for similarity-based retrieval
2. Augmented Generation - Feeds retrieved context into an LLM to generate grounded answers
3. Streamlit UI - Clean, interactive web interface for querying the knowledge base
4. CLI Support - Run queries directly from the terminal via main.py
5. Local Data - Bring your own documents; data is stored and indexed locally in the data/ directory

📂 Adding Your Data

Place your source documents in the data/ directory. Supported formats depend on the loaders configured in src/. After adding new documents, re-run the indexing step (if applicable) before querying.
