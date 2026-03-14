# 🔎 RAG Search

### Production-Grade Retrieval-Augmented Generation Pipeline for Intelligent Document Search

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Local-green)
![VectorDB](https://img.shields.io/badge/VectorDB-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **high-performance Retrieval-Augmented Generation (RAG) system**
designed for semantic search and knowledge-grounded question answering
across large document collections.

------------------------------------------------------------------------

# 🚀 Demo

Example query:

User Question: What is Retrieval-Augmented Generation?

System response:

Retrieval-Augmented Generation (RAG) is a hybrid AI architecture that
combines information retrieval with large language models. It retrieves
relevant documents from a knowledge base and uses them as context for
generating accurate responses.

------------------------------------------------------------------------

# ✨ Key Features

-   🔍 Semantic Search\
-   🧠 Context-Aware LLM Responses\
-   ⚡ High Performance Retrieval\
-   📄 Automated Document Pipeline\
-   🧩 Modular Architecture\
-   🏗 Production-Ready Design

------------------------------------------------------------------------

# 🧠 System Architecture

User Question → Query Embedding → Vector Database → Top-K Documents →
Context Builder → LLM → Final Response

------------------------------------------------------------------------

# 📂 Repository Structure

RAG_Search

├── data/

├── src/

│ ├── ingestion

│ ├── embeddings

│ ├── retrieval

│ ├── generation

│ └── utils

├── main.py

├── requirements.txt

└── README.md

------------------------------------------------------------------------

# ⚙️ Installation

## Clone Repository

git clone https://github.com/rohibindal17/RAG_Search.git cd RAG_Search

## Create Virtual Environment

python -m venv venv

Windows: venv`\Scripts`{=tex}`\activate`{=tex}

Mac/Linux: source venv/bin/activate

## Install Dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# ▶️ Run the Project

python main.py

------------------------------------------------------------------------

# 🛠 Tech Stack

  Layer        Technology
  ------------ -----------------------
  Language     Python
  LLM          OpenAI / Local LLM
  Framework    LangChain
  Vector DB    FAISS / Chroma
  Embeddings   Sentence Transformers

------------------------------------------------------------------------

# 👨‍💻 Author

**Rohi Bindal**\
AI/ML Researcher\
MSc Machine Learning --- Stevens Institute of Technology

GitHub: https://github.com/rohibindal17

------------------------------------------------------------------------

# ⭐ Support

If you found this useful:

⭐ Star the repository\
🍴 Fork the project\
📢 Share with others
