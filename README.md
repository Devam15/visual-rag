# 🔍 Visual RAG — Technical Document Q&A

An AI-powered system that lets you ask questions about technical documents — including charts, diagrams, and images — using Retrieval-Augmented Generation (RAG) with vision capabilities.

## 📌 What it does

Most AI systems only read text. This system can **see and reason** through technical documents by:

- Extracting text from PDFs and splitting into searchable chunks
- Detecting images and diagrams and auto-captioning them using GPT-4o Vision
- Storing everything in a vector database (ChromaDB) using semantic embeddings
- Answering natural language questions by retrieving the most relevant text **and** images
- Citing the exact page and document each answer came from
- Showing confidence scores for each retrieved source
- Supporting per-document and cross-document search

## 🖥️ Demo

![Visual RAG Demo](demo.png)