# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines document retrieval with answer generation. A basic RAG pipeline loads documents, splits them into chunks, converts chunks into searchable representations, retrieves relevant passages for a question, and uses those passages to form an answer.

This project uses TF-IDF vectors and cosine similarity for local retrieval. TF-IDF is lightweight and works without a hosted embedding model or API key. The current answer step is extractive: it returns the best matching passage and includes source excerpts. It is reproducible but is not a generative large language model.

To extend the project, use sentence-transformers or a hosted embedding model and a vector database such as FAISS or Chroma. For generated answers, pass retrieved context to a local or hosted LLM, keep citations tied to retrieved chunks, and evaluate groundedness and retrieval quality.
