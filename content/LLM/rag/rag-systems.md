---
title: "RAG Systems"
publishedAt: "2024-10-12"
summary: "What is a RAG and what does it do?"
tags:
  - LLM
  - ai
---

RAG is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs' generative process.

## What are the benefits of RAG?

RAG method combine the power of LLMs and a external base of knowledge for generate more accurate responses and reduce hallucinations by retrieving relevant documents, that will be more useful for user.

## RAG concepts

RAG system use a document, embedding or vector database for store relevant information and a LLM for generate the response. The process is the following:

1. Embedding the user request (transformation into a vector based on semantic meaning).
2. Search the storage using semantic search (e.g. vector database like chormaDB).
3. Generate a response with a LLM.

The RAG approach is particularly effective for tasks that require a deep understanding of context and the ability to reference multiple sources of information. This includes tasks such as question answering, where the model needs to consider multiple sources of knowledge and choose the most appropriate one based on the context of the question.

### Glossary

**Retrieval:** This refers to the process of obtaining or fetching data or information from a storage location. In the context of databases or search engines, it’s about fetching the relevant data based on a specific query.

**Vector Similarity Search:** At its core, vector similarity search involves comparing vectors (lists of numbers) to determine how similar they are.

**Vector Database:** A database that is designed to store vectors. These databases are optimized for vector similarity searches.

**LLM:** It stands for “Learned Language Model”, a type of machine learning model designed to work with text.

**Chunking:** This is the process of taking input data (like text) and dividing it into smaller, manageable pieces or “chunks”. This can make processing more efficient, especially in contexts like natural language processing where the input might be very long.

**Embeddings**/**Vectors**: In machine learning, embeddings refer to the conversion of discrete items (like words or products) into continuous vectors. These vectors capture the semantic meaning or relationship between the items in a way that can be processed by algorithms.

### Resources

- [retrieval augmented generation](https://redis.io/glossary/retrieval-augmented-generation/)
