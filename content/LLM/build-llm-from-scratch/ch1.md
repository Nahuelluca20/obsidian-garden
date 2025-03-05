---
title: "ch1 - Understanding the LLMs"
publishedAt: "2024-11-20"
summary: "Brief introduction on what is an LLM."
tags:
  - LLM
  - ai
---

LLMs have remarkable capabilities to understand, generate, and interpret human language.

> _Understand:_ in LLMs understand means that they can process and generate text in ways that appear coherent and contextually relevant, not that they possess human-like consciousness or
> comprehension.

Before LLMs, we had NLP(natural language process) that models be trained for specific tasks, but when LLMs appear, they are better in many aspects than NLP models.

The success behind LLMs can be attributed to the transformer architecture that
underpins many LLMs and the vast amounts of data on which LLMs are trained,
allowing them to capture a wide variety of linguistic nuances, contexts, and patterns
that would be challenging to encode manually.

## What is a LLM?

An LLM is a neural network designed to understand, generate, and respond to humanlike text.

Large language models are deep learning networks trained on billions of text data points on the Internet.

LLMs utilize an architecture called the transformer, which allows them to pay selective attention to different parts of the input when making predictions, making them
especially adept at handling the nuances and complexities of human language.

![Screenshot 2024-11-20 121800.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/edf1ac1a-c51e-41fb-89d7-6c64ff778a98)

## Stages of building and using LLMs

We can train an existing LLM model to perform a specific task. This process is called `fine-tuning`. Using a pre-trained model such as Llama3.2 like a _foundation model_ we pass a Labeled dataset to train this model in a specific domain.
![Screenshot 2024-11-20 134241.png](https://collected-notes.s3.us-west-2.amazonaws.com/uploads/28846/3918a142-7dbd-4d61-bfbd-12b8f33a1d35)

**Summary**

- LLMs have transformed the field of natural language processing, which previously mostly relied on explicit rule-based systems and simpler statistical methods. The advent of LLMs introduced new deep learning-driven approaches that led to advancements in understanding, generating, and translating human language.
- Modern LLMs are trained in two main steps:
  First, they are pretrained on a large corpus of unlabeled text by using the prediction of the next word in a sentence as a label.
  Then, they are fine-tuned on a smaller, labeled target dataset to follow instructions or perform classification tasks.
- LLMs are based on the transformer architecture. The key idea of the transformer architecture is an attention mechanism that gives the LLM selective access to the whole input sequence when generating the output one word at a time.
- The original transformer architecture consists of an encoder for parsing text and a decoder for generating text.
- LLMs for generating text and following instructions, such as GPT-3 and ChatGPT, only implement decoder modules, simplifying the architecture.
- Large datasets consisting of billions of words are essential for pretraining LLMs.
- While the general pretraining task for GPT-like models is to predict the next word in a sentence, these LLMs exhibit emergent properties, such as capabilities to classify, translate, or summarize texts.
- Once an LLM is pretrained, the resulting foundation model can be fine-tuned more efficiently for various downstream tasks.
- LLMs fine-tuned on custom datasets can outperform general LLMs on specific tasks.
