---
tags:
  - ai
publishedAt: "2025-02-27"
title: Deep dive into LLMs - Part 2
summary:
---

## Introduction

Here Andrej talks about the base model that some companies like Meta release for free. But, this are `base models` and we want something more useful that can response our questions for example.

## Try base models

For see the difference between base models and models ready for use we can use [hyperbolic](https://hyperbolic.xyz/) that allow us run models in the cloud. Used that we can see that the `base model` for now is not a assistant, is a token autocomplete (the only thing that model do is predict the next token in the sequence) and is a stochastic system.

The model tries to predict the next token, for example if we use a `base model` like `llama 3.1` and we put an old wikipedia entry probably the model will return the first tokens exactly like the wikipedia article this is called regurgitation, but if we use new data for example from 2024 (the dataset I used called 3.1 had data up to 2023) the model will start to freak out since that is not in its data.

### Resource

- [Hyperbolic](https://hyperbolic.xyz/)
-
