---
tags:
  - programming
  - ai
title: AI Agents
publishedAt: "2025-02-27"
summary:
---

## What are AI agents?

They are autonomous systems that perform decision-making tasks, manage tools, solve problems. While a traditional workflow is deterministic and linear, AI agents are non-linear, non-deterministic and can change every time they are re-executed.

With the help of LLMs the agents can make decisions and call the relevant tools for execute the task.

## How Agents work

The agents have 3 principal components:

- **Decision Engine**: a LLM that determinate the following action
- **Tool Integration**: the tools that the agents can use, ex. an APIs
- **Memory System**: keep the context and the progress of the task.

Then the agents operates in a loop:

1. Observing the current state or task
2. Planning the following action
3. Executing those action using the tools
4. Learning, store the results, update the progress.

## Workflows

They are a structure that coordinates how the components of an agent work together. It is a framework that controls how decisions are executed.

## Tools

Is the way that agents have for interact with external services, systems, manipulate data. Tools are typically implemented as function calls that the AI can use to accomplish specific tasks.

## Human-in-the-Loop (HITL)

Workflows have a designed point for stop and integrate a human judgment. These workflows pause at critical points for human review, validation, or decision-making before proceeding.

#### Resources

- [What are AI Agents - IBM](https://www.ibm.com/think/topics/ai-agents)
- [Agentes - Cloudflare](https://developers.cloudflare.com/agents/concepts/what-are-agents/)
-
