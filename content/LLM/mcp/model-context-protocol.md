---
title: Model Context Protocol
publishedAt: "2025-03-22"
tags:
  - ai
  - llm
summary: Exploring How the Model Context protocol works, the new way to add context to an LLM model.
---

The Model Context Protocol is a way tho standardizes how applications provide context to LLMs.

## Architecture

The general architecture use this stuff for work:

- **MCP Host:** Programs that enable us to access to data with MCP. e.g Claude Desktop
- **MCP Client:** Protocol clients that maintain 1:1 connections with servers
- **MCP Servers:** Programs that use MCP for expose capabilities
- **Local Data Sources:** Files, databases, and services that MCP servers can securely access
- **Remote Services**: External systems available over the internet
  ![[Pasted image 20250322163620.png]]

# Core Architecture

MCP follows a client-server architecture:

- Host are LLMs applications
- Clients maintain connections with server inside the host applications
- Servers provide context, tools and prompts to the clients.

## Core components

### Protocol Layer

The protocol layer handles message framing, request/response linking, and high-level communication patterns.

### Transport layer

The transport layer handles the actual communication between clients and servers.

### Message types

MCP has these main types of messages:

- Requests
- Results
- Errors
- Notifications

## Connection lifecycle

### Initialization

1. Client sends `initialize` request with protocol version and capabilities
2. Server responds with its protocol version and capabilities
3. Client sends `initialized` notification as acknowledgment
4. Normal message exchange begins

### Message exchange

After initialization, the following patterns are supported:

- **Request-Response**: Client or server sends requests, the other responds
- **Notifications**: Either party sends one-way messages

### Termination

Either party can terminate the connection:

- Clean shutdown via `close()`
- Transport disconnection
- Error conditions

# Resources

The data that the server expose for the LLMs in the client host. Kind of data like:

- File contents
- Database records
- API responses
- Live system data
- Screenshots and images
- Log files
- And more
