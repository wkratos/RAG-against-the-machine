*This project has been created as part of the 42 curriculum by wkrati*
# RAG Against the Machine

A Retrieval-Augmented Generation (RAG) system for answering questions about the vLLM source code and documentation.

## Overview

The project uses three main stages:

1. **Indexing:** Split source files into chunks and represent them using TF-IDF.
2. **Retrieval:** Find the most relevant chunks for a user's question.
3. **Generation:** Use Qwen3-0.6B to generate an answer based on the retrieved context.

## Technologies

- Python
- uv
- scikit-learn (TF-IDF)
- Transformers and PyTorch
- Qwen3-0.6B
- Python Fire

## Installation

```bash
make install
```

## Usage

Run the CLI:

```bash
make run
```

Search for relevant sources:

```bash
uv run python -m src search "What is LoRA?" --k=5
```

Generate an answer:

```bash
uv run python -m src answer "What is LoRA?" --k=5
```

## Evaluation

Retrieval is evaluated using Recall@5 and source-range IoU.

| Dataset | Public Recall@5 | Target |
|---|---:|---:|
| Documentation | 81% | 80% |
| Code | 52.53% | 50% |

These results were measured on the public datasets and do not guarantee the same performance on unseen questions.

## Current Status

The basic end-to-end RAG pipeline is functional.

Remaining tasks include batch answering, validation, type checking, and additional testing.
