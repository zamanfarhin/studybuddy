# StudyBuddy - Agentic RAG Study Assistant

An intelligent study assistant that combines Retrieval-Augmented Generation (RAG) with an agentic decision layer to answer questions from your own uploaded documents or general knowledge; automatically choosing the best approach for each query. Built with AI-assisted development.

---

## Features

- **PDF Upload**: Upload course materials, textbooks, or notes and ask questions directly about them
- **Three-Way Agentic Decision Layer**: For every query, the agent decides whether to answer from (1) general knowledge (`memory`), (2) uploaded documents (`retrieve`), or (3) both (`hybrid`) — using GPT-4o function calling for structured, reliable classification
- **Conversational Memory**: Maintains multi-turn conversation history for contextual follow-up questions
- **Experiment Framework**: Systematic 180-run evaluation across models, chunk sizes, and query types to measure routing accuracy and answer quality
- **Transparent Reasoning**: The UI shows the agent's routing decision and reasoning for every response

---

## Tech Stack

- **Python** — core application logic
- **Streamlit** — web UI
- **OpenAI GPT-4o** — query classification and response generation
- **ChromaDB** — vector store for document embeddings
- **PyMuPDF (fitz)** — PDF parsing and text extraction

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd studybuddy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Create a .env file with:
OPENAI_API_KEY=your_openai_api_key_here

# 4. Run the app
streamlit run app.py
```

---

## How It Works

Every user query passes through a three-stage pipeline:

1. **Classify** — The agent uses GPT-4o function calling to route the query to one of three strategies:
   - `memory` — general knowledge questions with no document reference
   - `retrieve` — questions explicitly about uploaded documents
   - `hybrid` — questions requiring both document context and general explanation

2. **Retrieve** (if needed) — Relevant chunks are fetched from ChromaDB using semantic similarity search

3. **Generate** — GPT-4o produces a response using the appropriate context (none, document chunks, or both), with conversation history included for multi-turn coherence

---

## Experiments

A systematic evaluation of **180 runs** was conducted across:
- 3 OpenAI models: `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`
- 3 chunk sizes: 256, 512, 1024 tokens
- 4 query types: memory, retrieve, hybrid, edge cases
- 5 trials per configuration

**Key findings:**
- Overall routing accuracy: **96.7%**
- GPT-4o achieved **100% routing accuracy** across all query types
- Chunk size had minimal impact on routing accuracy but affected retrieval quality
- Hybrid queries were the most challenging, with smaller models occasionally misclassifying them

---

## Course

**COGS 185 — Final Project**
University of California, San Diego
March 2026
