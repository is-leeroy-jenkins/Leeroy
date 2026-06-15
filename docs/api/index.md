# Leeroy Documentation

Leeroy is a local-first Streamlit application for local language-model inference,
retrieval-augmented generation, semantic search, prompt engineering, and SQLite-backed data
management.

The application is designed for analysts, developers, and data-science users who need a controlled
desktop or server-hosted assistant that can work with local prompts, uploaded documents, reusable
templates, embedded text, and tabular data.

## 🧭 Purpose

Leeroy provides a practical interface around a local GGUF model, local document retrieval,
persistent prompt storage, and SQLite data tooling. The application favors a durable architecture:
source files remain readable, runtime state is visible, and critical application data is stored in
local databases rather than hidden behind external services.

Leeroy supports five primary workflows:

| Workflow           | Description                                                                          |
| ------------------ | ------------------------------------------------------------------------------------ |
| Text Generation    | Chat with the configured local model using Streamlit chat controls.                  |
| Document Q&A       | Upload documents, extract text, build retrieval context, and ask grounded questions. |
| Semantic Search    | Embed uploaded text and use vector similarity for semantic context.                  |
| Prompt Engineering | Create, search, edit, and reuse prompt templates stored in SQLite.                   |
| Data Management    | Import, inspect, profile, filter, aggregate, visualize, and query SQLite tables.     |

## 🧱 Application Position

Leeroy sits between a user-facing Streamlit interface and a local model/runtime stack.

```text
User
  │
  ▼
Streamlit Interface
  │
  ├── Text Generation
  ├── Document Q&A
  ├── Semantic Search
  ├── Prompt Engineering
  └── Data Management
  │
  ▼
Shared Prompt, Retrieval, and Persistence Utilities
  │
  ├── Local GGUF model through llama.cpp
  ├── Sentence-transformer embeddings
  ├── SQLite persistence
  ├── sqlite-vec when available
  └── Plotly/pandas analytical views
```

## ✨ Key Capabilities

| Capability            | Purpose                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| Local model execution | Runs an optional local GGUF model through `llama-cpp-python`.                                  |
| Lazy model loading    | Allows the interface to load even when the model file is unavailable.                          |
| Streaming chat        | Streams generated output into the Streamlit chat interface.                                    |
| System instructions   | Lets users apply persistent behavioral instructions to model responses.                        |
| Prompt templates      | Stores reusable prompts in the SQLite `Prompts` table.                                         |
| Chat persistence      | Saves conversation history in SQLite.                                                          |
| Document upload       | Accepts user documents for preview and retrieval workflows.                                    |
| Text extraction       | Uses PyMuPDF for PDF text extraction and defensive decoding for text-like content.             |
| Chunking              | Splits documents into overlapping retrieval windows.                                           |
| Embeddings            | Uses `sentence-transformers` for vector representations.                                       |
| Vector retrieval      | Uses sqlite-vec when available, with cosine-similarity fallback.                               |
| SQLite management     | Provides table browsing, CRUD-style operations, profiling, schema actions, and guarded SQL.    |
| Visualization         | Renders histograms, bars, lines, scatter plots, box plots, pies, and correlations with Plotly. |
| Exception logging     | Writes wrapped exception diagnostics to the configured SQLite logging database.                |

## 🧩 Main Source Files

| File               | Purpose                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `app.py`           | Main Streamlit application, mode routing, prompt pipeline, document retrieval, semantic search, and data-management utilities. |
| `config.py`        | Central configuration for paths, model settings, runtime defaults, UI labels, app modes, logging paths, and help text.         |
| `boogr.py`         | Exception wrapper and SQLite-backed logger used by application error-handling paths.                                           |
| `requirements.txt` | Runtime and documentation dependencies.                                                                                        |
| `README.md`        | Project overview, installation notes, local model details, and application summary.                                            |

## 🧠 Local-First Design

Leeroy is built around a local-first operational model.

The model file can live outside the repository, the application database is local SQLite, and the
document retrieval pipeline can operate without requiring a remote vector database. This makes the
project suitable for controlled environments where repeatability, explainability, and operational
simplicity matter.

## 📚 Documentation Layout

| Section       | Description                                                                        |
| ------------- | ---------------------------------------------------------------------------------- |
| Architecture  | Explains the application layers, data flow, retrieval flow, and persistence model. |
| User Guide    | Provides task-oriented instructions for using each application mode.               |
| API Reference | Renders source-driven API documentation from Google-style Python docstrings.       |
| Development   | Describes setup, validation, MkDocs build steps, and GitHub Pages deployment.      |

## 🚀 Quick Start

Install dependencies from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the Streamlit app.

```powershell
streamlit run app.py
```

Build the documentation.

```powershell
mkdocs build
```

Serve the documentation locally.

```powershell
mkdocs serve
```

## ✅ Recommended Reading Order

1. Read the [Architecture](architecture.md) page to understand the system.
2. Review the [User Guide](user-guide/index.md) for task-oriented workflows.
3. Use the [API Reference](api/index.md) for source-level details.
4. Use the [Development](development.md) page when changing code or publishing documentation.
