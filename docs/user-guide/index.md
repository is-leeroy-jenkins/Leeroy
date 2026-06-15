# User Guide

The Leeroy user guide explains how to operate each application mode from the Streamlit interface.

Leeroy is organized around five primary workflows: Text Generation, Document Q&A, Semantic Search,
Prompt Engineering, and Data Management. Each mode is selected from the sidebar and uses shared
configuration, session state, local persistence, and application styling.

## 🧭 Purpose

This guide provides task-oriented instructions for using Leeroy as a local analytical assistant. It
focuses on how users interact with the application, how application state is managed, and how each
mode contributes to the larger local-first workflow.

## 🧱 Workflow Position

```text id="ogx2o6"
Start Leeroy
  │
  ▼
Select Application Mode
  │
  ├── Text Generation
  ├── Document Q&A
  ├── Semantic Search
  ├── Prompt Engineering
  └── Data Management
  │
  ▼
Adjust Controls
  │
  ▼
Run Task
  │
  ▼
Review Output
```

## 🖥️ Application Modes

| Mode               | Use When                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| Text Generation    | You want to chat with the configured local model, draft content, summarize ideas, or reason through a prompt. |
| Document Q&A       | You want to ask questions about uploaded documents using retrieved document excerpts.                         |
| Semantic Search    | You want to create a local semantic index from uploaded text files.                                           |
| Prompt Engineering | You want to manage reusable prompt templates in SQLite.                                                       |
| Data Management    | You want to inspect, profile, filter, visualize, query, or administer SQLite tables.                          |

## ⚙️ Shared Runtime Controls

Many Leeroy modes use shared runtime controls.

| Control           | Purpose                                                            |
| ----------------- | ------------------------------------------------------------------ |
| Temperature       | Controls response randomness. Lower values are more deterministic. |
| Top-P             | Controls nucleus sampling.                                         |
| Top-K             | Limits candidate tokens considered during sampling.                |
| Repeat Penalty    | Reduces repetitive generation.                                     |
| Repeat Window     | Defines the token window used for repetition control.              |
| Presence Penalty  | Penalizes tokens that have already appeared.                       |
| Frequency Penalty | Penalizes tokens based on repeated frequency.                      |
| Context Window    | Sets the local model context size.                                 |
| CPU Threads       | Controls local inference CPU thread allocation.                    |
| Max Tokens        | Limits response length.                                            |
| Random Seed       | Supports more reproducible outputs when fixed.                     |

## 🧠 System Instructions

System instructions let you guide the assistant’s behavior before user prompts are processed.

Use system instructions for:

| Use Case    | Example                                                     |
| ----------- | ----------------------------------------------------------- |
| Tone        | “Respond in concise federal analytical style.”              |
| Format      | “Use bullet points and include a recommendation.”           |
| Grounding   | “Only answer from the provided document excerpts.”          |
| Role        | “Act as a budget analyst reviewing appropriation language.” |
| Constraints | “Do not speculate beyond the supplied data.”                |

System instructions can be typed directly or loaded from reusable prompt templates in the Prompt
Engineering mode.

## 💾 Persistence

Leeroy uses SQLite to preserve important application data.

| Data             | Storage                                         |
| ---------------- | ----------------------------------------------- |
| Chat history     | `chat_history` table                            |
| Prompt templates | `Prompts` table                                 |
| Semantic chunks  | `embeddings` table                              |
| Imported data    | User-created SQLite tables                      |
| Exception logs   | Configured logging database and exception table |

## ✅ Recommended Operating Sequence

1. Start the app with `streamlit run app.py`.
2. Select the application mode in the sidebar.
3. Confirm the local model status.
4. Configure system instructions if the task requires a specific role or format.
5. Adjust generation controls only when needed.
6. Load documents, prompts, or tables depending on the mode.
7. Run the task.
8. Review output and sources.
9. Clear or reset state only when the workflow is complete.

## 🔗 Related Pages

| Page                                        | Purpose                                             |
| ------------------------------------------- | --------------------------------------------------- |
| [Text Generation](text-generation.md)       | Chat and local generation workflow.                 |
| [Document Q&A](document-qna.md)             | Document upload, retrieval, and grounded answering. |
| [Semantic Search](semantic-search.md)       | Embedding and semantic context workflow.            |
| [Prompt Engineering](prompt-engineering.md) | Prompt-template administration.                     |
| [Data Management](data-management.md)       | SQLite analysis and administration workflow.        |
