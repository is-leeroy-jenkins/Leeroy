# Text Generation

Text Generation mode provides the main chat workflow for interacting with the configured local
Leeroy model.

## 🧭 Purpose

Text Generation mode lets users submit prompts, apply system instructions, tune generation controls,
stream responses, and persist chat history in SQLite. It is the primary conversational interface for
drafting, analysis, summarization, explanation, and general local LLM work.

## 🧱 Workflow Position

```text

User Prompt
  │
  ├── System Instructions
  ├── Chat History
  ├── Optional Semantic Context
  └── Optional Basic Documents
  │
  ▼
Prompt Builder
  │
  ▼
Local llama.cpp Model
  │
  ▼
Streaming Chat Response
  │
  ▼
SQLite Chat History

```

## 🖥️ Opening Text Generation Mode

1. Start Leeroy.

```powershell

streamlit run app.py

```

2. In the sidebar, select:

```text

Text Generation

```

3. Confirm that the main page displays the Text Generation heading and chat interface.

## 🧠 System Instructions

Use the System Instructions expander to define behavior before submitting prompts.

Examples:

```text

Respond as a concise federal budget analyst. Use short paragraphs and include a recommendation.

```

```text

Explain technical terms clearly. Use a table when comparing options.

```

```text

Use only the provided context. If context is insufficient, say so.

```

System instructions are included in the prompt before the user message and can also be populated
from reusable templates.

## ⚙️ Response Controls

| Control       | Purpose                                              | Practical Guidance                                        |
| ------------- | ---------------------------------------------------- | --------------------------------------------------------- |
| Temperature   | Controls randomness.                                 | Use lower values for factual or deterministic output.     |
| Top-P         | Controls nucleus sampling.                           | Keep near default unless generation quality needs tuning. |
| Top-K         | Limits token candidates.                             | Lower values can make responses more conservative.        |
| Use Grounding | Indicates whether grounding behavior should be used. | Enable when using contextual material.                    |

## 🎚️ Probability Controls

| Control           | Purpose                                                  |
| ----------------- | -------------------------------------------------------- |
| Repeat Window     | Sets the recent token window used for repetition checks. |
| Repeat Penalty    | Discourages repeated phrases and loops.                  |
| Presence Penalty  | Penalizes tokens that have already appeared.             |
| Frequency Penalty | Penalizes tokens according to repeated frequency.        |

## 🎛️ Context Controls

| Control        | Purpose                                            |
| -------------- | -------------------------------------------------- |
| Context Window | Sets how much context the local model can process. |
| CPU Threads    | Controls CPU resources used by local inference.    |
| Max Tokens     | Limits the generated response length.              |
| Random Seed    | Supports reproducible output when fixed.           |

## 💬 Chat Workflow

1. Enter a prompt in the chat input.
2. Leeroy saves the user message.
3. Leeroy builds a model prompt from system instructions, optional context, and chat history.
4. The local model streams a response.
5. Leeroy saves the assistant response.
6. The conversation remains available through the current session and persisted SQLite history.

## 🧪 Example Prompts

```text

Summarize the operational risks of relying on a local GGUF model for document analysis.

```

```text

Create a concise checklist for validating a MkDocs documentation build.

```

```text 

Explain how SQLite-backed prompt storage improves repeatability for analysts.

```

## 🧹 Clearing Chat History

Use the Clear Chat button when you want to reset the current conversation.

Clearing chat history removes rows from the local `chat_history` table but does not remove prompt
templates, embeddings, imported data tables, or exception logs.

## ✅ Recommended Sequence

1. Set system instructions for the desired role or output style.
2. Use conservative generation settings for analytical work.
3. Submit the prompt.
4. Review the streamed response.
5. Adjust instructions rather than over-tuning sampling controls.
6. Clear chat history when moving to a different task context.

## 🔗 Related API Pages

| API Page                              | Purpose                                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| [App API](../api/app.md)              | Source documentation for prompt building, chat persistence, and local model execution. |
| [Configuration API](../api/config.md) | Runtime constants for model path, context defaults, and UI help text.                  |
