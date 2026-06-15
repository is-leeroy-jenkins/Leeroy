# Document Q&A

Document Q&A mode lets users upload documents, extract text, retrieve relevant chunks, and ask
grounded questions against document content.

## 🧭 Purpose

Document Q&A mode supports retrieval-augmented generation over uploaded files. It stores active
document bytes in Streamlit session state, extracts text, chunks the content, embeds chunks,
retrieves relevant excerpts, and builds a document-grounded prompt for the local model.

## 🧱 Workflow Position

```text

Upload Document
  │
  ▼
Store Document Bytes
  │
  ▼
Extract Text
  │
  ▼
Chunk Text
  │
  ▼
Generate Embeddings
  │
  ├── sqlite-vec Retrieval
  │
  └── Cosine Fallback Retrieval
  │
  ▼
Build Grounded Prompt
  │
  ▼
Generate Answer

```

## 📥 Loading Documents

1. Select `Document Q&A` from the sidebar.
2. Open the Document Loader expander.
3. Upload one or more supported files.

Supported upload types:

| Type | Notes                                                                                           |
| ---- | ----------------------------------------------------------------------------------------------- |
| PDF  | Text extraction uses PyMuPDF.                                                                   |
| TXT  | Text-like content can be decoded directly.                                                      |
| DOCX | Upload control accepts the file type; extraction support depends on the active extraction path. |

After upload, Leeroy stores:

| Session Key   | Purpose                                    |
| ------------- | ------------------------------------------ |
| `uploaded`    | Uploaded file objects.                     |
| `active_docs` | Names of active documents.                 |
| `doc_bytes`   | Raw document bytes keyed by document name. |

## 👁️ Document Preview

When a document is active and previewable, Leeroy displays the first active document in the preview
panel.

If preview bytes are unavailable, the app displays a notice instead of failing.

## 🧠 Retrieval Pipeline

Document Q&A uses a retrieval workflow instead of stuffing full documents directly into every
prompt.

| Step        | Description                                                                    |
| ----------- | ------------------------------------------------------------------------------ |
| Fingerprint | Builds a stable fingerprint from active document names and bytes.              |
| Extract     | Extracts document text from bytes.                                             |
| Chunk       | Splits extracted text into overlapping chunks.                                 |
| Embed       | Encodes chunks with the sentence-transformer model.                            |
| Store       | Uses sqlite-vec if available, otherwise stores fallback rows in session state. |
| Retrieve    | Retrieves the top relevant chunks for the user query.                          |
| Prompt      | Builds a grounded prompt using the retrieved excerpts.                         |

## 🔍 sqlite-vec and Fallback Retrieval

Leeroy attempts to use sqlite-vec for vector retrieval when available. If sqlite-vec is unavailable
or cannot be loaded, the app falls back to in-memory cosine similarity.

| Retrieval Path  | Use Case                                                   |
| --------------- | ---------------------------------------------------------- |
| sqlite-vec      | Preferred path when the extension is installed and usable. |
| Cosine fallback | Safe fallback when sqlite-vec is unavailable.              |

This keeps Document Q&A usable across more environments.

## 💬 Asking Questions

After loading a document, use the chat input to ask a question.

Example questions:

```text

What is the purpose of this document?

```

```text

Summarize the key findings and identify any risks or open issues.

```

```text

List the major requirements and explain which ones appear mandatory.

```

```text

What evidence in the document supports the main conclusion?

```

## 🧾 Grounded Answer Behavior

The Document Q&A prompt instructs the model to use retrieved document excerpts. If the excerpts do
not contain enough information, the model should say that the available context is insufficient.

For best results:

| Practice                              | Reason                                                          |
| ------------------------------------- | --------------------------------------------------------------- |
| Ask specific questions                | Improves retrieval precision.                                   |
| Use documents with extractable text   | Scanned images may not produce useful text unless OCR is added. |
| Keep related files loaded together    | Helps retrieval find related context.                           |
| Avoid extremely broad first questions | Broad questions can retrieve diffuse context.                   |

## 🧹 Unloading Documents

Use the Unload Document control to clear uploaded documents, active document names, and stored
document bytes.

This resets the active document context without clearing chat history or prompt templates.

## ✅ Recommended Sequence

1. Open Document Q&A.
2. Upload a document.
3. Confirm preview availability.
4. Ask a narrow question first.
5. Review the answer for document grounding.
6. Ask follow-up questions with specific terms from the document.
7. Unload documents when switching topics.

## 🔗 Related API Pages

| API Page                              | Purpose                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [App API](../api/app.md)              | Source documentation for document extraction, chunking, fingerprinting, indexing, retrieval, and grounded prompt construction. |
| [Configuration API](../api/config.md) | Runtime constants and UI help text used by Document Q&A.                                                                       |
