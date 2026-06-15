# Semantic Search

Semantic Search mode lets users upload text-oriented files, split content into chunks, generate
embeddings, and store semantic context in SQLite for later retrieval.

## 🧭 Purpose

Semantic Search mode supports local embedding workflows. It converts uploaded text into chunks,
encodes those chunks with the configured sentence-transformer model, and stores the resulting
text/vector pairs in the local SQLite `embeddings` table.

Semantic context can then be used by the Text Generation pipeline when semantic grounding is
enabled.

## 🧱 Workflow Position

```text id="mpsx0m"
Upload Files
  │
  ▼
Decode Text
  │
  ▼
Chunk Text
  │
  ▼
Generate Embeddings
  │
  ▼
Store Chunks and Vectors in SQLite
  │
  ▼
Use Semantic Context in Prompt Builder
```

## 🔍 Opening Semantic Search Mode

1. Start Leeroy.

```powershell id="emw3he"
streamlit run app.py
```

2. Select the sidebar mode:

```text id="v1ss95"
Semantic Search
```

3. Confirm that the Semantic Search page displays the semantic context toggle and upload control.

## 📥 Uploading Files

Use the upload control to load files for embedding.

The Semantic Search workflow reads uploaded file bytes and attempts to decode them as text. The
decoded text is split into overlapping chunks before embedding.

For best results, use files with extractable text content.

| File Content     | Expected Behavior                                                |
| ---------------- | ---------------------------------------------------------------- |
| Plain text       | Best fit for direct decoding and chunking.                       |
| Markdown         | Good fit because text structure is preserved.                    |
| CSV-like text    | Usable if the content is meaningful as text.                     |
| Binary documents | May not decode usefully in this mode. Use Document Q&A for PDFs. |

## 🧩 Chunking

Leeroy uses the shared text chunking helper to split uploaded content into overlapping chunks.

Chunking improves retrieval because it allows the application to compare the user’s prompt against
smaller, focused text windows rather than one large document.

| Chunking Feature | Purpose                                       |
| ---------------- | --------------------------------------------- |
| Fixed chunk size | Keeps retrieved context manageable.           |
| Overlap          | Preserves continuity across chunk boundaries. |
| Ordered chunks   | Keeps source text progression intact.         |

## 🧠 Embedding

Semantic Search uses the local sentence-transformer embedder loaded by the application.

The default model is:

```text id="tjg2bo"
all-MiniLM-L6-v2
```

Each chunk is encoded into a vector. Leeroy stores the chunk text and vector bytes in SQLite.

## 🗄️ SQLite Storage

Semantic Search writes to the local `embeddings` table.

| Column   | Purpose                                |
| -------- | -------------------------------------- |
| `id`     | Row identifier.                        |
| `chunk`  | Text chunk used as semantic context.   |
| `vector` | Vector representation stored as bytes. |

When a new semantic index is built, the app clears existing embedding rows and writes the new
chunk/vector set.

## 🔗 Semantic Context in Text Generation

When semantic context is enabled, the prompt builder can retrieve the most similar stored chunks for
a user prompt and inject those chunks into the system context before local model generation.

This lets Text Generation use previously embedded material without manually pasting it into the
prompt.

## 🧪 Example Workflow

1. Open Semantic Search.
2. Enable semantic context.
3. Upload one or more text files.
4. Wait for the semantic index to build.
5. Switch to Text Generation.
6. Ask a question that relates to the uploaded content.

Example prompt after indexing:

```text id="t2m5l4"
Using the available semantic context, summarize the main operating assumptions and risks.
```

## ✅ Recommended Sequence

1. Use clean, text-based files.
2. Build the semantic index before asking related questions.
3. Enable semantic context only when the indexed content is relevant.
4. Rebuild the index when changing the source material.
5. Keep prompts specific enough to retrieve the right chunks.

## 🧯 Troubleshooting

| Issue                           | Likely Cause                                                           | Fix                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| No semantic index built         | No readable text chunks were produced.                                 | Use text-based files or convert binary files to text.                                 |
| Embedding model unavailable     | Sentence-transformer model could not load.                             | Confirm dependencies are installed and internet/model cache availability is adequate. |
| Irrelevant context appears      | Indexed content is too broad or unrelated.                             | Rebuild the index with narrower material.                                             |
| Text Generation ignores context | Semantic context toggle is disabled or retrieval returns weak matches. | Enable semantic context and ask a more specific question.                             |

## 🔗 Related API Pages

| API Page                              | Purpose                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [App API](../api/app.md)              | Source documentation for chunking, embedding access, cosine similarity, and prompt construction. |
| [Configuration API](../api/config.md) | Runtime settings and help text used by the semantic workflow.                                    |
