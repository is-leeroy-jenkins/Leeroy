###### Leeroy
![](https://github.com/is-leeroy-jenkins/Leeroy/blob/main/resources/images/leeroy_project.png)

---


A small python application designed for text generation, retrieval-augmented 
generation (RAG), and semantic search  via `llama.cpp`.  The LLM itself is a fine-tuned 
variant of Meta's Llama 3.2 1B Instruct, quantized to Q4_K_M GGUF 
format for high-efficiency, and low-latency inference. 

With strong alignment capabilities, multilingual robustness, and support for complex multi-step reasoning,
Leeroy strikes a balance between performance, size, and instruction quality. Designed for use on CPUs 
and modest GPUs, Leeroy runs natively in llama.cpp, LM Studio, Ollama, and GGUF-compatible environments.


## ✨ Key Features

* 🗨️ **Text Generation** (streaming, llama.cpp native)
* 📄 **Retrieval Augmentation (Basic RAG)** via document injection
* 🔍 **Semantic Search** with embeddings (`sentence-transformers`)
* 🧠 **Editable System Instructions**
* 💾 **Persistent Chat History** (SQLite)
* 📤 **Export Conversations** to Markdown and PDF
* 🧮 **Live Token Usage Meter**
* ⚙️ **Dynamic Parameter Controls** (temperature, top-p, top-k, context, etc.)
* 🖥️ **Runs Fully Offline** (CPU-only supported)

---
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://leeroy.streamlit.app/)

![](https://github.com/is-leeroy-jenkins/Leeroy/blob/main/resources/images/leeroy-streamlit.gif)

## 🧱 Architecture Overview

```
┌──────────────────────────────┐
│        Leeroy App            │
│        (Streamlit)           │
│                              │
│  - UI / Tabs                 │
│  - RAG / Semantic Search     │
│  - SQLite Persistence        │
│                              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Leeroy LLM (GGUF)        │
│  Llama-3.2-1B-Instruct       │
│  Q4_K_M Quantized            │
│                              │
│  Loaded via llama.cpp        │
└──────────────────────────────┘
```

The **application and the model are intentionally decoupled**:

* The app lives in GitHub
* The model lives on Hugging Face
* Users control where the model is stored locally



## 📦 Requirements

### Python

* **Python 3.10+** recommended

### System

* CPU-only supported
* No GPU required
* Windows, macOS, Linux supported


## 🔧 Setting Up the Application

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-org>/leeroy.git
cd leeroy
```

---

### 2️⃣ Create a Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Downloading the Leeroy LLM (Required)

The Leeroy application **does not ship with the model**.

Download the file:

👉 [![HuggingFace](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/leeroy-jankins/leeroy)



```
Leeroy-3B-Instruct.Q4_K_M.gguf
```

### Recommended Locations

You may place the model **anywhere on your system**, for example:

* LM Studio model cache
* A shared `llm/` directory
* An external drive

Example (Windows):

```
C:\Users\<you>\source\llm\lmstudio\lmstudio-community\
leeroy-jankins\leeroy\Leeroy-3B-Instruct.Q4_K_M.gguf
```

---

## 🌱 Environment Variable Configuration

The application locates the model via an **environment variable**.

### Required Variable

```
LEEROY_LLM_PATH
```

### Windows (PowerShell)

```powershell
setx LEEROY_LLM_PATH "C:\path\to\Leeroy-3B-Instruct.Q4_K_M.gguf"
```

Restart your terminal after setting it.

### macOS / Linux

```bash
export LEEROY_LLM_PATH=/path/to/Leeroy-3B-Instruct.Q4_K_M.gguf
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧭 Application Tabs

| Tab                        | Description                              |
| -------------------------- | ---------------------------------------- |
| **System Instructions**    | Define global assistant behavior         |
| **Text Generation**        | Chat with Leeroy (streaming)             |
| **Retrieval Augmentation** | Upload documents for basic RAG           |
| **Semantic Search**        | Vector-based retrieval (optional toggle) |
| **Export**                 | Download chat history as Markdown or PDF |

---

## 🧮 Token Usage & Parameters

* **Footer (Left)**: prompt tokens, response tokens, % of context used
* **Footer (Right)**: live model parameters
* Updates automatically after each generation

---

## 📚 About the Leeroy LLM

The underlying model is **Leeroy**, a fine-tuned and quantized variant of Meta’s `Llama-3.2-1B-Instruct`, optimized for **local inference and instruction following**.

Key characteristics (from the model README):

* GGUF `Q4_K_M` quantization
* Optimized for llama.cpp, LM Studio, Ollama
* Strong performance on reasoning, policy, and technical tasks
* Trained on curated regulatory and financial datasets
* Apache-2.0 licensed base model with downstream licensing considerations

For full model details, training data, benchmarks, and usage examples, see the Hugging Face README .

---

## 🔒 Notes on Licensing & Usage

* This repository contains **application code only**
* The model is governed by its **own license on Hugging Face**
* Users are responsible for compliance with:

  * Meta Llama license
  * Dataset terms
  * Organizational policies

---

## 🧰 Troubleshooting

* **Model not found on startup**
  → Verify `LEEROY_LLM_PATH` and restart your terminal

* **Slow generation**
  → Increase CPU threads, reduce context window

* **High memory usage**
  → Lower context (`n_ctx`) or reduce max tokens

---

