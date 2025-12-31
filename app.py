"""
******************************************************************************************
Assembly:                Leeroy
Filename:                app.py
Author:                  Terry D. Eppler
Last Modified On:        2025-01-01
******************************************************************************************
"""

import os
import sqlite3
import multiprocessing
import base64
import io
from pathlib import Path
from typing import List
import streamlit as st
from streamlit.components.v1 import html
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# ==============================================================================
# Model Path Resolution (ENV-FIRST)
# ==============================================================================
DEFAULT_MODEL_PATH = "models/Leeroy-3B-Instruct.Q4_K_M.gguf"
HF_MODEL_URL = "https://huggingface.co/leeroy-jankins/leeroy"

MODEL_PATH = os.getenv("LEEROY_LLM_PATH", DEFAULT_MODEL_PATH)
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error(
        "❌ **Leeroy model not found**\n\n"
        f"Expected path:\n`{MODEL_PATH}`\n\n"
        "Download the model and set `LEEROY_LLM_PATH`.\n\n"
        f"{HF_MODEL_URL}"
    )
    st.stop()

# ==============================================================================
# Constants
# ==============================================================================
DB_PATH = "stores/sqlite/leeroy.db"
DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count()

# ==============================================================================
# Streamlit Config
# ==============================================================================
st.set_page_config(
    page_title="Leeroy",
    layout="wide",
    page_icon="resources/images/favicon.ico"
)

# ==============================================================================
# Utilities
# ==============================================================================
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==============================================================================
# Database
# ==============================================================================
def ensure_db():
    Path("stores/sqlite").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                vector BLOB
            )
        """)

def save_message(role, content):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )

def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()

# ==============================================================================
# Loaders
# ==============================================================================
@st.cache_resource
def load_llm(ctx, threads):
    return Llama(
        model_path=str(MODEL_PATH_OBJ),
        n_ctx=ctx,
        n_threads=threads,
        n_batch=512,
        verbose=False
    )

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================================
# Sidebar (Branding + Parameters ONLY)
# ==============================================================================
with st.sidebar:
    logo = image_to_base64("resources/images/leeroy_logo.png")
    st.markdown(
	    f"""
            <img src="data:image/png;base64,{logo}"
                 style="max-height:100px; display:block; margin-left:auto; margin-right:auto;">
            """,
	    unsafe_allow_html=True
    )

    st.header("⚙️ Model Parameters")
    ctx = st.slider("Context Window", 2048, 8192, DEFAULT_CTX, 512)
    threads = st.slider("CPU Threads", 1, CPU_CORES, CPU_CORES)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 20, 5)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.05)

# ==============================================================================
# Init
# ==============================================================================
ensure_db()
llm = load_llm(ctx, threads)
embedder = load_embedder()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Leeroy, a precise and technically accurate assistant."
    )

if "basic_docs" not in st.session_state:
    st.session_state.basic_docs = []

if "use_semantic" not in st.session_state:
    st.session_state.use_semantic = False

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "prompt": 0,
        "response": 0,
        "context_pct": 0.0
    }

# ==============================================================================
# Tabs (RENAMED + REORDERED)
# ==============================================================================
tab_system, tab_chat, tab_basic, tab_semantic, tab_export = st.tabs(
    [
        "System Instructions",
        "Text Generation",
        "Retrieval Augmentation",
        "Semantic Search",
        "Export"
    ]
)

# ==============================================================================
# Prompt Builder
# ==============================================================================
def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT chunk, vector FROM embeddings"
            ).fetchall()

        if rows:
            q_vec = embedder.encode([user_input])[0]
            scored = [
                (chunk, cosine_sim(q_vec, np.frombuffer(vec)))
                for chunk, vec in rows
            ]
            top_chunks = [
                c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
            ]

            prompt += "<|system|>\nSemantic Context:\n"
            for c in top_chunks:
                prompt += f"- {c}\n"
            prompt += "</s>\n"

    if st.session_state.basic_docs:
        prompt += "<|system|>\nDocument Context:\n"
        for d in st.session_state.basic_docs[:6]:
            prompt += f"- {d}\n"
        prompt += "</s>\n"

    for role, content in st.session_state.messages:
        prompt += f"<|{role}|>\n{content}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==============================================================================
# SYSTEM INSTRUCTIONS TAB
# ==============================================================================
with tab_system:
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=220
    )

# ==============================================================================
# TEXT GENERATION TAB
# ==============================================================================
with tab_chat:
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask Leeroy...")

    if user_input:
        save_message("user", user_input)
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)

        with st.chat_message("assistant"):
            response = ""
            for chunk in llm(
                prompt,
                stream=True,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["</s>"]
            ):
                response += chunk["choices"][0]["text"]
                st.markdown(response)

        save_message("assistant", response)
        st.session_state.messages.append(("assistant", response))

        # ---- TOKEN USAGE (SAFE) ----
        prompt_tokens = len(llm.tokenize(prompt.encode()))
        response_tokens = len(llm.tokenize(response.encode()))
        context_pct = (prompt_tokens + response_tokens) / ctx * 100

        st.session_state.token_usage = {
            "prompt": prompt_tokens,
            "response": response_tokens,
            "context_pct": context_pct
        }

# ==============================================================================
# RETRIEVAL AUGMENTATION TAB (BASIC RAG)
# ==============================================================================
with tab_basic:
    uploads = st.file_uploader(
        "Upload TXT / MD / PDF",
        accept_multiple_files=True
    )

    if uploads:
        st.session_state.basic_docs.clear()
        for f in uploads:
            text = f.read().decode(errors="ignore")
            st.session_state.basic_docs.extend(chunk_text(text))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded.")

# ==============================================================================
# SEMANTIC SEARCH TAB
# ==============================================================================
with tab_semantic:
    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context in Text Generation",
        value=st.session_state.use_semantic
    )

    uploads = st.file_uploader(
        "Upload Documents for Semantic Index",
        accept_multiple_files=True
    )

    if uploads:
        chunks = []
        for f in uploads:
            chunks.extend(chunk_text(f.read().decode(errors="ignore")))

        vectors = embedder.encode(chunks)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM embeddings")
            for c, v in zip(chunks, vectors):
                conn.execute(
                    "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                    (c, v.tobytes())
                )

        st.success("Semantic index built.")

# ==============================================================================
# EXPORT TAB
# ==============================================================================
with tab_export:
    history = load_history()
    md = "\n\n".join([f"**{r.upper()}**:\n{c}" for r, c in history])

    st.download_button("Download Markdown", md, "leeroy_chat.md")

    pdf_buf = io.BytesIO()
    pdf = canvas.Canvas(pdf_buf, pagesize=LETTER)
    y = 750

    for r, c in history:
        pdf.drawString(40, y, f"{r.upper()}: {c[:90]}")
        y -= 14
        if y < 50:
            pdf.showPage()
            y = 750

    pdf.save()
    st.download_button("Download PDF", pdf_buf.getvalue(), "leeroy_chat.pdf")

# ==============================================================================
# FOOTER (SAFE — READ-ONLY)
# ==============================================================================
html(
    f"""
    <style>
        .leeroy-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 6px 12px;
            font-size: 0.8rem;
            background-color: rgba(20,20,20,1);
            color: #ddd;
            display: flex;
            justify-content: space-between;
            z-index: 9999;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}
    </style>

    <div class="leeroy-footer">
        <div>
            🧮 Tokens —
            Prompt: {st.session_state.token_usage["prompt"]} |
            Response: {st.session_state.token_usage["response"]} |
            Context Used: {st.session_state.token_usage["context_pct"]:.1f}%
        </div>

        <div>
            ⚙️ ctx={ctx} · temp={temperature} · top_p={top_p} ·
            top_k={top_k} · repeat={repeat_penalty}
        </div>
    </div>
    """,
    height=40
)
