"""
******************************************************************************************
Assembly:                Leeroy
Filename:                app.py
Author:                  Terry D. Eppler
Last Modified On:        2025-01-01
******************************************************************************************
"""

from __future__ import annotations

import base64
import io
import multiprocessing
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from llama_cpp import Llama
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

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
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ==============================================================================
# Database
# ==============================================================================
def ensure_db() -> None:
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
        # Prompts table per your schema (safe to run even if already exists)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Prompts (
                PromptsId INTEGER NOT NULL UNIQUE,
                Name TEXT(80),
                Text TEXT,
                Version TEXT(80),
                ID TEXT(80),
                PRIMARY KEY(PromptsId AUTOINCREMENT)
            )
        """)

def save_message(role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )

def load_history() -> List[Tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT role, content FROM chat_history ORDER BY id"
        ).fetchall()

def clear_history() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_history")

# ==============================================================================
# Prompt Engineering DB helpers
# ==============================================================================
def fetch_prompts_df() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            'SELECT PromptsId, Name, Version, ID FROM Prompts ORDER BY PromptsId DESC',
            conn
        )
    if "Selected" not in df.columns:
        df.insert(0, "Selected", False)
    else:
        df["Selected"] = False
    return df

def fetch_prompt_by_id(pid: int) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            'SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?',
            (pid,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([c[0] for c in cur.description], row))

def fetch_prompt_by_name(name: str) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            'SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE Name=?',
            (name,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([c[0] for c in cur.description], row))

def insert_prompt(data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'INSERT INTO Prompts (Name, Text, Version, ID) VALUES (?, ?, ?, ?)',
            (data.get("Name", ""), data.get("Text", ""), data.get("Version", ""), data.get("ID", ""))
        )
        conn.commit()

def update_prompt(pid: int, data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'UPDATE Prompts SET Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?',
            (data.get("Name", ""), data.get("Text", ""), data.get("Version", ""), data.get("ID", ""), pid)
        )
        conn.commit()

def delete_prompt(pid: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM Prompts WHERE PromptsId=?', (pid,))
        conn.commit()

# ==============================================================================
# Loaders
# ==============================================================================
@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    return Llama(
        model_path=str(MODEL_PATH_OBJ),
        n_ctx=ctx,
        n_threads=threads,
        n_batch=512,
        verbose=False
    )

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================================
# Sidebar (Branding + Parameters ONLY)
# ==============================================================================
with st.sidebar:
    logo = image_to_base64("resources/images/leeroy_logo.png")
    st.markdown(
        f"""
        <img src="data:image/png;base64,{logo}"
             style="max-height:80px; display:block; margin-left:auto; margin-right:auto;">
        """,
        unsafe_allow_html=True
    )

    st.header("⚙️ Mind Controls")
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
    st.session_state.system_prompt = "You are Leeroy, a precise and technically accurate assistant."

if "basic_docs" not in st.session_state:
    st.session_state.basic_docs = []

if "use_semantic" not in st.session_state:
    st.session_state.use_semantic = False

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"prompt": 0, "response": 0, "context_pct": 0.0}

# Prompt Engineering selection state (authoritative)
if "selected_prompt_id" not in st.session_state:
    st.session_state.selected_prompt_id = None

# ==============================================================================
# Tabs
# ==============================================================================
tab_system, tab_chat, tab_basic, tab_semantic, tab_prompt, tab_export = st.tabs(
    [
        "System Instructions",
        "Text Generation",
        "Retrieval Augmentation",
        "Semantic Search",
        "Prompt Engineering",
        "Export"
    ]
)

# ==============================================================================
# Prompt Builder (restores: system + semantic + basic RAG + conversation)
# ==============================================================================
def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    # Semantic context
    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT chunk, vector FROM embeddings").fetchall()

        if rows:
            q_vec = embedder.encode([user_input])[0]
            scored = [(chunk, cosine_sim(q_vec, np.frombuffer(vec))) for chunk, vec in rows]
            top_chunks = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]

            if top_chunks:
                prompt += "<|system|>\nSemantic Context:\n"
                for c in top_chunks:
                    prompt += f"- {c}\n"
                prompt += "</s>\n"

    # Retrieval Augmentation (basic docs)
    if st.session_state.basic_docs:
        prompt += "<|system|>\nDocument Context:\n"
        for d in st.session_state.basic_docs[:6]:
            prompt += f"- {d}\n"
        prompt += "</s>\n"

    # Conversation
    for role, content in st.session_state.messages:
        prompt += f"<|{role}|>\n{content}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==============================================================================
# SYSTEM INSTRUCTIONS TAB (restored + enhanced, no silent overwrite)
# ==============================================================================
with tab_system:
    # Original functionality: editable textarea always present
    st.subheader("System Instructions")

    df_prompts = fetch_prompts_df()
    names = [""] + df_prompts["Name"].fillna("").tolist()

    left, right = st.columns([3, 2])
    with left:
        selected_name = st.selectbox("Load System Prompt", names, index=0, key="sys_prompt_name")
    with right:
        b1, b2, b3 = st.columns(3)
        with b1:
            load_clicked = st.button("Load", use_container_width=True)
        with b2:
            clear_clicked = st.button("Clear", use_container_width=True)
        with b3:
            edit_clicked = st.button("Edit", use_container_width=True)

    if clear_clicked:
        st.session_state.system_prompt = ""
        st.session_state.selected_prompt_id = None

    # Only load on explicit click (prevents overwriting manual edits)
    if load_clicked and selected_name:
        rec = fetch_prompt_by_name(selected_name)
        if rec:
            st.session_state.system_prompt = rec.get("Text", "") or ""
            st.session_state.selected_prompt_id = int(rec["PromptsId"])
        else:
            st.warning("Selected system prompt was not found in the database.")

    # Edit means: select record for Prompt Engineering tab
    if edit_clicked and selected_name:
        rec = fetch_prompt_by_name(selected_name)
        if rec:
            st.session_state.selected_prompt_id = int(rec["PromptsId"])
            st.info("Prompt selected for editing in the Prompt Engineering tab.")

    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=240,
        key="system_prompt_editor"
    )

# ==============================================================================
# TEXT GENERATION TAB (restored + Clear Chat option 2)
# ==============================================================================
with tab_chat:
    st.subheader("Text Generation")

    top_left, top_right = st.columns([6, 1])
    with top_right:
        if st.button("🧹 Clear Chat", use_container_width=True):
            # Clear UI state
            st.session_state.messages = []
            # (Optional) keep DB chat_history? Original saved to DB; we preserve behavior but also
            # make Clear Chat behave like a reset. Clearing DB avoids resurrecting messages.
            clear_history()
            st.session_state.token_usage = {"prompt": 0, "response": 0, "context_pct": 0.0}
            st.rerun()

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask Leeroy.")

    if user_input:
        save_message("user", user_input)
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
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
                placeholder.markdown(response + "▌")

            placeholder.markdown(response)

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

    # Footer token usage (restored)
    st.caption(
        f"Prompt tokens: {st.session_state.token_usage['prompt']} | "
        f"Response tokens: {st.session_state.token_usage['response']} | "
        f"Context used: {st.session_state.token_usage['context_pct']:.2f}%"
    )

# ==============================================================================
# RETRIEVAL AUGMENTATION TAB (BASIC RAG) - restored
# ==============================================================================
with tab_basic:
    st.subheader("Retrieval Augmentation")

    uploads = st.file_uploader("Upload TXT / MD / PDF", accept_multiple_files=True)

    if uploads:
        st.session_state.basic_docs.clear()
        for f in uploads:
            try:
                text = f.read().decode(errors="ignore")
            except Exception:
                text = ""
            st.session_state.basic_docs.extend(chunk_text(text))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded.")

# ==============================================================================
# SEMANTIC SEARCH TAB - restored
# ==============================================================================
with tab_semantic:
    st.subheader("Semantic Search")

    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context in Text Generation",
        value=st.session_state.use_semantic
    )

    uploads = st.file_uploader("Upload Documents for Semantic Index", accept_multiple_files=True)

    if uploads:
        chunks: List[str] = []
        for f in uploads:
            try:
                chunks.extend(chunk_text(f.read().decode(errors="ignore")))
            except Exception:
                continue

        if chunks:
            vectors = embedder.encode(chunks)

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM embeddings")
                for c, v in zip(chunks, vectors):
                    conn.execute(
                        "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                        (c, v.tobytes())
                    )

            st.success("Semantic index built.")
        else:
            st.warning("No readable text was found in the uploaded files.")

# ==============================================================================
# PROMPT ENGINEERING TAB (CRUD + checkbox selection that actually works)
# ==============================================================================
with tab_prompt:
    st.subheader("Prompt Engineering")

    df = fetch_prompts_df()

    # Reflect selected_prompt_id into the table before rendering
    if st.session_state.selected_prompt_id is not None and not df.empty:
        df["Selected"] = df["PromptsId"] == st.session_state.selected_prompt_id
    else:
        df["Selected"] = False

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=["PromptsId", "Name", "Version", "ID"],
        key="prompt_table"
    )

    # Persist selection into session_state (this is what was missing)
    selected_ids = edited.loc[edited["Selected"] == True, "PromptsId"].tolist()

    if len(selected_ids) == 1:
        st.session_state.selected_prompt_id = int(selected_ids[0])
    elif len(selected_ids) == 0:
        st.session_state.selected_prompt_id = None
    else:
        st.warning("Select only one prompt.")
        # keep existing selected_prompt_id unchanged

    # Right-aligned buttons: New then Delete
    spacer, btn_col = st.columns([5, 2])
    with btn_col:
        b_new, b_del = st.columns(2)

        with b_new:
            new_clicked = st.button("+ New", use_container_width=True)

        with b_del:
            del_clicked = st.button(
                "🗑 Delete",
                use_container_width=True,
                disabled=st.session_state.selected_prompt_id is None
            )

    if del_clicked and st.session_state.selected_prompt_id is not None:
        delete_prompt(st.session_state.selected_prompt_id)
        st.session_state.selected_prompt_id = None
        st.rerun()

    # Load selected record (if any) for editing
    selected_prompt: Dict[str, Any] = {"Name": "", "Text": "", "Version": "", "ID": ""}
    if not new_clicked and st.session_state.selected_prompt_id is not None:
        rec = fetch_prompt_by_id(st.session_state.selected_prompt_id)
        if rec:
            selected_prompt = rec
        else:
            # The record truly does not exist; clear selection
            st.session_state.selected_prompt_id = None
            selected_prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}

    st.divider()

    # Two-column controls + edit form below (as requested)
    left, right = st.columns(2)
    with left:
        name = st.text_input("Name", value=str(selected_prompt.get("Name", "") or ""))
        version = st.text_input("Version", value=str(selected_prompt.get("Version", "") or ""))
    with right:
        ext_id = st.text_input("ID", value=str(selected_prompt.get("ID", "") or ""))

    text = st.text_area(
        "Prompt Text",
        value=str(selected_prompt.get("Text", "") or ""),
        height=260
    )

    save_clicked = st.button("💾 Save")

    if save_clicked:
        data = {"Name": name, "Text": text, "Version": version, "ID": ext_id}
        if st.session_state.selected_prompt_id is None or new_clicked:
            insert_prompt(data)
        else:
            update_prompt(st.session_state.selected_prompt_id, data)
        st.rerun()

# ==============================================================================
# EXPORT TAB (restored)
# ==============================================================================
with tab_export:
    st.subheader("Export")

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
