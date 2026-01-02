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
import re
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer

# ==============================================================================
# Model Path Resolution
# ==============================================================================
DEFAULT_MODEL_PATH = "models/Leeroy-3B-Instruct.Q4_K_M.gguf"
MODEL_PATH = os.getenv("LEEROY_LLM_PATH", DEFAULT_MODEL_PATH)
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

# ==============================================================================
# Constants
# ==============================================================================
DB_PATH = "stores/sqlite/leeroy.db"
DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count()

XML_BLOCK_PATTERN: re.Pattern[str] = re.compile(
    r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>",
    re.DOTALL )

MARKDOWN_HEADING_PATTERN: re.Pattern[str] = re.compile(
    r"^##\s+(?P<title>.+?)\s*$")

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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def xml_converter(text: str) -> str:
    """
	    Purpose:
	        Convert XML-delimited prompt text into Markdown by treating XML-like
	        tags as section delimiters, not as strict XML.
	
	    Parameters:
	        text (str):
	            Prompt text containing XML-like opening and closing tags.
	
	    Returns:
	        str:
	            Markdown-formatted text using level-2 headings (##).
    """

    markdown_blocks: List[str] = []

    for match in XML_BLOCK_PATTERN.finditer(text):
        raw_tag: str = match.group("tag")
        body: str = match.group("body").strip()

        # Humanize tag name for Markdown heading
        heading: str = raw_tag.replace("_", " ").replace("-", " ").title()

        markdown_blocks.append(f"## {heading}")
        if body:
            markdown_blocks.append(body)

    return "\n\n".join(markdown_blocks)

def markdown_converter( markdown: str ) -> str:
    """
    
	    Purpose:
	        Convert Markdown-formatted system instructions back into
	        XML-delimited prompt text by treating level-2 headings (##)
	        as section delimiters.
	
	    Parameters:
	        markdown (str):
	            Markdown text using '##' headings to indicate sections.
	
	    Returns:
	        str:
	            XML-delimited text suitable for storage in the prompt database.
	            
    """

    lines: List[str] = markdown.splitlines( )
    output: List[str] = []

    current_tag: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        """
        Emit the currently buffered section as an XML-delimited block.
        """
        nonlocal current_tag, buffer

        if current_tag is None:
            return

        body: str = "\n".join(buffer).strip()

        output.append(f"<{current_tag}>")
        if body:
            output.append(body)
        output.append(f"</{current_tag}>")
        output.append("")

        buffer.clear()

    for line in lines:
        match = MARKDOWN_HEADING_PATTERN.match(line)

        if match:
            flush()

            title: str = match.group("title")
            current_tag = (
                title.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
        else:
            if current_tag is not None:
                buffer.append(line)

    flush()

    # Remove trailing whitespace blocks
    while output and not output[-1].strip():
        output.pop()

    return "\n".join(output)

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
# Prompt DB helpers
# ==============================================================================
def fetch_prompts_df() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT PromptsId, Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
            conn
        )
    df.insert(0, "Selected", False)
    return df

def fetch_prompt_by_id(pid: int) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
            (pid,)
        )
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

def fetch_prompt_by_name(name: str) -> Dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT PromptsId, Name, Text, Version, ID FROM Prompts WHERE Name=?",
            (name,)
        )
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

def insert_prompt(data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO Prompts (Name, Text, Version, ID) VALUES (?, ?, ?, ?)",
            (data["Name"], data["Text"], data["Version"], data["ID"])
        )

def update_prompt(pid: int, data: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE Prompts SET Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
            (data["Name"], data["Text"], data["Version"], data["ID"], pid)
        )

def delete_prompt(pid: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM Prompts WHERE PromptsId=?", (pid,))

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
# Sidebar (Model Parameters)
# Purpose:
#     Collect runtime model parameters used to configure llama.cpp initialization and inference.
# Parameters:
#     None
# Returns:
#     None
# ==============================================================================
with st.sidebar:
    try:
        logo = image_to_base64("resources/images/leeroy_logo.png")
        st.markdown(
            f"<img src='data:image/png;base64,{logo}' "
            f"style='max-height:55px; display:block; margin:auto;'>",
            unsafe_allow_html=True
        )
    except Exception:
        st.write("Bro")

    st.header("⚙️ Mind Controls")

    # --------------------------------------------------------------------------
    # Model initialization parameters
    # --------------------------------------------------------------------------
    ctx: int = st.slider(
	    
        "Context Window",
        min_value=2048,
        max_value=8192,
        value=DEFAULT_CTX,
        step=512,
        help=(
            "Maximum number of tokens the model can consider at once, including system instructions, "
            "history, and context."
        ),
    )

    threads: int = st.slider(
        "CPU Threads",
        min_value=1,
        max_value=CPU_CORES,
        value=CPU_CORES,
        step=1,
        help=(
            "Number of CPU threads used for inference; higher values improve speed but increase CPU "
            "usage."
        ),
    )

    # --------------------------------------------------------------------------
    # Inference parameters (already present + recommended additions)
    # --------------------------------------------------------------------------
    max_tokens: int = st.slider(
        "Max Tokens",
        min_value=128,
        max_value=4096,
        value=1024,
        step=128,
        help="Maximum number of tokens generated per response.",
    )

    temperature: float = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help=(
            "Controls randomness in generation; lower values are more deterministic, higher values "
            "increase creativity."
        ),
    )

    top_p: float = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help=(
            "Nucleus sampling threshold; limits token selection to the smallest set whose cumulative "
            "probability exceeds this value."
        ),
    )

    top_k: int = st.slider(
        "Top-k",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Limits token selection to the top K most probable tokens at each step.",
    )

    repeat_penalty: float = st.slider(
        "Repeat Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.05,
        help="Penalizes repeated tokens to reduce looping and redundant responses.",
    )

    # --------------------------------------------------------------------------
    # Recommended additions
    # --------------------------------------------------------------------------
    repeat_last_n: int = st.slider(
        "Repeat Window",
        min_value=0,
        max_value=1024,
        value=64,
        step=16,
        help="Number of recent tokens considered for repetition penalties; 0 disables the window.",
    )

    presence_penalty: float = st.slider(
        "Presence Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.05,
        help="Encourages introducing new topics by penalizing tokens already present in the context.",
    )

    frequency_penalty: float = st.slider(
        "Frequency Penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.05,
        help="Reduces repeated phrasing by penalizing tokens based on how often they appear.",
    )

    seed: int = st.number_input(
        "Random Seed",
        value=-1,
        step=1,
        help="Set to a fixed value for reproducible outputs; use -1 for a random seed each run.",
    )


# ==============================================================================
# Init
# ==============================================================================
ensure_db()
llm = load_llm(ctx, threads)
embedder = load_embedder()

st.session_state.setdefault("messages", load_history())
st.session_state.setdefault("system_prompt", "")
st.session_state.setdefault("basic_docs", [])
st.session_state.setdefault("use_semantic", False)
st.session_state.setdefault("selected_prompt_id", None)
st.session_state.setdefault("pending_system_prompt_name", None)

# ==============================================================================
# Tabs
# ==============================================================================
tab_system, tab_chat, tab_basic, tab_semantic, tab_prompt, tab_export = st.tabs(
    ["System Instructions", "Text Generation", "Retrieval Augmentation",
     "Semantic Search", "Prompt Engineering", "Export"]
)

# ==============================================================================
# Prompt Builder
# ==============================================================================
def build_prompt(user_input: str) -> str:
    prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"

    if st.session_state.use_semantic:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT chunk, vector FROM embeddings").fetchall()
        if rows:
            q = embedder.encode([user_input])[0]
            scored = [(c, cosine_sim(q, np.frombuffer(v))) for c, v in rows]
            for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
                prompt += f"<|system|>\n{c}\n</s>\n"

    for d in st.session_state.basic_docs[:6]:
        prompt += f"<|system|>\n{d}\n</s>\n"

    for r, c in st.session_state.messages:
        prompt += f"<|{r}|>\n{c}\n</s>\n"

    prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
    return prompt

# ==============================================================================
# System Instructions Tab (DROP-IN SAFE, LIFECYCLE-CORRECT)
# ==============================================================================
with tab_system:
    st.subheader("System Instructions")

    # ------------------------------------------------------------------
    # Prompt selector
    # ------------------------------------------------------------------
    df_prompts = fetch_prompts_df()
    prompt_names = [""] + df_prompts["Name"].tolist()

    selected_name: str = st.selectbox(
        "Load System Prompt",
        prompt_names,
        key="system_prompt_selector"
    )

    st.session_state.pending_system_prompt_name = (
        selected_name if selected_name else None
    )

    # ------------------------------------------------------------------
    # Action buttons (single row, audited)
    # ------------------------------------------------------------------
    col_load, col_clear, col_edit, col_spacer, col_xml_md, col_md_xml = st.columns(
        [1, 1, 1, 0.5, 1.5, 1.5]
    )

    with col_load:
        load_clicked: bool = st.button(
            "Load",
            disabled=st.session_state.pending_system_prompt_name is None
        )

    with col_clear:
        clear_clicked: bool = st.button("Clear")

    with col_edit:
        edit_clicked: bool = st.button(
            "Edit",
            disabled=st.session_state.pending_system_prompt_name is None
        )

    with col_spacer:
        st.write("")

    with col_xml_md:
        to_markdown_clicked: bool = st.button(
            "XML → Markdown",
            help="Replace XML-like delimiters with Markdown headings (##)."
        )

    with col_md_xml:
        to_xml_clicked: bool = st.button(
            "Markdown → XML",
            help="Replace Markdown headings (##) with XML-like delimiters."
        )

    # ------------------------------------------------------------------
    # Button behaviors (audited: no missing logic)
    # ------------------------------------------------------------------
    if load_clicked:
        record = fetch_prompt_by_name(st.session_state.pending_system_prompt_name)
        if record:
            st.session_state.system_prompt = record["Text"]
            st.session_state.selected_prompt_id = record["PromptsId"]

    if clear_clicked:
        st.session_state.system_prompt = ""
        st.session_state.selected_prompt_id = None

    if edit_clicked:
        record = fetch_prompt_by_name(st.session_state.pending_system_prompt_name)
        if record:
            st.session_state.selected_prompt_id = record["PromptsId"]

    if to_markdown_clicked:
        try:
            st.session_state.system_prompt = xml_converter(
                st.session_state.system_prompt
            )
            st.success("Converted to Markdown.")
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")

    if to_xml_clicked:
        try:
            st.session_state.system_prompt = markdown_converter(
                st.session_state.system_prompt
            )
            st.success("Converted to XML-delimited format.")
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")

    # ------------------------------------------------------------------
    # System prompt editor
    # ------------------------------------------------------------------
    st.text_area(
        "System Prompt",
        key="system_prompt",
        height=260,
        help=(
            "Edit system instructions here. "
            "Use XML-like tags or Markdown headings (##). "
            "Conversion tools are provided above."
        )
    )


# ==============================================================================
# Text Generation Tab
# ==============================================================================
with tab_chat:
    if st.button("🧹 Clear Chat"):
        clear_history()
        st.session_state.messages = []
        st.rerun()

    for r, c in st.session_state.messages:
        with st.chat_message(r):
            st.markdown(c)

    user_input = st.chat_input("Ask Leeroy…")
    if user_input:
        save_message("user", user_input)
        st.session_state.messages.append(("user", user_input))

        prompt = build_prompt(user_input)
        with st.chat_message("assistant"):
            out, buf = st.empty(), ""
            for chunk in llm(prompt, stream=True, max_tokens=1024,
                              temperature=temperature, top_p=top_p,
                              repeat_penalty=repeat_penalty, stop=["</s>"]):
                buf += chunk["choices"][0]["text"]
                out.markdown(buf + "▌")
            out.markdown(buf)

        save_message("assistant", buf)
        st.session_state.messages.append(("assistant", buf))

# ==============================================================================
# Retrieval Augmentation Tab
# ==============================================================================
with tab_basic:
    files = st.file_uploader("Upload documents", accept_multiple_files=True)
    if files:
        st.session_state.basic_docs.clear()
        for f in files:
            st.session_state.basic_docs.extend(chunk_text(f.read().decode(errors="ignore")))
        st.success(f"{len(st.session_state.basic_docs)} chunks loaded")

# ==============================================================================
# Semantic Search Tab
# ==============================================================================
with tab_semantic:
    st.session_state.use_semantic = st.checkbox(
        "Use Semantic Context", st.session_state.use_semantic
    )
    files = st.file_uploader("Upload for embedding", accept_multiple_files=True)
    if files:
        chunks = []
        for f in files:
            chunks.extend(chunk_text(f.read().decode(errors="ignore")))
        vecs = embedder.encode(chunks)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM embeddings")
            for c, v in zip(chunks, vecs):
                conn.execute(
                    "INSERT INTO embeddings (chunk, vector) VALUES (?, ?)",
                    (c, v.tobytes())
                )
        st.success("Semantic index built")

# ==============================================================================
# Prompt Engineering Tab
# ==============================================================================
with tab_prompt:
    df = fetch_prompts_df()
    if st.session_state.selected_prompt_id:
        df["Selected"] = df["PromptsId"] == st.session_state.selected_prompt_id

    edited = st.data_editor(df, hide_index=True, use_container_width=True)
    sel = edited.loc[edited["Selected"], "PromptsId"].tolist()
    st.session_state.selected_prompt_id = sel[0] if len(sel) == 1 else None

    prompt = fetch_prompt_by_id(st.session_state.selected_prompt_id) or \
             {"Name": "", "Text": "", "Version": "", "ID": ""}

    c1, c2 = st.columns(2)
    with c1:
        if st.button("+ New"):
            st.session_state.selected_prompt_id = None
            prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}
    with c2:
        if st.button("🗑 Delete", disabled=st.session_state.selected_prompt_id is None):
            delete_prompt(st.session_state.selected_prompt_id)
            st.session_state.selected_prompt_id = None
            st.rerun()

    name = st.text_input("Name", prompt["Name"])
    version = st.text_input("Version", prompt["Version"])
    pid = st.text_input("ID", prompt["ID"])
    text = st.text_area("Prompt Text", prompt["Text"], height=260)

    if st.button("💾 Save"):
        data = {"Name": name, "Text": text, "Version": version, "ID": pid}
        if st.session_state.selected_prompt_id:
            update_prompt(st.session_state.selected_prompt_id, data)
        else:
            insert_prompt(data)
        st.rerun()

# ==============================================================================
# Export Tab
# ==============================================================================
with tab_export:
    st.subheader("Export")

    # ------------------------------------------------------------
    # Prompt export (System Instructions)
    # ------------------------------------------------------------
    st.markdown("### System Instructions")

    export_format = st.radio(
        "Export format",
        options=["XML-delimited", "Markdown"],
        horizontal=True,
        help="Choose how system instructions should be exported."
    )

    prompt_text: str = st.session_state.get("system_prompt", "")

    if export_format == "Markdown":
        try:
            export_text: str = xml_converter(prompt_text)
            export_filename: str = "leeroy_system_instructions.md"
        except Exception as exc:
            st.error(f"Markdown conversion failed: {exc}")
            export_text = ""
            export_filename = ""
    else:
        export_text = prompt_text
        export_filename = "leeroy_system_instructions.xml"

    st.download_button(
        label="Download System Instructions",
        data=export_text,
        file_name=export_filename,
        mime="text/plain",
        disabled=not bool(export_text.strip())
    )

    # ------------------------------------------------------------
    # Existing chat history export (UNCHANGED)
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Chat History")

    hist = load_history()
    md_history = "\n\n".join(
        [f"**{role.upper()}**\n{content}" for role, content in hist]
    )

    st.download_button(
        "Download Chat History (Markdown)",
        md_history,
        "leeroy_chat.md",
        mime="text/markdown"
    )

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=LETTER)
    y = 750

    for role, content in hist:
        pdf.drawString(40, y, f"{role.upper()}: {content[:90]}")
        y -= 14
        if y < 50:
            pdf.showPage()
            y = 750

    pdf.save()

    st.download_button(
        "Download Chat History (PDF)",
        buf.getvalue(),
        "leeroy_chat.pdf",
        mime="application/pdf"
    )

