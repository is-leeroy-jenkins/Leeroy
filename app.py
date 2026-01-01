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
import pandas as pd
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
# [SNIPPED HEADER + IMPORTS — unchanged from your last version]

# ==============================
# SESSION STATE (authoritative)
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

if "selected_prompt_id" not in st.session_state:
    st.session_state.selected_prompt_id = None

if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

# ==============================
# System Instructions Tab
# ==============================
with tab_system:
    df_prompts = fetch_prompts_df()
    names = [""] + df_prompts["Name"].tolist()

    selected_name = st.selectbox(
        "Load System Prompt",
        names,
        index=0
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear"):
            st.session_state.system_prompt = ""
            st.session_state.selected_prompt_id = None

    with col2:
        if st.button("✏️ Edit") and selected_name:
            record = fetch_prompt_by_name(selected_name)
            if record:
                st.session_state.selected_prompt_id = record["PromptsId"]

    # ONLY load text when user selects a name
    if selected_name:
        record = fetch_prompt_by_name(selected_name)
        if record:
            st.session_state.system_prompt = record["Text"]

    st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=260,
        key="system_prompt_editor"
    )

# ==============================
# Prompt Engineering Tab
# ==============================
with tab_prompt:
    st.subheader("Prompt Engineering")

    df = fetch_prompts_df()

    # Reflect current selection in table
    if st.session_state.selected_prompt_id is not None:
        df["Selected"] = df["PromptsId"] == st.session_state.selected_prompt_id
    else:
        df["Selected"] = False

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=["PromptsId", "Name", "Version", "ID"],
        key="prompt_editor_table"
    )

    # Persist selection
    selected_rows = edited.loc[edited["Selected"]].index.tolist()

    if len(selected_rows) == 1:
        pid = int(edited.loc[selected_rows[0], "PromptsId"])
        st.session_state.selected_prompt_id = pid
    elif len(selected_rows) == 0:
        st.session_state.selected_prompt_id = None
    else:
        st.warning("Select only one prompt.")

    # Load prompt once, cache it
    prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}

    if st.session_state.selected_prompt_id:
        if st.session_state.selected_prompt_id not in st.session_state.prompt_cache:
            record = fetch_prompt_by_id(st.session_state.selected_prompt_id)
            if record:
                st.session_state.prompt_cache[
                    st.session_state.selected_prompt_id
                ] = record
        prompt = st.session_state.prompt_cache.get(
            st.session_state.selected_prompt_id,
            prompt
        )

    # Buttons (right aligned)
    spacer, btns = st.columns([5, 2])
    with btns:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("➕ New", use_container_width=True):
                st.session_state.selected_prompt_id = None
                prompt = {"Name": "", "Text": "", "Version": "", "ID": ""}
        with b2:
            if st.button(
                "🗑 Delete",
                use_container_width=True,
                disabled=st.session_state.selected_prompt_id is None
            ):
                delete_prompt(st.session_state.selected_prompt_id)
                st.session_state.selected_prompt_id = None
                st.session_state.prompt_cache.clear()
                st.rerun()

    st.divider()

    # Editor form
    with st.form("prompt_editor_form"):
        name = st.text_input("Name", prompt["Name"])
        version = st.text_input("Version", prompt["Version"])
        ext_id = st.text_input("ID", prompt["ID"])
        text = st.text_area("Prompt Text", prompt["Text"], height=260)

        if st.form_submit_button("💾 Save"):
            data = {
                "Name": name,
                "Text": text,
                "Version": version,
                "ID": ext_id,
            }
            if st.session_state.selected_prompt_id:
                update_prompt(st.session_state.selected_prompt_id, data)
            else:
                insert_prompt(data)

            st.session_state.prompt_cache.clear()
            st.rerun()
