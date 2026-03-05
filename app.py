'''
	******************************************************************************************
	    Assembly:                Leeroy
	    Filename:                app.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2024
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">
	
	           Leeroy is a data analysis tool integrating various Generative GPT, Text-Processing, and
	           Machine-Learning algorithms for federal analysts.
	           Copyright ©  2022  Terry Eppler
	
	   Permission is hereby granted, free of charge, to any person obtaining a copy
	   of this software and associated documentation files (the “Software”),
	   to deal in the Software without restriction,
	   including without limitation the rights to use,
	   copy, modify, merge, publish, distribute, sublicense,
	   and/or sell copies of the Software,
	   and to permit persons to whom the Software is furnished to do so,
	   subject to the following conditions:
	
	   The above copyright notice and this permission notice shall be included in all
	   copies or substantial portions of the Software.
	
	   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	   FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	   DEALINGS IN THE SOFTWARE.
	
	   You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
	  app.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

import base64
import config as cfg
import io
import multiprocessing
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
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
DEFAULT_MODEL_PATH = 'models/Leeroy-3B-Instruct.Q4_K_M.gguf'
MODEL_PATH = os.getenv( 'LEEROY_LLM_PATH', DEFAULT_MODEL_PATH)
MODEL_PATH_OBJ = Path(MODEL_PATH)

if not MODEL_PATH_OBJ.exists():
    st.error( f'Model not found at {MODEL_PATH}' )
    st.stop( )

# ==============================================================================
# CONSTANTS
# ==============================================================================
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
DB_PATH = 'stores/sqlite/leeroy.db'
DEFAULT_CTX = 4096
CPU_CORES = multiprocessing.cpu_count( )
FAVICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/leeroy_logo.ico'
XML_BLOCK_PATTERN = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>", re.DOTALL )
MARKDOWN_HEADING_PATTERN = re.compile( r"^##\s+(?P<title>.+?)\s*$" )
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"
TABS = [ 'System Instructions', 'Text Generation', 'Retrieval Augmentation',
         'Semantic Search', 'Prompt Engineering', 'Export' ]

# ==============================================================================
# UTILITIES
# ==============================================================================
def image_to_base64( path: str ) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, size: int=1200, overlap: int=200 ) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def cosine_sim( a: np.ndarray, b: np.ndarray ) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float( np.dot(a, b) / denom ) if denom else 0.0

def ensure_db( ) -> None:
    Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS chat_history ( id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                               role TEXT,
                                                               content TEXT  ) """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                vector BLOB ) """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Prompts (
                PromptsId INTEGER NOT NULL UNIQUE,
                Name TEXT(80),
                Text TEXT,
                Version TEXT(80),
                ID TEXT(80),
                PRIMARY KEY(PromptsId AUTOINCREMENT) ) """)

# -------- Chat & Text --------------------

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def markdown_converter( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )
	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

#-------- Prompt Engineering ----------------
def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )

def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Caption" ], data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ],
			 pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	prompt = f"<|system|>\n{st.session_state.system_prompt}\n</s>\n"
	
	if st.session_state.use_semantic:
		with sqlite3.connect( DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_sim( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in st.session_state.basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	for r, c in st.session_state.messages:
		prompt += f"<|{r}|>\n{c}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

# -------------- LLM -------------------
@st.cache_resource
def load_llm(ctx: int, threads: int) -> Llama:
    return Llama(  model_path=str( MODEL_PATH_OBJ ),  n_ctx=ctx,  n_threads=threads,
        n_batch=512,  verbose=False )

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'mode' not in st.session_state:
	st.session_state[ 'mode' ] = ''

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = ''

if 'system_prompt' not in st.session_state:
	st.session_state[ 'system_prompt' ] = ''

if 'context_window' not in st.session_state:
	st.session_state[ 'context_window' ] = 0

if 'cpu_threads' not in st.session_state:
	st.session_state[ 'cpu_threads' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'top_k' not in st.session_state:
	st.session_state[ 'top_k' ] = 0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presence_penalty' ] = 0.0

if 'repeat_penalty' not in st.session_state:
	st.session_state[ 'repeat_penalty' ] = ''

if 'repeat_window' not in st.session_state:
	st.session_state[ 'repeat_window' ] = 0

if 'random_seed' not in st.session_state:
	st.session_state[ 'random_seed' ] = 0

if 'basic_docs' not in st.session_state:
	st.session_state[ 'basic_docs' ] = [ ]

if 'use_semantic' not in st.session_state:
	st.session_state[ 'use_semantic' ] = False

if 'selected_prompt_id' not in st.session_state:
	st.session_state[ 'selected_prompt_id' ] = ''

if 'pending_system_prompt_name' not in st.session_state:
	st.session_state[ 'pending_system_prompt_name' ] = ''

ensure_db( )
embedder = load_embedder( )
st.session_state.setdefault( 'messages', load_history( ) )
st.session_state.setdefault( 'system_prompt', "" )
st.set_page_config( page_title='Leeroy', layout='wide', page_icon=cfg.FAVICON )

# ==============================================================================
# TABS
# ==============================================================================
instruction_tab, chat_tab, retrieval_tab, semantic_tab, prompt_tab, export_tab = st.tabs( TABS )
MODES = [ 'Text', 'Retrieval',
          'Semantic', 'Prompt', 'Data' ]
# ==============================================================================
# Sidebar (Model Parameters)
# ==============================================================================
with st.sidebar:
	style_subheaders( )
	st.logo( cfg.LOGO, size='large' )
	
	c1, c2 = st.columns( [ 0.1, 0.9] )
	with c2:
		st.subheader( '⚙️ Application Mode' )
		mode = st.radio( label='', options=cfg.MODES, index=0 )
	
	st.divider( )
	

# ==============================================================================
# Text Generation Mode
# ==============================================================================
if mode == 'Text Generation':
	st.subheader( "💬 Chat Completions", help=cfg.CHAT_COMPLETIONS )
	st.divider( )
	max_tokens = st.session_state.get( 'max_tokens', 0 )
	top_percent = st.session_state.get( 'top_percent', 0.0 )
	top_k = st.session_state.get( 'top_k', 0 )
	frequency_penalty = st.session_state.get( 'frequency_penalty', 0.0 )
	presense_penalty = st.session_state.get( 'presense_penalty', 0.0 )
	temperature = st.session_state.get( 'temperature', 0.0 )
	repeat_penalty = st.session_state.get( 'repeat_penalty', 0.0 )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='🧠 Mind Controls', expanded=False ):
			mind_c1, mind_c2, mind_c3 = st.columns( [ .33, .33, .33 ], border=True, gap='medium' )
			
			# ------------- Temperature ----------
			with mind_c1:
				set_temperature = st.slider( label='Temperature', min_value=0.1, max_value=1.5,
					value=0.7, step=0.1, help=cfg.TEMPERATURE )
			
				temperature = st.session_state[ 'temperature' ]
				
			# ------------- Top-P ----------
			with mind_c2:
				set_top_p = st.slider( label='Top-P', min_value=0.1, max_value=1.0,
					value=0.9, step=0.05, help=cfg.TOP_P )
				
				top_percent = st.session_state[ 'top_percent' ]
			
			# ------------- Top-K ----------
			with mind_c3:
				set_top_k = st.slider( label='Top-K', min_value=1, max_value=50, step=1,
					key='top_k', help=cfg.TOP_K )
				
				top_k = st.session_state[ 'top_k' ]
				
			# ------------- Reset Settings ----------
			if st.button( label='Reset', key='mind_controls_reset', width='stretch' ):
				for key in [ 'top_k', 'top_p', 'temperature' ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Probability Controls
		# ------------------------------------------------------------------
		with st.expander( label='🧠 Probability Controls', expanded=False ):
			prob_c1, prob_c2, prob_c3, prob_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
				border=True, gap='medium' )
			
			# ------------- Repeat Window ----------
			with prob_c1:
				set_repeat_last_n = st.slider( label='Repeat Window', min_value=0,
					max_value=1024, key='repeat_window', step=16, help=cfg.REPEAT_WINDOW )
				
				repeat_window = st.session_state[ 'repeat_window' ]
			
			# ------------- Repeat Penalty ----------
			with prob_c2:
				set_repeat_penalty = st.slider( label='Repeat Penalty', min_value=1.0, max_value=2.0,
					key='repeat_penalty', step=0.05, help=cfg.REPEAT_PENALTY )
				
				repeat_penalty = st.session_state[ 'repeat_penalty' ]
			
			# ------------- Presense Penalty ----------
			with prob_c3:
				set_presence_penalty = st.slider( label='Presence Penalty', min_value=0.0, max_value=2.0,
					key='presense_penalty', step=0.05, help=cfg.PRESENCE_PENALTY )
				
				presense_penalty = st.session_state[ 'presense_penalty' ]
			
			# ------------- Frequency Penalty ----------
			with prob_c4:
				set_frequency_penalty = st.slider( label='Frequency Penalty', min_value=0.0, max_value=2.0,
					key='frequency_penalty', step=0.05, help=cfg.FREQUENCY_PENALTY )
				
				frequency_penalty = st.session_state[ 'frequency_penalty' ]
			
			# ------------- Reset Settings ----------
			if st.button( label='Reset', key='probability_controls_reset', width='stretch' ):
				for key in [ 'frequency_penalty', 'presense_penalty',
				             'temperature', 'repeat_penalty', 'repeat_window' ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — Context Controls
		# ------------------------------------------------------------------
		with st.expander( label=' Context Controls', expanded=False ):
			ctx_c1, ctx_c2, ctx_c3, ctx_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
				border=True, gap='medium' )
			
			# ------------- Context Window ----------
			with ctx_c1:
				set_ctx = st.slider( label='Context Window', min_value=2048, max_value=8192,
					key='context_window', step=512, help=cfg.CONTEXT_WINDOW )
				
				ctx = st.session_state[ 'context_window' ]
			
			# ------------- CPU Threads ----------
			with ctx_c2:
				set_threads = st.slider( label='CPU Threads', min_value=1, max_value=CPU_CORES,
					key='cpu_threads', step=1, help=cfg.CPU_CORES, )
				
				threads = st.session_state[ 'cpu_threads' ]
			
			# ------------- Max Tokens ----------
			with ctx_c3:
				set_max_tokens = st.slider( label='Max Tokens', min_value=128, max_value=4096, step=128,
					key='max_tokens', help=cfg.MAX_TOKENS, )
			
			# ------------- Random Seed ----------
			with ctx_c4:
				set_seed = st.slider( label="Random Seed", min_value=0, max_value=4096, step=1,
					key='random_seed', help=cfg.SEED )
			
			# ------------- Reset Settings ----------
			if st.button( label='Reset', key='context_controls_reset', width='stretch' ):
				for key in [ 'random_seed', 'max_tokens', 'cpu_threads', 'context_window' ]:
					if key in st.session_state:
						del st.session_state[ key ]
				
				st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			ins_left, ins_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			with ins_left:
				st.text_area( label='Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'system_instructions' ] = text
			
			with ins_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( label='Clear Instructions', width='stretch', on_click=_on_clear )
		
		llm = load_llm( ctx, threads )
		for r, c in st.session_state.messages:
			with st.chat_message( r ):
				st.markdown( c )
			user_input = st.chat_input( 'Ask Leeroy…' )
			if user_input:
				save_message( 'user', user_input )
				st.session_state.messages.append( ('user', user_input) )
				prompt = build_prompt( user_input )
				with st.chat_message( 'assistant' ):
					out, buf = st.empty( ), ''
					for chunk in llm( prompt, stream=True, max_tokens=1024,
							temperature=temperature, top_p=top_percent,
							repeat_penalty=repeat_penalty, stop=[ '</s>' ] ):
						buf += chunk[ 'choices' ][ 0 ][ 'text' ]
						out.markdown( buf + '▌' )
					out.markdown( buf )
				
				save_message( 'assistant', buf )
				st.session_state.messages.append( ('assistant', buf) )
			
		if st.button( '🧹 Clear Chat' ):
			clear_history( )
			st.session_state.messages = [ ]
			st.rerun( )
		
# ==============================================================================
# RETRIEVAL AUGMENTATION
# ==============================================================================
elif mode == 'Retrieval Augmentation':
	files = st.file_uploader( 'Upload documents', accept_multiple_files=True )
	if files:
		st.session_state.basic_docs.clear( )
		for f in files:
			st.session_state.basic_docs.extend( chunk_text( f.read( ).decode( errors='ignore' ) ) )
		st.success( f'{len( st.session_state.basic_docs )} chunks loaded' )

# ==============================================================================
# SEMANTIC SEARCH
# ==============================================================================
elif mode == 'Semantic Search':
	st.session_state.use_semantic = st.checkbox( 'Use Semantic Context',
		st.session_state.use_semantic )
	files = st.file_uploader( 'Upload for embedding', accept_multiple_files=True )
	if files:
		chunks = [ ]
		for f in files:
			chunks.extend( chunk_text( f.read( ).decode( errors='ignore' ) ) )
		vecs = embedder.encode( chunks )
		with sqlite3.connect( DB_PATH ) as conn:
			conn.execute( 'DELETE FROM embeddings' )
			for c, v in zip( chunks, vecs ):
				conn.execute(
					'INSERT INTO embeddings (chunk, vector) VALUES (?, ?)',
					(c, v.tobytes( ))
				)
		st.success( 'Semantic index built' )

# ==============================================================================
# PROMPT ENGINEERING MODE
# ==============================================================================
elif mode == 'Prompt Engineering':
    df = fetch_prompts_df()
    if st.session_state.selected_prompt_id:
        df['Selected'] = df['PromptsId'] == st.session_state.selected_prompt_id

    edited = st.data_editor(df, hide_index=True, use_container_width=True)
    sel = edited.loc[edited['Selected'], 'PromptsId'].tolist()
    st.session_state.selected_prompt_id = sel[0] if len(sel) == 1 else None

    prompt = fetch_prompt_by_id(st.session_state.selected_prompt_id) or \
             {'Name': '', 'Text': '', 'Version': '', 'ID': ''}

    c1, c2 = st.columns( 2 )
    with c1:
        if st.button( '+ New' ):
            st.session_state.selected_prompt_id = None
            prompt = {'Name': '', 'Text': '', 'Version': '', 'ID': ''}
    with c2:
        if st.button("🗑 Delete", disabled=st.session_state.selected_prompt_id is None):
            delete_prompt(st.session_state.selected_prompt_id)
            st.session_state.selected_prompt_id = None
            st.rerun()
    
    name = st.text_input( 'Name', prompt[ 'Name' ] )
    version = st.text_input( 'Version', prompt[ 'Version' ] )
    pid = st.text_input( 'ID', prompt[ 'ID' ] )
    text = st.text_area( 'Prompt Text', prompt[ 'Text' ], height=260 )
    
    if st.button( '💾 Save' ):
	    data = { 'Name': name, 'Text': text, 'Version': version, 'ID': pid }
	    if st.session_state.selected_prompt_id:
		    update_prompt( st.session_state.selected_prompt_id, data )
	    else:
		    insert_prompt( data )
	    st.rerun( )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
    st.subheader( 'Export' )
    st.markdown( '### System Instructions' )
    export_format = st.radio( 'Export format',options=[ 'XML-delimited', 'Markdown' ], 
	    horizontal=True, help='Choose how system instructions should be exported.' )

    prompt_text: str = st.session_state.get( 'system_prompt', '' )
    if export_format == 'Markdown':
        try:
            export_text = convert_xml(prompt_text)
            export_filename: str = 'leeroy_system_instructions.md'
        except Exception as exc:
            st.error( f'Markdown conversion failed: {exc}' )
            export_text = ''
            export_filename = ''
    else:
        export_text = prompt_text
        export_filename = 'leeroy_system_instructions.xml'

    st.download_button( label='Download System Instructions', data=export_text,
        file_name=export_filename, mime='text/plain', disabled=not bool( export_text.strip( ) ) )

    # ------------------------------------------------------------
    # Existing chat history export (UNCHANGED)
    # ------------------------------------------------------------
    st.markdown( '---' )
    st.markdown( '### Chat History' )

    hist = load_history()
    md_history = '\n\n'.join( [ f'**{role.upper()}**\n{content}' for role, content in hist ] )

    st.download_button( 'Download Chat History (Markdown)', md_history, 'leeroy_chat.md',
	    mime='text/markdown' )

    buf = io.BytesIO( )
    pdf = canvas.Canvas( buf, pagesize=LETTER )
    y = 750
    for role, content in hist:
        pdf.drawString( 40, y, f'{role.upper( )}: {content[ :90 ]}' )
        y -= 14
        if y < 50:
            pdf.showPage( )
            y = 750

    pdf.save( )

    st.download_button( 'Download Chat History (PDF)',  buf.getvalue(), ''"leeroy_chat.pdf",
        mime='application/pdf' )

# ======================================================================================
# FOOTER — SECTION
# ======================================================================================
st.markdown(
	"""
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

# ---- Fixed Container
st.markdown(
	"""
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(17, 17, 17, 0.95);
		border-top: 1px solid #2a2a2a;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #35618c;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================

mode_val = mode or '—'
active_mode = st.session_state.get( 'mode', None )
right_parts = [ ]
if active_mode is not None:
	right_parts.append( f'Model: {active_mode}' )
right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendered Variables
if mode == 'Text':
	temperature = st.session_state.get( 'text_temperature' )
	top_p = st.session_state.get( 'text_top_percent' )
	freq = st.session_state.get( 'text_frequency_penalty' )
	presence = st.session_state.get( 'text_presense_penalty' )
	number = st.session_state.get( 'text_number' )
	stream = st.session_state.get( 'text_stream' )
	parallel_tools = st.session_state.get( 'text_parallel_tools' )
	max_calls = st.session_state.get( 'text_max_tools' )
	store = st.session_state.get( 'text_store' )
	tools = st.session_state.get( 'text_tools' )
	include = st.session_state.get( 'text_include' )
	domains = st.session_state.get( 'text_domains' )
	input_mode = st.session_state.get( 'text_input' )
	tool_choice = st.session_state.get( 'text_tool_choice' )
	background = st.session_state.get( 'text_background' )
	messages = st.session_state.get( 'text_messages' )
	max_tokens = st.session_state.get( 'text_max_tokens' )
	
	if temperature is not None:
		right_parts.append( f'Temp: {temperature:.1%}' )
	if top_p is not None:
		right_parts.append( f'Top-P: {top_p:.1%}' )
	if freq is not None:
		right_parts.append( f'Freq: {freq:.2f}' )
	if presence is not None:
		right_parts.append( f'Presence: {presence:.2f}' )
	if number is not None:
		right_parts.append( f'N: {number}' )
	if max_tokens is not None:
		right_parts.append( f'Max Tokens: {max_tokens}' )
	
	if stream:
		right_parts.append( 'Stream: On' )
	if parallel_tools:
		right_parts.append( 'Parallel Tools: On' )
	if max_calls is not None:
		right_parts.append( f'Max Calls: {max_calls}' )
	if store:
		right_parts.append( 'Store: On' )
	if tools:
		right_parts.append( f'Tools: {len( tools )}' )
	if include:
		right_parts.append( 'Include: On' )
	if domains:
		right_parts.append( 'Domains: Set' )
	if input_mode:
		right_parts.append( 'Input: Set' )
	if tool_choice:
		right_parts.append( f'Tool Choice: On' )
	if background:
		right_parts.append( 'Background: On' )
	if messages:
		right_parts.append( 'Messages: Set' )


# ---- Rendering Method
st.markdown(
	f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{active_mode} — {mode_val}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """,
	unsafe_allow_html=True,
)