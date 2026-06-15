"""SQLite-backed exception wrapping and logging utilities for Leeroy.

Purpose:
	Defines the application exception wrapper and logger used by Leeroy source files to
	capture structured diagnostic metadata and persist failure records to the configured
	SQLite exception database. The module centralizes error metadata, traceback capture,
	and database writes so application code can use a consistent logging pattern without
	duplicating SQLite setup or traceback formatting logic.
"""
from __future__ import annotations

import os
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from sys import exc_info
from typing import Any, List

import config as cfg

HEADLESS = ("STREAMLIT_SERVER_RUNNING" in os.environ
            or "streamlit" in os.environ.get( "PYTHONPATH", "" ).lower( ))

class Error( Exception ):
	"""Wrap an exception with application-specific diagnostic metadata.

	Purpose:
		Captures the original exception, active traceback text, optional display heading,
		logical cause, source module, and stable method signature used by Leeroy exception
		handlers. The wrapper provides a consistent diagnostic object that can be raised by
		application code and written by ``Logger`` without forcing each caller to format
		traceback details manually.

	Attributes:
		exception: Original exception instance being wrapped.
		heading: Optional display heading associated with the error.
		cause: Logical component or class responsible for the failure.
		method: Stable method or function signature where the failure occurred.
		module: Source module where the failure occurred.
		type: Active exception type reported by ``sys.exc_info``.
		trace: Formatted traceback text captured when the wrapper is constructed.
		info: Combined exception type and formatted traceback text.
	"""
	
	def __init__( self, error: Exception, heading: str = None, cause: str = None,
			method: str = None, module: str = None ):
		"""Initialize the error wrapper.

		Purpose:
			Stores the original exception and optional diagnostic metadata, then captures
			the active exception type and traceback immediately. Immediate capture preserves
			the failure context for later logging even after execution has left the original
			``except`` block.

		Args:
			error: Original exception instance being wrapped.
			heading: Optional display heading associated with the error.
			cause: Logical component or class responsible for the failure.
			method: Stable method or function signature where the failure occurred.
			module: Source module where the failure occurred.
		"""
		super( ).__init__( )
		self.exception = error
		self.heading = heading
		self.cause = cause
		self.method = method
		self.module = module
		self.type = exc_info( )[ 0 ]
		self.trace = traceback.format_exc( )
		self.info = str( exc_info( )[ 0 ] ) + ': \r\n \r\n' + traceback.format_exc( )
	
	def __str__( self ) -> str | None:
		"""Return the captured diagnostic text.

		Purpose:
			Returns the formatted exception information captured when the wrapper was
			created. This representation supports direct display, diagnostic logging, and
			debugging without requiring callers to inspect individual metadata fields.

		Returns:
			str | None: Captured exception information when available.
		"""
		if self.info is not None:
			return self.info
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return the public diagnostic member names.

		Purpose:
			Provides a stable member list for inspectors, debuggers, documentation tools,
			and interactive sessions that need to display the primary diagnostic fields
			carried by the wrapper.

		Returns:
			list[str]: Ordered diagnostic member names exposed by the wrapper.
		"""
		return [ 'message', 'cause', 'method', 'module', 'scaler', 'stack_trace', 'info' ]

class Logger( ):
	"""Persist wrapped exception details to the configured SQLite logging database.

	Purpose:
		Writes ``Error`` metadata to the SQLite database identified by ``config.LOG_PATH``
		and the table identified by ``config.LOG_FILE``. The class creates the logging
		directory and exception table when needed, then records cause, module, method,
		message, formatted diagnostic information, traceback text, and creation time for
		later troubleshooting.

	Attributes:
		path: Filesystem path to the SQLite logging database.
		table: SQLite table name used for persisted exception records.
		query: SQL statement prepared for the active logging operation.
		values: Values prepared for the active logging operation.
	"""
	
	def __init__( self ) -> None:
		"""Initialize the logger.

		Purpose:
			Loads the configured logging database path and exception table name from the
			central Leeroy configuration. The constructor prepares local state for later
			bounded setup and write operations without opening a persistent database
			connection.
		"""
		self.path = Path( cfg.LOG_PATH ).resolve( )
		self.table = str( cfg.LOG_FILE or 'Exceptions' )
		self.query = None
		self.values = None
	
	def __dir__( self ) -> List[ str ]:
		"""Return the public logger member names.

		Purpose:
			Provides a stable member list for inspection and debugging of logger
			configuration, including the configured path, table name, SQL state, and write
			helper methods.

		Returns:
			list[str]: Ordered logger member names.
		"""
		return [ 'path', 'table', 'query', 'values', 'create_table', 'write' ]
	
	def create_table( self ) -> None:
		"""Create the exception table when it does not already exist.

		Purpose:
			Ensures the logging directory and SQLite exception table exist before exception
			records are written. The schema stores stable diagnostic fields used by Leeroy
			modules and suppresses setup failures so logging does not mask the original
			application exception.
		"""
		try:
			self.path.parent.mkdir( parents=True, exist_ok=True )
			self.query = f'''
				CREATE TABLE IF NOT EXISTS {self.table} (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					created TEXT,
					cause TEXT,
					module TEXT,
					method TEXT,
					message TEXT,
					info TEXT,
					trace TEXT
				)
			'''
			with sqlite3.connect( self.path ) as connection:
				connection.execute( self.query )
				connection.commit( )
		except Exception:
			return None
	
	def write( self, error: Error ) -> None:
		"""Write an error record to the logging database.

		Purpose:
			Persists a wrapped ``Error`` object to the configured SQLite database using the
			standard Leeroy exception schema. Logging is intentionally failure-safe so a
			database or filesystem problem during logging does not suppress or replace the
			original application error.

		Args:
			error: Wrapped exception object containing diagnostic metadata to persist.
		"""
		try:
			self.create_table( )
			message = str( getattr( error, 'exception', '' ) )
			self.query = f'''
				INSERT INTO {self.table} (
					created,
					cause,
					module,
					method,
					message,
					info,
					trace
				)
				VALUES (?, ?, ?, ?, ?, ?, ?)
			'''
			self.values = (
					datetime.now( ).isoformat( timespec='seconds' ),
					getattr( error, 'cause', None ),
					getattr( error, 'module', None ),
					getattr( error, 'method', None ),
					message,
					getattr( error, 'info', None ),
					getattr( error, 'trace', None ),
			)
			with sqlite3.connect( self.path ) as connection:
				connection.execute( self.query, self.values )
				connection.commit( )
		except Exception:
			return None