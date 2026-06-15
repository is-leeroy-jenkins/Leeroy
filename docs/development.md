# Development

This page describes the recommended development, validation, documentation, and deployment workflow
for Leeroy.

## 🧭 Purpose

The development workflow is designed to preserve application behavior while improving documentation,
logging, MkDocs compatibility, and deployment readiness. Leeroy uses Streamlit for the application
interface, SQLite for persistence, Google-style Python docstrings for source documentation, and
MkDocs Material for the documentation site.

## 🧰 Environment Setup

Create and activate a virtual environment from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip and install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🧪 Runtime Validation

Compile the main files after source changes.

```powershell
python -m py_compile .\app.py
python -m py_compile .\config.py
python -m py_compile .\boogr.py
```

Compile the whole repository.

```powershell
python -m compileall .
```

Run the Streamlit application.

```powershell
streamlit run app.py
```

Verify the following UI areas after changes:

| Area               | Validation                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------- |
| Sidebar            | Logo, mode selector, local model status, and navigation controls render.                 |
| Text Generation    | Chat input, system instructions, and inference controls render.                          |
| Document Q&A       | Upload controls, preview pane, chat history, and retrieval prompt flow render.           |
| Semantic Search    | File upload, chunking, embedding, and indexing controls render.                          |
| Prompt Engineering | Search, sort, paging, selection, edit, create, and delete controls render.               |
| Data Management    | Import, browse, CRUD, explore, filter, aggregate, visualize, admin, and SQL tabs render. |

## 📚 Documentation Build

Build the documentation from the repository root.

```powershell

mkdocs build

```

Serve the documentation locally.

```powershell

mkdocs serve

```

Open the local documentation site.

```text

http://127.0.0.1:8000/

```

## ✅ Documentation Validation Checklist

Use this checklist before publishing.

| Check             | Expected Result                                              |
| ----------------- | ------------------------------------------------------------ |
| `mkdocs build`    | Completes without documentation errors.                      |
| Home page         | Renders the project overview.                                |
| Architecture page | Renders system layers and workflow explanation.              |
| User Guide pages  | Render task-oriented application guidance.                   |
| API pages         | Render `app`, `config`, and `boogr` through mkdocstrings.    |
| Search            | Search box returns relevant page results.                    |
| Tables            | Tables render with dark-mode styling.                        |
| Code blocks       | Copy buttons and language labels work.                       |
| API tools         | API filter and expand/collapse controls appear on API pages. |
| Header            | Dark blue title/header background appears.                   |
| Navigation        | All configured nav entries resolve to existing files.        |

## 🧾 Google-Style Docstring Standard

Public modules, classes, functions, methods, and properties should use Google-style docstrings.

Recommended structure:

```python

def function_name( value: str ) -> str:
    """Return a normalized value.

    Purpose:
        Normalizes a source value for use in prompt, retrieval, or persistence workflows.

    Args:
        value: Source value to normalize.

    Returns:
        str: Normalized value.
    """

```

Use these section names only when applicable:

```text

Purpose:
Args:
Attributes:
Returns:
Raises:
Notes:
Examples:

```

Avoid these patterns:

```text

Parameters:
-----------
Returns:
--------
Returns:
    None:
    
```

Do not add `Returns:` sections to `__init__` methods.

## 🧯 Logging Pattern

Use the application logging pattern for existing exception handlers that need diagnostics.

```python

from boogr import Error, Logger

try:
    result = operation()
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'database'
    exception.method = 'operation_name( ) -> object'
    Logger( ).write( exception )
    raise exception

```

For fallback helpers, preserve the original fallback behavior.

```python

try:
    return bool( value )
except Exception as e:
    exception = Error( e )
    exception.module = 'app'
    exception.cause = 'runtime'
    exception.method = 'helper_name( ) -> bool'
    Logger( ).write( exception )
    return False

```

The `exception.method` value must be a stable signature string. It must not include prompts, user
input, document text, SQL text, paths, tokens, secrets, dataframe contents, or object memory
addresses.

## 🗂️ Documentation Structure

Recommended documentation tree:

```text

docs/
├── index.md
├── architecture.md
├── development.md
├── user-guide/
│   ├── index.md
│   ├── text-generation.md
│   ├── document-qna.md
│   ├── semantic-search.md
│   ├── prompt-engineering.md
│   └── data-management.md
├── api/
│   ├── index.md
│   ├── app.md
│   ├── config.md
│   └── boogr.md
├── assets/
│   ├── css/
│   │   └── leeroy.css
│   └── js/
│       └── leeroy.js
└── images/
    ├── leeroy-architecture.png
    └── leeroy-class-map.png
    
```

## 🚢 GitHub Pages Deployment

Build locally first.

```powershell
mkdocs build
```

Deploy to GitHub Pages.

```powershell
mkdocs gh-deploy
```

The documentation site should publish to:

```text
https://is-leeroy-jenkins.github.io/Leeroy/
```

## 🧹 Common Build Issues

| Message                         | Cause                                                                     | Fix                                                       |
| ------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| Page exists but is not in `nav` | A Markdown file exists under `docs/` but is not listed in `mkdocs.yml`.   | Add it to `nav` or remove the unused file.                |
| Nav references missing page     | `mkdocs.yml` points to a file that does not exist.                        | Create the file or remove the nav entry.                  |
| Griffe docstring warning        | A Python docstring uses malformed Google-style sections.                  | Fix the source docstring, then rebuild.                   |
| Missing image target            | Markdown references an image path that does not exist.                    | Add the image or correct the path.                        |
| API page renders empty          | mkdocstrings cannot import the module or the module lacks public members. | Check module path, imports, and `::: module_name` syntax. |

## ✅ Recommended Change Sequence

1. Modify source code.
2. Run `python -m py_compile` for changed Python files.
3. Run `python -m compileall .`.
4. Run `streamlit run app.py`.
5. Run `mkdocs build`.
6. Preview with `mkdocs serve`.
7. Commit and push.
8. Deploy with `mkdocs gh-deploy`.
