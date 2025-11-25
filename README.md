# ScreenshotMCP — Smart Screenshot Cleaner + Search

Me and probably millions of other people have the same problem: We hoard hundreds (sometimes thousands) of screenshots because they’re important *in the moment*, but they instantly vanish into chaos. There’s no way to search them. No way to organize them. No way to actually use them later.

**ScreenshotMCP fixes that.**  
It turns your screenshots into a fully searchable, semantically aware, privacy-preserving knowledge base — powered by a local MCP server. ScreenshotMCP transforms your screenshot folder into a structured, queryable dataset using OCR + LLMs, exposed through a local MCP server so AI tools like Cursor and Claude Desktop can interact with them safely.

https://github.com/user-attachments/assets/dbd2bc88-d014-4861-9d60-1e62e576763f

## Problem

Screenshots folders contain job postings, receipts, flight confirmations, bug reports, memes, TODO reminders.

But:

- filenames mean nothing  
- you can’t search text inside images  
- deleting is tedious  
- screenshots turn into digital clutter  

**Screenshots = unstructured chaos.**

## Solution

ScreenshotMCP introduces an intelligent screenshot workflow:

#### ✔️ OCR every screenshot: Extracts real text from images.

#### ✔️ LLM-powered classification: Each screenshot is analyzed and labeled with:
- a short description  
- a category (`KEEP_IMPORTANT`, `KEEP`, `REVIEW`, `DELETE`)  
- a reason  
- suggested action  

#### ✔️ Natural-language search: Search your screenshots conversationally

#### ✔️ Safe deletion workflow: Only (bulk) deletes after explicit confirmation

#### ✔️ Fully local & privacy-first: Your images never leave your machine

### ✔️ Exposed through MCP: Allows AI tools to read, classify, search, and manage screenshots through tool calls

## Features

| Feature | Description |
|--------|-------------|
| OCR Extraction | Reads text inside PNG/JPG screenshots |
| Screenshot Classification | LLM assigns categories + descriptions |
| Natural-Language Search | Semantic retrieval of screenshots |
| Safe Deletion | Confirmation-based deletion flow |
| Click-to-Open | Open screenshot files directly |
| MCP Tools | 4 consistent tool endpoints |

## MCP API Tools (Primary Tools)

| Tool Name               | Parameters                                                                                       | Description                                      |
|-------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------|
| **mcp_list_screenshots**   | `folder` (string), `max_files` (number, default: 20)                                             | Lists screenshot files in a folder.              |
| **mcp_analyze_screenshots** | `folder` (string), `max_files` (number, default: 20)                                             | Runs OCR + LLM classification on screenshots.    |
| **mcp_search_screenshots**  | `folder` (string), `query` (string), `max_files` (20), `top_k` (10), `min_score` (0.2)           | Natural-language screenshot search.              |
| **mcp_delete_screenshots**  | `folder` (string), `filenames` (string or list, default: \[\])                                  | Deletes screenshots by filename.                 |

## UI / Gradio Tools (Internal UI Tools)

| Tool Name                        | Parameters                                          | Description                                          |
|----------------------------------|-----------------------------------------------------|------------------------------------------------------|
| **process_folder**               | `folder_path` (string)                              | Full folder OCR + classification pipeline.           |
| **prepare_delete**               | `rows` (table), `file_paths` (dict)                 | Prepares deletion confirmation dialog.               |
| **apply_deletes**                | `rows` (table), `file_paths` (dict)                 | Executes screenshot deletions.                       |
| **cancel_delete**                | *(none)*                                            | Cancels deletion dialog.                             |
| **open_screenshot_from_table**   | `rows` (table), `file_paths` (dict), `evt`          | Opens screenshot from UI table click.                |
| **search_folder_for_query**      | `folder` (string), `query` (string)                 | Natural-language search in UI.                        |
| **open_screenshot_from_search**  | `results` (table), `evt`                            | Opens screenshot from search results.                |


## Instructions

### 1. Clone the repo

_git clone https://github.com/suzana-ilic/screenshot_app
_
_cd screenshot-app
_
### 2. Create a virtual environment
_python3 -m venv venv
_
_source venv/bin/activate
_
### 3. Install dependencies
_pip install -r requirements.txt
_

### 4. Run the MCP server
_python app.py
_

## Contributors & dev tools used
- [Suzana Ilić](https://x.com/suzatweet)
- **VS Code & Codex; Cursor** — for development, debugging, MCP testing  
- **Gradio** — UI + MCP integration  
- **Python** — server + business logic  
- **Tesseract OCR** — text extraction  
- **OpenAI GPT-4o** — screenshot semantic analysis  
- **httpx / sseclient** — streaming MCP calls 
