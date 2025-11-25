# ScreenshotMCP — Smart Screenshot Cleaner + Search

Me and probably millions of other people have the same problem:  
we hoard hundreds (sometimes thousands) of screenshots because they’re important *in the moment*, but they instantly vanish into chaos. There’s no way to search them. No way to organize them. No way to actually use them later.

**ScreenshotMCP fixes that.**  
It turns your screenshots into a fully searchable, semantically aware, privacy-preserving knowledge base — powered by a local MCP server.

## Overview

Screenshots are fast to capture and impossible to find later.

ScreenshotMCP transforms your screenshot folder into a structured, queryable dataset using **OCR + LLMs**, exposed through a local MCP server so AI tools like **Cursor** and **Claude Desktop** can interact with them safely.

## Problem

Screenshots folders contain

- job postings  
- receipts  
- flight confirmations  
- UI bug reports  
- memes  
- medical instructions  
- TODO reminders  
- conversations  

But:

- filenames mean nothing  
- you can’t search text inside images  
- deleting is tedious  
- screenshots turn into digital clutter  

**Screenshots = unstructured chaos.**

## Solution

ScreenshotMCP introduces an intelligent screenshot workflow:

### ✔️ OCR every screenshot (Tesseract)
Extracts real text from images.

### ✔️ LLM-powered classification
Each screenshot is analyzed and labeled with:

- a short description  
- a category (`KEEP_IMPORTANT`, `KEEP`, `REVIEW`, `DELETE`)  
- a reason  
- suggested action  

### ✔️ Natural-language search  
Search your screenshots conversationally:

- “Find job applications”  
- “Screenshots containing my passport number”  
- “Flight bookings this year”  
- “Screenshots with error messages”  

### ✔️ Safe deletion workflow  
Only (bulk) deletes after explicit confirmation.

### ✔️ Fully local & privacy-first  
Your images never leave your machine.

### ✔️ Exposed through MCP  
Allows AI tools to read, classify, search, and manage screenshots through tool calls.

## Features

| Feature | Description |
|--------|-------------|
| OCR Extraction | Reads text inside PNG/JPG screenshots |
| Screenshot Classification | LLM assigns categories + descriptions |
| Natural-Language Search | Semantic retrieval of screenshots |
| Safe Deletion | Confirmation-based deletion flow |
| Click-to-Open | Open screenshot files directly |
| MCP Tools | 4 consistent tool endpoints |

## MCP Tools

| Tool | Purpose |
|------|---------|
| `mcp_list_screenshots` | List files in a screenshot folder |
| `mcp_analyze_screenshots` | OCR + description + category suggestions |
| `mcp_search_screenshots` | Natural-language search with scoring |
| `mcp_delete_screenshots` | Safely delete selected screenshots | 

## Installation

### 1. Clone the repo

git clone https://github.com/suzana-ilic/screenshot_app
cd screenshot-app

### 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Install Tesseract (macOS)
brew install tesseract

### 5. Run the MCP server
python app.py


## Contributors & dev tools used
- [Suzana Ilić](https://x.com/suzatweet)
- **VS Code & Codex; Cursor** — for development, debugging, MCP testing  
- **Gradio** — UI + MCP integration  
- **Python** — server + business logic  
- **Tesseract OCR** — text extraction  
- **OpenAI GPT-4o** — screenshot semantic analysis  
- **httpx / sseclient** — streaming MCP calls 