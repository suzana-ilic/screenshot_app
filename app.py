"""
ScreenshotMCP – Smart Screenshot Cleaner + Search (Gradio app)

- Scans a local folder of screenshots
- Uses Tesseract OCR + OpenAI to describe and categorize each screenshot
- Lets you mark screenshots as KEEP / REVIEW / DELETE
- Shows a confirmation dialog before deleting any files from disk
- Click a filename row to open that screenshot locally
- NEW: search screenshots with a natural-language query (e.g. "job posting")
"""

import os
import sys
import json
import yaml
import subprocess
from glob import glob
import textwrap
import inspect
import gradio.mcp as gr_mcp

import gradio as gr
from PIL import Image
import pytesseract
from openai import OpenAI

# =========================
# OpenAI config
# =========================
# We load the API key from ~/.config/openai/config.yaml
# with structure:
# api_key: "sk-xxxxx"
#
# This file should NEVER be committed to Git.

CONFIG_PATH = os.path.expanduser("~/.config/openai/config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(
        f"OpenAI config file not found at {CONFIG_PATH}. "
        "Create it with: api_key: \"YOUR_KEY_HERE\""
    )

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f) or {}

api_key = config.get("api_key")
if not api_key:
    raise RuntimeError("No 'api_key' field found in config.yaml")

client = OpenAI(api_key=api_key)

# =========================
# Configuration
# =========================

# Default screenshot folder (can be changed in the UI)
DEFAULT_SCREENSHOT_FOLDER = os.path.expanduser("~/Desktop/Screenshots")
MCP_MAX_FILES = 100


# =========================
# JSON parsing helper
# =========================

def _parse_json_block(content: str):
    """
    Try hard to extract a JSON object from an LLM response.
    Handles ```json fences and extra text.
    """
    content = content.strip()
    if content.startswith("```"):
        lines = [
            ln for ln in content.splitlines()
            if not ln.strip().startswith("```")
        ]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = content[start: end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from LLM output")


# =========================
# Normalize table rows
# =========================

def normalize_rows(rows):
    """
    Gradio sometimes returns a pandas.DataFrame, sometimes a list.
    Convert to a list-of-lists.
    """
    if rows is None:
        return []
    try:
        import pandas as pd  # type: ignore
        if isinstance(rows, pd.DataFrame):
            return rows.values.tolist()
    except Exception:
        pass
    if isinstance(rows, list):
        return rows
    return []


# =========================
# Files + OCR helpers
# =========================

def ocr_image(img: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        text = f"[OCR error: {e}]"
    return text.strip()


def list_screenshot_files(folder_path: str):
    folder_path = os.path.expanduser(folder_path)
    if not os.path.isdir(folder_path):
        return []
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(folder_path, pattern)))
    return sorted(files)


# =========================
# LLM: per-screenshot classification
# =========================

def llm_describe_and_classify(ocr_text: str):
    """
    Use the LLM to get a short description + KEEP/REVIEW/DELETE suggestion.
    """
    if not ocr_text.strip():
        return "(no text detected)", "DELETE – empty/low information", "DELETE"

    prompt = """
You are helping a busy professional clean up their screenshots folder.
Each screenshot is something the user decided to capture on their screen today:
emails, chats, dashboards, notices, tickets, documents, etc.

Given the OCR text extracted from ONE screenshot:

1. Write ONE short, human-friendly description of what this screenshot is (max 15 words).
2. Choose a category:
   - KEEP_IMPORTANT  (legal/financial docs, travel, job/offer emails, serious notices)
   - KEEP_NICE       (learning material, notes, ideas, useful references)
   - REVIEW          (sensitive, ambiguous, or unclear – user should inspect)
   - DELETE          (throwaway chat, memes, transient UI, low-signal noise)
3. Give a short reason in plain language.

Return ONLY JSON with keys: description, category, reason.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ocr_text[:6000]},
            ],
        )

        content = resp.choices[0].message.content
        data = _parse_json_block(content)

        description = (data.get("description") or "").strip() or "(no description)"
        category = (data.get("category") or "REVIEW").strip().upper()
        reason = (data.get("reason") or "").strip()

    except Exception:
        description = (
            ocr_text.splitlines()[0][:120] if ocr_text else "(no text detected)"
        )
        return description, "REVIEW – fallback classification", "REVIEW"

    # map category -> suggestion + default action
    if category == "KEEP_IMPORTANT":
        suggestion = f"KEEP – important ({reason})" if reason else "KEEP – important"
        action = "KEEP"
    elif category == "KEEP_NICE":
        suggestion = f"KEEP – useful ({reason})" if reason else "KEEP – useful"
        action = "KEEP"
    elif category == "DELETE":
        suggestion = f"DELETE – {reason}" if reason else "DELETE"
        action = "DELETE"
    else:
        suggestion = f"REVIEW – {reason}" if reason else "REVIEW – check manually"
        action = "REVIEW"

    return description, suggestion, action


# =========================
# LLM: global summary + actions
# =========================

def llm_summary_and_actions(all_text: str):
    if not all_text.strip():
        return "No readable text found.", ["(No actions detected.)"]

    prompt = """
You are looking at OCR text extracted from many screenshots the user took today.
Each screenshot represents something the user found important enough to capture
(emails, chats, dashboards, notices, forms, tickets, docs, etc.).

Your goals:
1. Summarize the day in 3–6 sentences: what topics, tasks, and themes appear?
2. Surface concrete, useful action items that would help the user move forward.
   Each action item should start with an imperative verb:
   - Reply to ...
   - Schedule ...
   - Review ...
   - Pay ...
   - Save ...
   - Follow up with ...

Only include actions that are clearly implied by the content.

Return ONLY JSON:
{
  "summary": " ... ",
  "actions": [" ... ", " ... "]
}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": all_text[:12000]},
            ],
        )
        content = resp.choices[0].message.content
        data = _parse_json_block(content)

        summary = (data.get("summary") or "").strip()
        actions = [a.strip() for a in (data.get("actions") or []) if a.strip()]

        if not summary:
            summary = "No clear summary available."
        if not actions:
            actions = ["(No obvious actions found.)"]

        return summary, actions

    except Exception:
        if not all_text.strip():
            return "No clear summary.", ["(LLM fallback – no actions)"]
        snippet = textwrap.fill(all_text[:400], 100)
        return snippet, ["(LLM fallback – no actions)"]


# =========================
# NEW: LLM relevance for search
# =========================

def llm_search_relevance(ocr_text: str, query: str):
    """
    Given OCR text from one screenshot and a natural-language query,
    return a relevance score and a short description.
    """
    if not ocr_text.strip():
        return 0.0, "(no text detected)"

    prompt = f"""
You are helping a user search their screenshots.

User query:
{query}

OCR text from ONE screenshot (may be noisy):
{ocr_text[:6000]}

1. Rate how relevant this screenshot is to the user query.
   Use a float between 0.0 and 1.0 (0 = not relevant at all, 1 = perfect match).
2. Write ONE short description (max 15 words) of what is shown in this screenshot.

Return ONLY JSON:
{{
  "score": 0.0,
  "description": "..."
}}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ocr_text[:6000]},
            ],
        )
        content = resp.choices[0].message.content
        data = _parse_json_block(content)

        score = float(data.get("score", 0.0))
        description = (data.get("description") or "").strip() or "(no description)"
        score = max(0.0, min(1.0, score))
        return score, description
    except Exception:
        # cheap fallback: keyword presence
        score = 0.8 if query.lower() in ocr_text.lower() else 0.1
        description = ocr_text.splitlines()[0][:120] if ocr_text else "(no text)"
        return score, description


# =========================
# Core processing
# =========================

def process_folder(folder_path: str):
    """
    Scan the folder, run OCR + LLM, and return:
    - summary markdown
    - actions markdown
    - table rows
    - debug text
    - cleanup script (bash)
    - mapping filename -> full path
    - visibility updates
    """
    folder_path = folder_path.strip()
    if not folder_path:
        return (
            "Please provide a folder path.",
            "",
            [],
            "No folder provided.",
            "_",
            {},
            gr.update(visible=False),  # review button
            gr.update(visible=False),  # dialog
        )

    files = list_screenshot_files(folder_path)
    if not files:
        return (
            "No screenshots found.",
            "",
            [],
            f"Scanned folder: {folder_path} — no images found.",
            "_No DELETE candidates._",
            {},
            gr.update(visible=False),
            gr.update(visible=False),
        )

    rows = []
    all_texts = []
    file_paths = {}

    for path in files:
        full_path = os.path.abspath(path)
        filename = os.path.basename(path)
        file_paths[filename] = full_path

        try:
            img = Image.open(path)
            ocr_text = ocr_image(img)
        except Exception as e:
            ocr_text = f"[Error opening: {e}]"
            description = "(error opening file)"
            suggestion = "SKIP"
            action = "REVIEW"
        else:
            description, suggestion, action = llm_describe_and_classify(ocr_text)

        all_texts.append(ocr_text)
        rows.append([filename, description, suggestion, action])

    combined_text = "\n\n".join(all_texts)
    summary, actions = llm_summary_and_actions(combined_text)

    summary_md = "### Summary\n\n" + summary
    actions_md = "### Action items\n\n" + "\n".join(f"- {a}" for a in actions)
    debug = f"Scanned folder: {folder_path}\nTotal screenshots: {len(files)}"

    delete_lines = []
    for filename, description, suggestion, action in rows:
        if isinstance(action, str) and action.strip().upper() == "DELETE":
            delete_lines.append(f'rm "{file_paths.get(filename, "")}"')
    if delete_lines:
        cleanup_script = "```bash\n" + "\n".join(delete_lines) + "\n```"
    else:
        cleanup_script = "_No DELETE candidates (based on Action column)._"

    review_btn_update = gr.update(visible=True)
    dialog_update = gr.update(visible=False)

    return (
        summary_md,
        actions_md,
        rows,
        debug,
        cleanup_script,
        file_paths,
        review_btn_update,
        dialog_update,
    )


def prepare_delete(rows, file_paths):
    """
    Build the preview text for the confirmation dialog and show it.
    """
    if not isinstance(file_paths, dict):
        return "Internal error: file path mapping missing.", gr.update(visible=False), gr.update(interactive=False)

    rows = normalize_rows(rows)
    to_delete = []

    for row in rows:
        if not row or len(row) < 4:
            continue
        filename, description, suggestion, action = row
        if isinstance(action, str) and action.strip().upper() == "DELETE":
            to_delete.append(filename)

    if not to_delete:
        preview = "No files currently marked `DELETE` in the Action column."
        return preview, gr.update(visible=True), gr.update(interactive=False)

    preview_lines = "\n".join(f"- {fn}" for fn in to_delete)
    preview_md = (
        "### Confirm deletion\n\n"
        "The following screenshots will be **deleted from disk** if you click "
        "**Proceed & delete**:\n\n"
        f"{preview_lines}"
    )
    return preview_md, gr.update(visible=True), gr.update(interactive=True)


def apply_deletes(rows, file_paths):
    """
    After confirmation, delete all files with Action == DELETE.
    """
    if not isinstance(file_paths, dict):
        return "Internal error: file path mapping missing."

    rows = normalize_rows(rows)

    deleted = 0
    missing = 0
    failed = 0

    for row in rows:
        if not row or len(row) < 4:
            continue
        filename, description, suggestion, action = row
        if not isinstance(action, str):
            continue
        if action.strip().upper() != "DELETE":
            continue

        path = file_paths.get(filename)
        if not path:
            missing += 1
            continue

        try:
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
            else:
                missing += 1
        except Exception:
            failed += 1

    msg_parts = [f"Deleted: {deleted}"]
    if missing:
        msg_parts.append(f"Missing: {missing}")
    if failed:
        msg_parts.append(f"Failed: {failed}")

    if deleted == 0 and missing == 0 and failed == 0:
        return "No files deleted (no rows with Action = DELETE)."

    return "✅ " + " | ".join(msg_parts)


def cancel_delete():
    """Hide the confirmation dialog and clear its text."""
    return "", "", gr.update(visible=False)


def open_screenshot_from_table(rows, file_paths, evt: gr.SelectData):
    """
    When the user clicks a row in the table, open that screenshot locally.
    """
    if not isinstance(file_paths, dict):
        return "Internal error: file path mapping missing."

    rows = normalize_rows(rows)

    if not evt or evt.index is None:
        return "Nothing selected."
    row_idx = evt.index[0]

    if row_idx < 0 or row_idx >= len(rows):
        return "Invalid selection."

    filename = rows[row_idx][0]
    path = file_paths.get(filename)

    if not path:
        return f"Path not found for '{filename}'."
    if not os.path.exists(path):
        return f"File no longer exists on disk: {path}"

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
        return f"Opened: {path}"
    except Exception as e:
        return f"Failed to open file: {e}"


# =========================
# NEW: natural-language search over screenshots
# =========================

def search_folder_for_query(folder_path: str, query: str,
                            min_score: float = 0.2,
                            top_k: int = 20):
    """
    Scan the folder, run OCR + relevance scoring for a natural-language query.

    Returns ONLY the hits (score >= min_score), sorted by score desc,
    limited to top_k.

    Outputs:
    - rows: [Filename, Match description, Score]
    - mapping filename -> full path (for click-to-open)
    - debug text
    """
    folder_path = folder_path.strip()
    if not folder_path:
        return [], {}, "Please provide a folder path."

    if not query.strip():
        return [], {}, "Please provide a search query."

    files = list_screenshot_files(folder_path)
    if not files:
        return [], {}, f"Scanned folder: {folder_path} — no images found."

    rows = []
    file_paths = {}

    for path in files:
        full_path = os.path.abspath(path)
        filename = os.path.basename(path)
        file_paths[filename] = full_path

        try:
            img = Image.open(path)
            ocr_text = ocr_image(img)
        except Exception as e:
            score = 0.0
            description = f"(error opening file: {e})"
        else:
            score, description = llm_search_relevance(ocr_text, query)

        rows.append([filename, description, float(score)])

    # keep only hits
    hits = [r for r in rows if r[2] >= min_score]

    # sort by score descending
    hits.sort(key=lambda r: r[2], reverse=True)

    # cap number of results
    hits = hits[:top_k]

    debug = (
        f"Scanned folder: {folder_path}\n"
        f"Total screenshots: {len(files)}\n"
        f"Returned hits (score ≥ {min_score}): {len(hits)}"
    )
    return hits, file_paths, debug


def open_screenshot_from_search(rows, file_paths, evt: gr.SelectData):
    """
    Same as open_screenshot_from_table, but for the search results table.
    """
    return open_screenshot_from_table(rows, file_paths, evt)


# =========================
# MCP tool wrappers (API-first)
# =========================


def _resolve_folder(folder_path: str) -> str:
    """Expand user paths and validate the folder exists."""
    folder_path = os.path.abspath(os.path.expanduser(folder_path or DEFAULT_SCREENSHOT_FOLDER))
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    return folder_path


@gr_mcp.tool(name="list_screenshots", description="List screenshot files in a folder.")
def mcp_list_screenshots(folder: str = DEFAULT_SCREENSHOT_FOLDER,
                         max_files: int = 50):
    folder_path = _resolve_folder(folder)
    max_files = int(max_files)
    files = list_screenshot_files(folder_path)[: min(max_files, MCP_MAX_FILES)]
    return [
        {"filename": os.path.basename(p), "path": p}
        for p in files
    ]


@gr_mcp.tool(name="analyze_screenshots", description="OCR + classify screenshots in a folder.")
def mcp_analyze_screenshots(folder: str = DEFAULT_SCREENSHOT_FOLDER,
                            max_files: int = 20):
    folder_path = _resolve_folder(folder)
    max_files = int(max_files)
    files = list_screenshot_files(folder_path)[: min(max_files, MCP_MAX_FILES)]
    results = []

    for path in files:
        filename = os.path.basename(path)
        try:
            img = Image.open(path)
            ocr_text = ocr_image(img)
        except Exception as e:
            results.append(
                {
                    "filename": filename,
                    "description": f"(error opening: {e})",
                    "suggestion": "SKIP",
                    "action": "REVIEW",
                }
            )
            continue

        description, suggestion, action = llm_describe_and_classify(ocr_text)
        results.append(
            {
                "filename": filename,
                "description": description,
                "suggestion": suggestion,
                "action": action,
            }
        )

    return {"folder": folder_path, "results": results}


@gr_mcp.tool(name="search_screenshots", description="Search screenshots with a natural-language query.")
def mcp_search_screenshots(folder: str = DEFAULT_SCREENSHOT_FOLDER,
                           query: str = "",
                           max_files: int = 50,
                           top_k: int = 10,
                           min_score: float = 0.2):
    folder_path = _resolve_folder(folder)
    max_files = int(max_files)
    top_k = int(top_k)
    files = list_screenshot_files(folder_path)[: min(max_files, MCP_MAX_FILES)]
    if not query.strip():
        return {"error": "Please provide a search query."}

    rows = []
    for path in files:
        filename = os.path.basename(path)
        try:
            img = Image.open(path)
            ocr_text = ocr_image(img)
        except Exception as e:
            score = 0.0
            description = f"(error opening file: {e})"
        else:
            score, description = llm_search_relevance(ocr_text, query)

        rows.append([filename, description, float(score)])

    hits = [r for r in rows if r[2] >= min_score]
    hits.sort(key=lambda r: r[2], reverse=True)
    hits = hits[:top_k]

    return {
        "folder": folder_path,
        "query": query,
        "results": [
            {"filename": f, "description": desc, "score": score}
            for f, desc, score in hits
        ],
    }


@gr_mcp.tool(name="delete_screenshots", description="Delete screenshots by filename within a folder.")
def mcp_delete_screenshots(folder: str = DEFAULT_SCREENSHOT_FOLDER,
                           filenames: list[str] | str = None):
    folder_path = _resolve_folder(folder)
    if filenames is None:
        return {"deleted": [], "deleted_count": 0, "skipped": []}

    if isinstance(filenames, str):
        try:
            filenames = json.loads(filenames)
        except Exception:
            filenames = [filenames]

    deleted = []
    skipped = []
    for fname in filenames:
        path = os.path.join(folder_path, fname)
        if not os.path.exists(path):
            skipped.append([fname, "not found"])
            continue
        try:
            os.remove(path)
            deleted.append(fname)
        except Exception as e:
            skipped.append([fname, f"error: {e}"])

    return {
        "folder": folder_path,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "skipped": skipped,
    }


# =========================
# Gradio UI
# =========================

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🧹 ScreenshotMCP – Smart Screenshot Cleaner

        - Scan a **local folder** of screenshots  
        - Use OCR + LLM to describe and categorize them  
        - Edit the **Action** column (KEEP / REVIEW / DELETE)  
        - Use **Review & delete** to confirm before any files are removed  
        - Click any **Filename** row to open that screenshot locally  
        - 🔍 NEW: search screenshots with a natural-language query
        """
    )

    # ---------- Main cleaner UI (original) ----------
    folder_input = gr.Textbox(
        label="Screenshot folder path",
        value=DEFAULT_SCREENSHOT_FOLDER,
    )

    with gr.Row():
        scan_btn = gr.Button("Scan folder", variant="primary")
        review_delete_btn = gr.Button(
            "Review & delete screenshots marked DELETE",
            variant="secondary",
            visible=False,
        )

    summary_output = gr.Markdown(label="Summary")
    actions_output = gr.Markdown(label="Action items")
    debug_output = gr.Markdown(label="Scan info")
    cleanup_output = gr.Markdown(label="Cleanup script (optional)")
    open_status = gr.Markdown(label="Open status")

    table = gr.Dataframe(
        headers=["Filename", "Description", "Suggestion", "Action"],
        datatype=["str", "str", "str", "str"],
        interactive=True,
        label="Screenshot analysis (click Filename to open)",
    )

    file_state = gr.State({})  # filename -> full path

    # "Dialog" group for delete confirmation
    with gr.Group(visible=False) as delete_dialog:
        delete_preview_output = gr.Markdown()
        with gr.Row():
            proceed_delete_btn = gr.Button("Proceed & delete", variant="stop")
            cancel_delete_btn = gr.Button("Cancel", variant="secondary")
        delete_status = gr.Markdown()

    # Scan folder
    scan_btn.click(
        fn=process_folder,
        inputs=folder_input,
        outputs=[
            summary_output,
            actions_output,
            table,
            debug_output,
            cleanup_output,
            file_state,
            review_delete_btn,
            delete_dialog,
        ],
    )

    # Show confirmation dialog
    review_delete_btn.click(
        fn=prepare_delete,
        inputs=[table, file_state],
        outputs=[delete_preview_output, delete_dialog, proceed_delete_btn],
    )

    # Confirm delete
    proceed_delete_btn.click(
        fn=apply_deletes,
        inputs=[table, file_state],
        outputs=delete_status,
    )

    # Cancel dialog
    cancel_delete_btn.click(
        fn=cancel_delete,
        inputs=None,
        outputs=[delete_preview_output, delete_status, delete_dialog],
    )

    # Click row -> open screenshot
    table.select(
        fn=open_screenshot_from_table,
        inputs=[table, file_state],
        outputs=open_status,
    )

    # ---------- NEW: Search UI ----------
    gr.Markdown("## 🔍 Search screenshots")

    search_folder_input = gr.Textbox(
        label="Screenshot folder path (for search)",
        value=DEFAULT_SCREENSHOT_FOLDER,
    )
    search_query_input = gr.Textbox(
        label="Search query",
        placeholder="e.g. job posting",
    )

    with gr.Row():
        search_btn = gr.Button("Search screenshots", variant="primary")

    search_results = gr.Dataframe(
        headers=["Filename", "Match description", "Score"],
        datatype=["str", "str", "number"],
        interactive=False,
        label="Search results (click row to open screenshot)",
    )
    search_debug = gr.Markdown(label="Search info")
    search_open_status = gr.Markdown(label="Search open status")
    search_file_state = gr.State({})  # filename -> full path for search

    search_btn.click(
        fn=search_folder_for_query,
        inputs=[search_folder_input, search_query_input],
        outputs=[search_results, search_file_state, search_debug],
    )

    search_results.select(
        fn=open_screenshot_from_search,
        inputs=[search_results, search_file_state],
        outputs=search_open_status,
    )

    # ---------- MCP tool endpoints (hidden UI, API-only) ----------
    with gr.Group(visible=False):
        mcp_folder_input = gr.Textbox(value=DEFAULT_SCREENSHOT_FOLDER, label="MCP folder")
        mcp_max_files_input = gr.Number(value=20, label="MCP max files")
        mcp_query_input = gr.Textbox(value="", label="MCP search query")
        mcp_min_score_input = gr.Number(value=0.2, label="MCP min score")
        mcp_top_k_input = gr.Number(value=10, label="MCP top_k")
        mcp_filenames_input = gr.Textbox(value="[]", label="MCP filenames (JSON list)")

        mcp_list_output = gr.JSON(label="mcp_list_screenshots")
        mcp_analyze_output = gr.JSON(label="mcp_analyze_screenshots")
        mcp_search_output = gr.JSON(label="mcp_search_screenshots")
        mcp_delete_output = gr.JSON(label="mcp_delete_screenshots")

        gr.Button("Run mcp_list_screenshots", visible=False).click(
            fn=mcp_list_screenshots,
            inputs=[mcp_folder_input, mcp_max_files_input],
            outputs=mcp_list_output,
            api_name="mcp_list_screenshots",
        )

        gr.Button("Run mcp_analyze_screenshots", visible=False).click(
            fn=mcp_analyze_screenshots,
            inputs=[mcp_folder_input, mcp_max_files_input],
            outputs=mcp_analyze_output,
            api_name="mcp_analyze_screenshots",
        )

        gr.Button("Run mcp_search_screenshots", visible=False).click(
            fn=mcp_search_screenshots,
            inputs=[
                mcp_folder_input,
                mcp_query_input,
                mcp_max_files_input,
                mcp_top_k_input,
                mcp_min_score_input,
            ],
            outputs=mcp_search_output,
            api_name="mcp_search_screenshots",
        )

        gr.Button("Run mcp_delete_screenshots", visible=False).click(
            fn=mcp_delete_screenshots,
            inputs=[mcp_folder_input, mcp_filenames_input],
            outputs=mcp_delete_output,
            api_name="mcp_delete_screenshots",
        )

if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }

    # Enable MCP server mode when supported by this Gradio version.
    if "mcp_server" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs["mcp_server"] = True
        print("Launching with MCP server enabled.")
    else:
        print(
            "gradio.Blocks.launch() has no `mcp_server` argument. "
            "Upgrade to gradio[mcp]>=4.48.0 to expose MCP tools."
        )

    demo.launch(**launch_kwargs)
