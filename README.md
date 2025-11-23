# ScreenshotMCP – Smart Screenshot Cleaner (Gradio)

A small local tool that scans a folder of screenshots, uses OCR + an LLM
to describe them, suggests KEEP / REVIEW / DELETE, and lets you safely
bulk-delete clutter after a confirmation step.

## Features

- Scans a local folder of screenshots (`.png`, `.jpg`, `.jpeg`)
- Uses Tesseract OCR + OpenAI GPT to describe each screenshot
- Suggests actions per screenshot (KEEP / REVIEW / DELETE)
- Click a filename row to open the screenshot locally
- Review & confirm before any screenshots are deleted

# Safety

The app never uploads your screenshots anywhere. Screenshot files are only deleted after you:
- Mark them as DELETE in the Action column, and
- Confirm in the “Proceed & delete” dialog.

## Installation

```bash
git clone https://github.com/suzana-ilic/screenshot-app.git
cd screenshot-app

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
