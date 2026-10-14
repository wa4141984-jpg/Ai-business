# Ai-# 📄 AI Resume ATS Analyzer

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get an
**estimated ATS score**, a category breakdown, prioritised fixes, missing keywords and
before → after bullet rewrites. Built with **Streamlit** and **Google Gemini Flash**.

> ⚠️ The score is an AI-based estimate, not the output of a real ATS. Different employers
> use different systems, so treat it as guidance.

## Features

- PDF / DOCX / TXT parsing (with clear errors for scanned or password-protected files)
- Weighted score across 5 categories: keywords, impact, formatting, structure, clarity
- Optional job-description matching
- Prioritised issues with concrete fixes, found/missing keywords, bullet rewrites
- Downloadable Markdown report

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then create
`.streamlit/secrets.toml` (this file is git-ignored — never commit it):

```toml
GEMINI_API_KEY = "your-key-here"
```

```bash
streamlit run app.py
```

If you skip the secrets file, the app asks for the key in the sidebar instead.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets or env var | required (or enter in sidebar) |
| `GEMINI_MODEL` | secrets or env var | `gemini-flash-latest`, falling back to `gemini-3.6-flash` |

`gemini-flash-latest` always points to Google's current Flash model, so the app keeps
working as older models are retired. Set `GEMINI_MODEL` to pin a specific model.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo, branch `main`, file `app.py`.
3. Open **Advanced settings → Secrets** and paste: `GEMINI_API_KEY = "your-key-here"`
4. Click **Deploy**.

## Project structure

```
app.py             # Streamlit app (UI, parsing, Gemini call, scoring)
requirements.txt   # Python dependencies
README.md
```

## Privacy

Resume text is sent to Google's Gemini API for analysis. The app itself does not store
uploaded files or results. Review Google's Gemini API terms before uploading sensitive data.

## License

MITbusiness
