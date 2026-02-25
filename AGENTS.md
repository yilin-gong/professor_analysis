# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

A Python-based web scraping + AI analysis tool (教授研究兴趣分析器) that crawls university websites, extracts professor pages and research interests via a Volcengine/Doubao LLM API, and provides interest-matching with similarity scores. Two interfaces: **Streamlit web UI** (`app.py`, port 8501) and **CLI** (`main.py`).

### Running the application

```bash
streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

The app requires a Volcengine Doubao API key entered at runtime in the sidebar. Without the key the UI loads but analysis features are disabled.

### Running tests

Offline/unit tests (no network or API key required):

```bash
python3 -c "
from test_fixes import test_adaptive_adjustment, test_illinois_pagination_detection, test_clean_faculty_url_preserves_filters
test_adaptive_adjustment()
test_illinois_pagination_detection()
test_clean_faculty_url_preserves_filters()
print('All offline tests passed')
"
```

`test_parameter_recommendations` and `test_link_extraction` in `test_fixes.py` require network access to university websites.

### Important notes

- The `~/.local/bin` directory must be on `PATH` for `streamlit` CLI to be found (packages install there with `--user` flag).
- There is no formal linter configured in this project (no `pyproject.toml`, `setup.cfg`, `flake8`, `ruff`, or similar). Basic Python syntax checks can be done with `python3 -m py_compile <file>`.
- Playwright is an optional dependency for headless browser scraping; `playwright install` is needed if browser-based scraping is used, but is not required for the core Streamlit app.
