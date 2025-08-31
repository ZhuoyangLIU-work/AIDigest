AI Digest Agent
===============

Continuously collects recent AI research/news from a fixed set of institutions, deduplicates, groups by topic, and auto‑summarizes into a Markdown digest.

Init
----
- Python: 3.12 recommended
- System deps (Linux/macOS):
  - build-essential (or Xcode CLT on macOS)
  - libomp-dev (for certain ML libs; optional)
- Create venv and install packages:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 lxml python-dateutil tenacity numpy pandas pyyaml
# optional (fallbacks exist if missing)
pip install scikit-learn hdbscan sentence-transformers openai anthropic groq
# tooling (optional)
pip install ruff black mypy
```

Configuration
-------------
- Env vars:
  - SERPAPI_API_KEY (optional) — enables Scholar via SerpAPI
  - HTTP_PROXY / HTTPS_PROXY (optional)

Typical Runs
------------
- Sanity only (no network):
```
python Ai-digest-agent.py --selftest
```
- Fast live run (fewer requests):
```
python Ai-digest-agent.py --days 3 --out digests --no-enrich
```
- Full run with extras + LLM summary:
```
SERPAPI_API_KEY=... python Ai-digest-agent.py \
  --days 7 --out digests --max-per-source 40 \
  --include-openreview --include-arxiv --include-crossref --include-gdelt \
  --summarizer llm --llm-provider openai --llm-model gpt-4o-mini --max-tokens 700
```

Outputs are written to ./digests/ as a Markdown digest and a CSV index. A brief coverage summary is printed at the end.

Note: A lowercase wrapper ai-digest-agent.py is included for convenience on case‑sensitive systems.
