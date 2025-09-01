AI Digest Agent
===============

Continuously collects recent AI research/news from a fixed set of institutions, deduplicates, groups by topic, and auto‑summarizes into a Markdown digest.

Quick Start
-----------
```
# setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install requests beautifulsoup4 lxml python-dateutil tenacity numpy pandas pyyaml

# sanity (no network)
python Ai-digest-agent.py --selftest

# fast live run (filtered, fewer requests)
python Ai-digest-agent.py --days 3 --out digests --no-enrich --filter-ai-only

# full run with providers + optional LLM
SERPAPI_API_KEY=... OPENAI_API_KEY=... \
python Ai-digest-agent.py \
  --days 7 --out digests --max-per-source 40 \
  --include-openreview --include-arxiv --include-crossref --include-gdelt --include-scholar \
  --summarizer llm --llm-provider openai --llm-model gpt-4o-mini --max-tokens 700 \
  --filter-ai-only
```

What It Covers
--------------
- Universities/labs: Stanford (SAIL, HAI), Berkeley (BAIR), MIT (CSAIL topic), CMU (MLD), Princeton (NLP).
- Companies/institutes: OpenAI, Google AI Blog, Google DeepMind, Anthropic (Research), Microsoft Research Blog, Meta FAIR Blog, xAI.
- Optional providers: OpenReview, Google Scholar (via SerpAPI), arXiv, Crossref, GDELT.

Install
-------
- Python: 3.12 recommended
- System deps (Linux/macOS):
  - build-essential (or Xcode CLT on macOS)
  - libomp-dev (optional; used by some ML libs)
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
  - OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY (optional) — enables LLM summarization
  - HTTP_PROXY / HTTPS_PROXY (optional) — useful when certain sites rate-limit or block

CLI Basics
----------
- Window/output: `--days N` (default 7), `--out DIR` (default `digests`), `--max-per-source N`
- Enrichment: `--no-enrich` to skip fetching article bodies/dates (faster, fewer requests)
- Summarizer: `--summarizer {textrank|llm}`; `--llm-provider {openai|anthropic|groq}`; `--llm-model`; `--max-tokens`
- Providers: `--include-{openreview|scholar|arxiv|crossref|gdelt}`
- Sources: `--sources-yaml PATH` to extend/override; `--emit-sources-template` to write an example YAML
- Filter: `--filter-ai-only` to drop items weakly related to AI/ML
- Selftest: `--selftest` (no network)

Typical Runs
------------
- Sanity only (no network):
```
python Ai-digest-agent.py --selftest
```
- Fast live run (fewer requests):
```
python Ai-digest-agent.py --days 3 --out digests --no-enrich --filter-ai-only
```
- Full run with extras + LLM summary:
```
SERPAPI_API_KEY=... OPENAI_API_KEY=... \
python Ai-digest-agent.py \
  --days 7 --out digests --max-per-source 40 \
  --include-openreview --include-arxiv --include-crossref --include-gdelt --include-scholar \
  --summarizer llm --llm-provider openai --llm-model gpt-4o-mini --max-tokens 700 \
  --filter-ai-only
```

Outputs
-------
- Markdown digest and CSV index under `./digests/` with timestamped filenames.
- A brief per-organization coverage summary prints at the end of the run.

GitHub Pages & Schedule
-----------------------
- This repo includes a workflow at `.github/workflows/digest.yml` that:
  - Runs every 3 days (and on manual dispatch).
  - Builds digests with `--days 3 --no-enrich --filter-ai-only`.
  - Publishes a static site with digests to GitHub Pages.
- To enable:
  1) In GitHub repo Settings → Pages, ensure Source is set to “GitHub Actions”.
  2) (Optional) Add secrets for better coverage:
     - `SERPAPI_API_KEY` (Scholar provider)
     - `OPENAI_API_KEY` (LLM summarizer)
     - `HTTP_PROXY` / `HTTPS_PROXY` if your network requires it
  3) Trigger the workflow via the Actions tab (or wait for the schedule).

Extending Sources
-----------------
- Write a template of common lab publication pages:
```
python Ai-digest-agent.py --emit-sources-template --out digests
```
- Extend/override defaults with your own YAML:
```
python Ai-digest-agent.py --sources-yaml path/to/sources.yaml
```

Noise Reduction Tips
--------------------
- Use `--filter-ai-only` to drop weakly related items (applies to providers and generic scraping).
- Keep `--no-enrich` for speed; enable enrichment for fuller text that improves filtering and summaries.
- Consider provider selection: arXiv/OpenReview are highly relevant; Crossref/GDELT/Scholar benefit most from filtering.

Troubleshooting
---------------
- 403/400 on some sites (OpenAI/xAI/FAIR):
  - The agent sends browser-like headers, but some endpoints still block. Try `HTTP_PROXY`/`HTTPS_PROXY` or run from a network with broader access.
- LLM summaries not appearing:
  - Ensure the appropriate API key is set and the provider library is installed (e.g., `pip install openai`). Fallback is extractive TextRank.

Notes
-----
- Lowercase wrapper `ai-digest-agent.py` may not be necessary on macOS; use `Ai-digest-agent.py`.
