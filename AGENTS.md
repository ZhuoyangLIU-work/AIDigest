# AI Digest Agent — Operations Guide (AGENTS.md)

## Purpose
Continuously collect recent AI research/news from a fixed set of institutions (Stanford/SAIL/HAI, Berkeley/BAIR, MIT/CSAIL, CMU/MLD, Princeton/NLP, OpenAI, Google/DeepMind, Anthropic, Microsoft Research, Meta FAIR, xAI), deduplicate, group by topic, and auto-summarize into a Markdown digest.

## Agent Goals
1) **Setup** a Python 3.12 env with required packages.
2) **Review** the repository for correctness, missing deps, and obvious runtime issues.
3) **Run** internal tests (`--selftest`) and then a live pull with safe defaults.
4) **Publish** artifacts: Markdown digest and CSV index in `./digests/`.
5) **Report** findings and next steps (failures, rate limit warnings, unreachable sources).

## Capabilities & Tools
- **Shell**: install system deps (`build-essential`, `libomp-dev`), set env vars.
- **Python**: run scripts and small probes.
- **Editor**: non-invasive edits (formatting, comments). Do **not** change logic unless asked.

## Initialization
Run the `/init` from the README (or this document). It:
- Installs **runtime deps**: `requests`, `beautifulsoup4`, `lxml`, `python-dateutil`, `tenacity`, `numpy`, `pandas`, `pyyaml`.
- Installs **optional deps**: `scikit-learn`, `hdbscan`, `sentence-transformers`, `openai`, `anthropic`, `groq` (the code has fallbacks).
- Installs **tooling**: `ruff`, `black`, `mypy` for static checks.
- Executes a **non-network** `--selftest` to validate the RSS/Atom parser and “undated” tagging.

## Configuration
Environment variables:
- `SERPAPI_API_KEY` (optional) — enables Scholar via SerpAPI.
- `HTTP_PROXY` / `HTTPS_PROXY` (optional) — if your network requires it.

Command-line flags (key ones):
- `--days N`    : lookback window (default 7)
- `--out DIR`   : output directory (default `digests`)
- `--max-per-source N` : cap items/source before enrichment
- `--no-enrich` : skip fetching article bodies/dates (faster, fewer requests)
- `--summarizer {textrank|llm}` : summary backend (default `textrank`)
- `--llm-provider {openai|anthropic|groq}` and `--llm-model`
- `--max-tokens N` : LLM summary token cap
- `--include-{openreview|scholar|arxiv|crossref|gdelt}` : more providers
- `--emit-sources-template` : write lab publication YAML to `--out`
- `--sources-yaml PATH` : extend/override default sources
- `--selftest` : internal tests (no network)

## Typical Runs
- **Sanity only (no network)**  
  `python ai-digest-agent.py --selftest`

- **Fast live run (no enrichment)**  
  `python ai-digest-agent.py --days 3 --out digests --no-enrich`

- **Full run with extra providers + LLM summary**  
