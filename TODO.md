AI Digest Agent — TODO
======================

Status Summary
--------------
- Implemented: HAI/FAIR default sources; tz-aware date normalization; TextRank-style summarizer; theme-based fallback clustering; final coverage report; browser-like HTTP headers; site-specific scrapers (OpenAI, xAI, FAIR); AI-only relevance filter across providers and generic scraping.
- Selftest: passed (RSS/Atom parsing, undated tagging).
- Live runs: artifacts in ./digests; some sources still 403/400 (OpenAI News, xAI News, FAIR Blog). Coverage improved with AI-only filter.

Outstanding Issues
------------------
- Site blocks: 403/400 on OpenAI/xAI/FAIR despite browser headers.
- Relevance: Generic fallbacks and broad provider queries still risk edge-case non‑AI items if AI filter disabled.
- Summarizer: LLM path depends on API keys; otherwise extractive TextRank used.

Proposals
---------
- Access/coverage:
  - Provide HTTP(S)_PROXY or run from a network that can fetch blocked sites.
  - Add per-site article parsers (OpenAI/xAI/FAIR/Google AI/BAIR/CSAIL) to extract titles/dates and reduce noise.
  - Optionally scrape JSON‑LD (application/ld+json) for structured article metadata where available.
- Relevance controls:
  - Keep `--filter-ai-only` on by default in scheduled runs.
  - Tighten `AI_FILTER_REGEX` (e.g., require multiple AI terms) or add negative keywords to exclude policy/press-only posts.
  - After enrichment, drop items with very short body text (< 400–600 chars) unless domain whitelisted (e.g., arXiv/OpenReview/Microsoft/Google/DeepMind).
  - Weight or whitelist academic/research hosts (arxiv.org, openreview.net, research.google, deepmind.google) for higher confidence.
- Enrichment & rate limits:
  - Run with enrichment enabled for fuller text (better filtering/summarization) and a higher politeness delay (e.g., 0.4–0.7s per request) if rate‑limited.
  - Add retries with jitter for enrich fetches; cache bodies for 24h to avoid refetch across runs.
- Summaries:
  - Use LLM summaries when `OPENAI_API_KEY` (or Anthropic/Groq) is available; keep `--max-tokens` modest (400–700).

Options
-------
- Strict AI filtering (recommended for noise reduction):
  - `--filter-ai-only` with providers: `--include-openreview --include-arxiv --include-crossref --include-gdelt --include-scholar`
- Faster pulls (fewer requests):
  - `--no-enrich` and smaller `--days` (e.g., 3) and `--max-per-source` (e.g., 25–40)
- Higher quality digests:
  - Enable enrichment; increase `--days`; use LLM summarizer with API key.

Next Actions
------------
1) Provide network access (proxy or different IP) to mitigate 403/400.
2) Decide default for `--filter-ai-only` (on/off) for scheduled runs.
3) Approve adding per‑site article parsers for: OpenAI, xAI, FAIR, Google AI Blog, BAIR, MIT News/CSAIL.
4) Enable LLM with `OPENAI_API_KEY` (or Anthropic/Groq) for higher‑quality summaries.
5) Consider content‑length threshold and host whitelisting during post‑filter.

Handy Commands
--------------
- Selftest (no network):
  - `python Ai-digest-agent.py --selftest`
- Fast run (filtering, no enrichment):
  - `python Ai-digest-agent.py --days 3 --out digests --no-enrich --filter-ai-only`
- Full run with providers + AI filter:
  - `python Ai-digest-agent.py --days 7 --out digests --include-openreview --include-arxiv --include-crossref --include-gdelt --include-scholar --filter-ai-only`
- LLM summaries (requires key):
  - `OPENAI_API_KEY=... python Ai-digest-agent.py --days 7 --out digests --summarizer llm --llm-provider openai --llm-model gpt-4o-mini --max-tokens 700`

