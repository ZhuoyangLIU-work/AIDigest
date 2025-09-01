#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Research Digest Agent (no external RSS library required)
==========================================================

Collects recent work from Stanford, Berkeley (BAIR), MIT (CSAIL), CMU (MLD), Princeton (CS/Princeton NLP),
and from OpenAI, Google/DeepMind, Anthropic, Microsoft Research, Meta FAIR, and xAI; then:
  • normalizes and deduplicates items
  • groups them by themes via embeddings + clustering (with safe fallbacks)
  • generates concise summaries (LLM or extractive fallback)
  • writes a Markdown digest + CSV index

This version removes the `feedparser` dependency (which triggered ModuleNotFoundError) and
parses RSS/Atom using BeautifulSoup (XML parser). It also:
  - Adds optional providers: OpenReview, Google Scholar (via SerpAPI), arXiv, Crossref, GDELT
  - Expands affiliation maps (SAIL, HAI, BAIR, CSAIL, FAIR/Meta, etc.)
  - Provides robust fallbacks if optional ML libs are missing
  - Includes a `--selftest` that validates the feed parser on toy RSS/Atom inputs and checks "undated" tagging
  - Emits a ready-to-edit YAML with major labs' publications pages via `--emit-sources-template`
  - **Adds flags**: `--no-enrich` to skip fetching article bodies; `--max-tokens` to control LLM summary length

Run examples:
  python ai-digest-agent.py --days 7 --out digests --max-per-source 40 \
    --include-openreview --include-arxiv --include-crossref --include-gdelt \
    --summarizer textrank --no-enrich

  SERPAPI_API_KEY=your_key \
  python ai-digest-agent.py --days 3 --include-scholar --summarizer llm --llm-provider openai --max-tokens 600

Notes:
- Respect robots.txt and site ToS when scraping.
- Some sites rotate markup; fetchers are defensive and best-effort.
"""

from __future__ import annotations
import os, re, time, json, argparse, textwrap, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Optional deps & fallbacks -------------------------------------------------
_HAS_ST = False
_HAS_HDBSCAN = False
_HAS_SKLEARN = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    pass
try:
    from hdbscan import HDBSCAN  # type: ignore
    _HAS_HDBSCAN = True
except Exception:
    pass
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import DBSCAN  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    # Minimal fallbacks when sklearn not present
    TfidfVectorizer = None  # type: ignore
    DBSCAN = None  # type: ignore

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Source registry
# ------------------------------------------------------------------------------

DEFAULT_SOURCES: Dict[str, Dict[str, Any]] = {
    # Universities / labs
    "stanford_sail": {
        "kind": "rss",
        "label": "Stanford SAIL Blog",
        "url": "https://ai.stanford.edu/blog/feed.xml",
        "home": "https://ai.stanford.edu/blog/",
        "tags": ["university", "stanford"],
    },
    "berkeley_bair": {
        "kind": "rss_or_scrape",
        "label": "Berkeley BAIR Blog",
        "url": "https://bair.berkeley.edu/blog/",
        "home": "https://bair.berkeley.edu/blog/",
        "tags": ["university", "berkeley"],
    },
    "mit_csail_news": {
        "kind": "rss_or_scrape",
        "label": "MIT News – CSAIL topic",
        "url": "https://news.mit.edu/topic/computer-science-and-artificial-intelligence-laboratory-csail",
        "home": "https://news.mit.edu/topic/computer-science-and-artificial-intelligence-laboratory-csail",
        "tags": ["university", "mit"],
    },
    "cmu_ml_blog": {
        "kind": "rss",
        "label": "CMU ML Blog",
        "url": "https://blog.ml.cmu.edu/feed/",
        "home": "https://blog.ml.cmu.edu/",
        "tags": ["university", "cmu"],
    },
    "princeton_nlp_blog": {
        "kind": "rss_or_scrape",
        "label": "Princeton NLP Blog",
        "url": "https://princeton-nlp.github.io/blog/",
        "home": "https://princeton-nlp.github.io/blog/",
        "tags": ["university", "princeton"],
    },
    # Stanford HAI (research news)
    "stanford_hai_news": {
        "kind": "rss_or_scrape",
        "label": "Stanford HAI — Research News",
        "url": "https://hai.stanford.edu/news/research",
        "home": "https://hai.stanford.edu/news/research",
        "tags": ["university", "stanford", "hai"],
    },

    # Companies / institutes
    "openai_news": {
        "kind": "rss_or_scrape",
        "label": "OpenAI News",
        "url": "https://openai.com/news/",
        "home": "https://openai.com/news/",
        "tags": ["company", "openai"],
    },
    "google_ai_blog": {
        "kind": "rss_or_scrape",
        "label": "Google Blog – AI",
        "url": "https://blog.google/technology/ai/",
        "home": "https://blog.google/technology/ai/",
        "tags": ["company", "google"],
    },
    "google_deepmind": {
        "kind": "scrape",
        "label": "Google DeepMind – Discover/Blog",
        "url": "https://deepmind.google/discover/blog/",
        "home": "https://deepmind.google/discover/blog/",
        "tags": ["company", "deepmind"],
    },
    "anthropic_news": {
        "kind": "scrape",
        "label": "Anthropic – Research",
        "url": "https://www.anthropic.com/research",
        "home": "https://www.anthropic.com/research",
        "tags": ["company", "anthropic"],
    },
    "microsoft_research_blog": {
        "kind": "rss",
        "label": "Microsoft Research Blog",
        "url": "https://www.microsoft.com/en-us/research/feed/",
        "home": "https://www.microsoft.com/en-us/research/blog/",
        "tags": ["company", "microsoft"],
    },
    "xai_news": {
        "kind": "scrape",
        "label": "xAI – News",
        "url": "https://x.ai/news",
        "home": "https://x.ai/news",
        "tags": ["company", "xai"],
    },
    # Meta FAIR — blog
    "meta_fair_blog": {
        "kind": "rss_or_scrape",
        "label": "Meta FAIR — Blog",
        "url": "https://ai.facebook.com/blog/",
        "home": "https://ai.facebook.com/blog/",
        "tags": ["company", "fair", "meta"],
    },
}

# ------------------------------------------------------------------------------
# Utilities & lightweight RSS/Atom parsing
# ------------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

class FetchError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True,
       retry=retry_if_exception_type(FetchError))
def http_get(url: str, timeout: int = 25) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code >= 400:
        raise FetchError(f"HTTP {r.status_code} for {url}")
    return r

def parse_date(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = dateparse.parse(dt_str)
        if not dt:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

@dataclass
class Item:
    source: str
    title: str
    url: str
    published: Optional[datetime]
    summary: str
    content: str
    org: str  # normalized org key

    def key(self) -> str:
        base = (self.title or "") + "|" + (self.url or "")
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

ORG_MAP = {
    "stanford_sail": "stanford",
    "stanford_hai_news": "stanford",
    "berkeley_bair": "berkeley",
    "mit_csail_news": "mit",
    "cmu_ml_blog": "cmu",
    "princeton_nlp_blog": "princeton",
    "openai_news": "openai",
    "google_ai_blog": "google",
    "google_deepmind": "deepmind",
    "anthropic_news": "anthropic",
    "microsoft_research_blog": "microsoft",
    "xai_news": "xai",
    "meta_fair_blog": "fair",
    # external provider tags
    "fair": "fair",
}

# --- Expanded affiliation maps (domains + keywords) ---------------------------
AFFIL_DOMAINS: Dict[str, List[str]] = {
    "stanford": ["stanford.edu", "sail.stanford.edu", "hai.stanford.edu"],
    "berkeley": ["berkeley.edu", "bair.berkeley.edu", "eecs.berkeley.edu"],
    "mit": ["mit.edu", "csail.mit.edu"],
    "cmu": ["cmu.edu", "ml.cmu.edu"],
    "princeton": ["princeton.edu"],
    "openai": ["openai.com"],
    "deepmind": ["deepmind.com"],
    "google": ["google.com", "research.google"],
    "anthropic": ["anthropic.com"],
    "microsoft": ["microsoft.com", "microsoftresearch.com"],
    "xai": ["x.ai"],
    "fair": ["meta.com", "ai.facebook.com", "fb.com", "meta.ai"],
}

AFFIL_KEYWORDS: Dict[str, List[str]] = {
    "stanford": ["Stanford University", "Stanford HAI", "SAIL"],
    "berkeley": ["UC Berkeley", "University of California, Berkeley", "BAIR", "Berkeley AI Research"],
    "mit": ["MIT", "Massachusetts Institute of Technology", "CSAIL"],
    "cmu": ["Carnegie Mellon University", "CMU", "Machine Learning Department"],
    "princeton": ["Princeton University", "Princeton NLP"],
    "openai": ["OpenAI"],
    "deepmind": ["DeepMind", "Google DeepMind"],
    "google": ["Google Research", "Google AI", "Google Brain"],
    "anthropic": ["Anthropic"],
    "microsoft": ["Microsoft Research", "MSR"],
    "xai": ["xAI"],
    "fair": ["FAIR", "Meta AI", "Facebook AI Research"],
}

# ----------------------------- Feed parsing -----------------------------------

def _extract_text(el: Optional[BeautifulSoup]) -> str:
    if not el:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ").strip())

def parse_feed_text(source_key: str, text: str) -> List[Item]:
    soup = BeautifulSoup(text, "xml")
    items: List[Item] = []
    channel = soup.find("channel")
    if channel:
        for it in channel.find_all("item"):
            title = _extract_text(it.find("title"))
            link_el = it.find("link")
            link = (link_el.text or link_el.get("href") or "") if link_el else ""
            pub = parse_date(_extract_text(it.find("pubDate")) or _extract_text(it.find("dc:date")))
            desc = it.find("description")
            content = it.find("content:encoded") or desc
            summary = _extract_text(desc)
            body = _extract_text(content)
            items.append(Item(
                source=source_key, title=title, url=link, published=pub,
                summary=summary[:2000], content=(body or summary)[:8000],
                org=ORG_MAP.get(source_key, source_key)))
        return items
    feed = soup.find("feed")
    if feed:
        for e in feed.find_all("entry"):
            title = _extract_text(e.find("title"))
            link = ""
            for le in e.find_all("link"):
                if le.get("rel") in (None, "alternate") and (le.get("href")):
                    link = le.get("href"); break
            if not link:
                id_el = e.find("id")
                link = id_el.text if id_el else ""
            pub = parse_date(_extract_text(e.find("updated")) or _extract_text(e.find("published")))
            summary_el = e.find("summary")
            content_el = e.find("content") or summary_el
            summary = _extract_text(summary_el)
            body = _extract_text(content_el)
            items.append(Item(
                source=source_key, title=title, url=link, published=pub,
                summary=summary[:2000], content=(body or summary)[:8000],
                org=ORG_MAP.get(source_key, source_key)))
    return items

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_rss(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    return parse_feed_text(source_key, r.text)

# ---------------------- Non-RSS source fetchers -------------------------------

def fetch_princeton_nlp(source_key: str, url: str) -> List[Item]:
    # Try feed.xml first
    try:
        rss_items = fetch_rss(source_key, url.rstrip("/") + "/feed.xml")
        if rss_items:
            return rss_items
    except Exception:
        pass
    # Fallback: scrape listing
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    posts = soup.select("article, .post, .post-list-item")
    items: List[Item] = []
    for p in posts:
        a = p.find("a")
        if not a or not a.get("href"):
            continue
        title = a.get_text(strip=True)
        link = a["href"]
        if link.startswith("/"):
            link = url.rstrip("/") + link
        date_text = None
        date_el = p.find("time")
        if date_el and date_el.get("datetime"):
            date_text = date_el["datetime"]
        elif date_el:
            date_text = date_el.get_text(strip=True)
        items.append(Item(
            source=source_key, title=title, url=link,
            published=parse_date(date_text), summary="", content="",
            org=ORG_MAP[source_key]))
    return items


def fetch_deepmind_blog(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    cards = soup.select("a[href*='/discover/blog/']")
    seen = set(); items: List[Item] = []
    for a in cards:
        href = a.get("href")
        if not href or href in seen:
            continue
        seen.add(href)
        link = href if href.startswith("http") else "https://deepmind.google" + href
        title = a.get_text(" ", strip=True) or "DeepMind Blog Post"
        items.append(Item(source=source_key, title=title, url=link, published=None,
                          summary="", content="", org=ORG_MAP[source_key]))
    return items


def fetch_anthropic(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    cards = soup.select("a[href^='/research/'], a[href^='/news/']")
    items: List[Item] = []; seen = set()
    for a in cards:
        href = a.get("href")
        if not href or href in seen:
            continue
        seen.add(href)
        link = href if href.startswith("http") else "https://www.anthropic.com" + href
        title = a.get_text(" ", strip=True) or "Anthropic Post"
        items.append(Item(source=source_key, title=title, url=link, published=None,
                          summary="", content="", org=ORG_MAP[source_key]))
    return items


def fetch_openai_news(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # Target article card anchors only within main content
    main = soup.find("main") or soup
    cards = main.select("a[href^='/news/']")
    items: List[Item] = []; seen = set()
    for a in cards:
        href = a.get("href")
        if not href or href in seen or href.rstrip("/") == "/news":
            continue
        title = a.get_text(" ", strip=True)
        if not title or len(title) < 6:
            continue
        seen.add(href)
        link = href if href.startswith("http") else "https://openai.com" + href
        items.append(Item(source=source_key, title=title, url=link, published=None,
                          summary="", content="", org=ORG_MAP[source_key]))
    return items


def fetch_xai(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup
    cards = main.select("a[href^='/news/'], a[href^='/blog/']")
    items: List[Item] = []; seen = set()
    for a in cards:
        href = a.get("href")
        if not href or href in seen or href.rstrip("/") in ("/news", "/blog"):
            continue
        title = a.get_text(" ", strip=True)
        if not title or len(title) < 6:
            continue
        seen.add(href)
        link = href if href.startswith("http") else "https://x.ai" + href
        items.append(Item(source=source_key, title=title, url=link, published=None,
                          summary="", content="", org=ORG_MAP[source_key]))
    return items

def fetch_meta_fair(source_key: str, url: str) -> List[Item]:
    r = http_get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup
    # FAIR blog posts typically under /blog/slug
    cards = main.select("a[href^='/blog/']")
    items: List[Item] = []; seen = set()
    for a in cards:
        href = a.get("href")
        if not href or href in seen or href.rstrip("/") == "/blog":
            continue
        title = a.get_text(" ", strip=True)
        if not title or len(title) < 6:
            continue
        seen.add(href)
        link = href if href.startswith("http") else "https://ai.facebook.com" + href
        items.append(Item(source=source_key, title=title, url=link, published=None,
                          summary="", content="", org=ORG_MAP[source_key]))
    return items

# ---------------- Normalization, windowing, dedup -----------------------------

THEME_REGEX = [
    ("LLMs", r"\b(LLM|language model|decoder|transformer|token|inference|prompt|alignment)\b"),
    ("Agents", r"\b(agent|tool-use|planner|auto\w*|loop|delegat)\b"),
    ("Reasoning", r"\b(chain-of-thought|CoT|reason\w+|solve|math|theorem|proof)\b"),
    ("Vision", r"\b(vision|image|video|multimodal|VLM|diffusion|clip)\b"),
    ("Robotics", r"\b(robot|manipulat|locomot|RL\b|reinforcement learning)\b"),
    ("Safety/Alignment", r"\b(safety|alignment|harms|red team|eval|jailbreak|RSP)\b"),
    ("Theory", r"\b(generalization|sample complexity|theory|bounds|proof|approximation)\b"),
]

# Broad AI/ML keyword filter for relevance checks
AI_FILTER_REGEX = re.compile(
    r"\b(artificial intelligence|ai|machine learning|ml|deep learning|neural|neuron|transformer|attention|llm|language model|gpt|bert|clip|diffusion|vision|computer vision|reinforcement learning|rl|policy gradient|self[- ]?supervised|contrastive|embedding|token|inference|fine[- ]?tune|prompt|alignment|safety|reasoning|chain[- ]?of[- ]?thought|cot|multimodal|vlm|speech recognition|nlp|natural language processing|large language model)\b",
    flags=re.I,
)

def is_relevant_item(it: Item) -> bool:
    text = f"{it.title} {it.summary} {it.content}"[:5000]
    return bool(AI_FILTER_REGEX.search(text))

def within_window(item: Item, since: datetime) -> bool:
    # Always include items with no publish date; otherwise enforce date window
    if item.published is None:
        return True
    return item.published >= since

def dedup(items: List[Item]) -> List[Item]:
    seen = {}
    keep: List[Item] = []
    for it in items:
        k = it.key()
        if k in seen:
            continue
        seen[k] = True
        keep.append(it)
    return keep

# ---------------- Enrichment (body & dates) ----------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
def enrich_item(it: Item) -> Item:
    try:
        if not it.url:
            return it
        r = http_get(it.url)
        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"(post|article|content)"))
        text = article.get_text(" ") if article else soup.get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            it.content = text[:12000]
            if not it.summary:
                it.summary = text[:1000]
        if not it.published:
            t = soup.find("time")
            if t and t.get("datetime"):
                it.published = parse_date(t["datetime"]) or it.published
            elif t:
                it.published = parse_date(t.get_text(strip=True)) or it.published
        return it
    except Exception:
        return it

# ---------------- OpenReview & Scholar integrations --------------------------

def _affil_match(text: Optional[str]) -> Optional[str]:
    t = (text or "").lower()
    for org, kws in AFFIL_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in t:
                return org
    for org, doms in AFFIL_DOMAINS.items():
        for d in doms:
            if d.lower() in t:
                return org
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_openreview_recent(since_days: int = 7, limit: int = 200) -> List[Item]:
    base = "https://api.openreview.net"
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - since_days * 24 * 3600 * 1000
    items: List[Item] = []
    notes: List[Dict[str, Any]] = []
    try:
        payload = {"term": "", "content": {}, "group": "all", "limit": limit, "sort": "tmdate:desc"}
        r = requests.post(f"{base}/notes/search", json=payload, timeout=30)
        if r.ok:
            notes = r.json().get('notes', [])
    except Exception:
        notes = []
    if not notes:
        try:
            r = requests.get(f"{base}/notes", params={"details": "invitation,tags,replyCount", "limit": limit, "sort": "tmdate:desc"}, timeout=30)
            if r.ok:
                notes = r.json()
        except Exception:
            notes = []
    for n in notes:
        cdate = n.get('cdate') or n.get('tmdate') or 0
        if cdate and cdate < since_ms:
            continue
        content = n.get('content') or {}
        title = content.get('title') or n.get('forumContent', {}).get('title') or 'OpenReview Paper'
        abstract = content.get('abstract') or ''
        authors = content.get('authors') or []
        authorids = content.get('authorids') or []
        # Fetch author profiles when possible
        profiles: List[Dict[str, Any]] = []
        for aid in authorids:
            if isinstance(aid, str) and aid.startswith('~'):
                try:
                    pr = requests.get(f"{base}/profiles", params={"id": aid}, timeout=20)
                    if pr.ok:
                        data = pr.json()
                        prof = (data.get('profiles') or data.get('results') or [])
                        if isinstance(prof, list):
                            profiles.extend(prof)
                except Exception:
                    pass
        org = None
        # 1) profile -> 2) authors list -> 3) abstract/title
        for p in profiles or []:
            for em in p.get('emailsConfirmed', []) or []:
                org = _affil_match(em)
                if org: break
            if org: break
            for h in (p.get('history') or []):
                inst = h.get('institution') or {}
                txt = " ".join([inst.get('name', ''), inst.get('domain', ''), str(h.get('position', ''))])
                org = _affil_match(txt)
                if org: break
            if org: break
        org = org or _affil_match(" ".join(authors)) or _affil_match(abstract or title)
        if not org:
            continue
        url = f"https://openreview.net/forum?id={n.get('forum', n.get('id'))}"
        pub = datetime.fromtimestamp((cdate or now_ms)/1000, tz=timezone.utc) if cdate else None
        items.append(Item(source="openreview", title=str(title), url=url, published=pub,
                          summary=abstract[:1200], content=abstract[:4000], org=org))
    return items

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_scholar_serpapi(since_days: int = 7, per_org: int = 10, serpapi_key: Optional[str] = None, ai_only: bool = False, org_limit: Optional[int] = None) -> List[Item]:
    key = serpapi_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        return []
    year_low = datetime.now().year if since_days <= 365 else datetime.now().year - 1
    items: List[Item] = []
    items: List[Item] = []
    pairs = list(AFFIL_KEYWORDS.items())
    if org_limit is not None:
        pairs = pairs[: max(0, int(org_limit))]
    for org, kws in pairs:
        base_q = f"{kws[0]} artificial intelligence"
        if ai_only:
            base_q += " OR (machine learning) OR (deep learning) OR (transformer) OR (neural)"
        q = base_q
        params = {"engine": "google_scholar", "q": q, "as_ylo": year_low, "num": per_org, "api_key": key}
        try:
            r = requests.get("https://serpapi.com/search", params=params, timeout=30)
            if not r.ok:
                continue
            data = r.json()
            for res in data.get('organic_results', [])[:per_org]:
                title = res.get('title') or 'Scholar result'
                link = res.get('link') or ''
                snippet = res.get('snippet') or ''
                pub_info = res.get('publication_info') or {}
                year = None
                try:
                    year = int(pub_info.get('year')) if pub_info.get('year') else None
                except Exception:
                    year = None
                pub_dt = datetime(year, 1, 1, tzinfo=timezone.utc) if year else None
                item = Item(source="scholar_serpapi", title=title, url=link, published=pub_dt,
                            summary=snippet[:1200], content=snippet[:4000], org=org)
                if ai_only and not is_relevant_item(item):
                    continue
                items.append(item)
        except Exception:
            continue
    return items

# ---------------- arXiv, Crossref, and GDELT providers -----------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_arxiv_recent(since_days: int = 7, max_results: int = 200, categories: Optional[List[str]] = None) -> List[Item]:
    cats = categories or ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]
    query = " OR ".join([f"cat:{c}" for c in cats])
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending", "start": 0, "max_results": max_results}
    try:
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": HEADERS["User-Agent"]+" arxiv"})
        if not r.ok:
            return []
        soup = BeautifulSoup(r.text, "xml")
        entries = soup.find_all("entry")
        items: List[Item] = []
        since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
        for e in entries:
            title = (e.title.text or "").strip()
            link = e.id.text if e.id else (e.link.get("href") if e.link else "")
            updated = parse_date(e.updated.text if e.updated else None) or parse_date(e.published.text if e.published else None)
            if updated and updated < since_dt:
                continue
            abstract = (e.summary.text or "") if e.summary else ""
            aff_texts = []
            for a in e.find_all("author"):
                aff = a.find("arxiv:affiliation")
                if aff and aff.text:
                    aff_texts.append(aff.text)
            org = None
            for t in (aff_texts or [abstract, title]):
                org = _affil_match(t or "")
                if org: break
            if not org:
                continue
            items.append(Item(source="arxiv", title=title, url=link, published=updated,
                              summary=abstract[:1200], content=abstract[:4000], org=org))
        return items
    except Exception:
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_crossref_recent(since_days: int = 7, per_org: int = 50, ai_only: bool = False) -> List[Item]:
    base = "https://api.crossref.org/works"
    since_dt = (datetime.utcnow() - timedelta(days=since_days)).date().isoformat()
    items: List[Item] = []
    for org, kws in AFFIL_KEYWORDS.items():
        qaff = kws[0]
        params = {"rows": per_org, "filter": f"from-pub-date:{since_dt}", "query.affiliation": qaff,
                  "select": "title,URL,issued,abstract,subject,type", "sort": "issued", "order": "desc"}
        if ai_only:
            params["query.bibliographic"] = "artificial intelligence OR machine learning OR deep learning OR neural OR transformer OR language model"
        try:
            r = requests.get(base, params=params, timeout=30)
            if not r.ok:
                continue
            data = r.json()
            for it in (data.get("message", {}).get("items", []) or []):
                title = (it.get("title") or ["Crossref item"])[0]
                link = it.get("URL") or ""
                issued = it.get("issued", {}).get("date-parts", [[None]])[0]
                pub = None
                if issued and issued[0]:
                    y = issued[0]; m = issued[1] if len(issued) > 1 else 1; d = issued[2] if len(issued) > 2 else 1
                    pub = datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
                abstract = it.get("abstract") or ""
                txt = BeautifulSoup(abstract, "html.parser").get_text(" ")
                item = Item(source="crossref", title=title, url=link, published=pub,
                            summary=txt[:1200], content=txt[:4000], org=org)
                if ai_only and not is_relevant_item(item):
                    continue
                items.append(item)
        except Exception:
            continue
    return items

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def fetch_gdelt_docs(since_days: int = 7, per_org: int = 30, ai_only: bool = False) -> List[Item]:
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    timespan = f"d{since_days}"
    items: List[Item] = []
    for org, doms in AFFIL_DOMAINS.items():
        if not doms:
            continue
        q = " OR ".join([f"site:{d}" for d in doms]) + " AND (AI OR \"artificial intelligence\")"
        if ai_only:
            q += " OR (\"machine learning\") OR (\"deep learning\") OR (transformer) OR (neural) OR (\"language model\")"
        params = {"query": q, "timespan": timespan, "format": "json", "maxrecords": per_org, "sort": "datedesc"}
        try:
            r = requests.get(base, params=params, timeout=30)
            if not r.ok:
                continue
            data = r.json()
            for d in data.get("articles", [])[:per_org]:
                title = d.get("title") or "GDELT article"
                link = d.get("url") or ""
                dt = parse_date(d.get("seendate"))
                snippet = d.get("snippet") or ""
                item = Item(source="gdelt", title=title, url=link, published=dt,
                            summary=snippet[:1200], content=snippet[:4000], org=org)
                if ai_only and not is_relevant_item(item):
                    continue
                items.append(item)
        except Exception:
            continue
    return items

# ---------------- Embedding, clustering, labeling -----------------------------

class Grouper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.has_st = _HAS_ST
        self.model = None
        if self.has_st:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.has_st = False

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.has_st and self.model is not None:
            return np.array(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        if _HAS_SKLEARN and TfidfVectorizer is not None:
            vec = TfidfVectorizer(max_features=4096, ngram_range=(1,2), stop_words='english')
            X = vec.fit_transform(texts)
            # L2 normalize rows
            norms = np.sqrt((X.multiply(X)).sum(axis=1)) + 1e-8
            X = X.multiply(1.0 / norms)
            return X.toarray()
        # ultra-minimal embedding: bag-of-words via hashing (no external deps)
        # This is crude but prevents crashes when sklearn is unavailable.
        vocab = {}
        rows = []
        for t in texts:
            vec = {}
            for w in re.findall(r"[a-z]{2,}", t.lower()):
                h = hash(w) % 1024
                vec[h] = vec.get(h, 0) + 1
            rows.append(vec)
        M = np.zeros((len(rows), 1024), dtype=float)
        for i, v in enumerate(rows):
            for k, val in v.items():
                M[i, k] = val
            n = np.linalg.norm(M[i]) or 1.0
            M[i] /= n
        return M

    def cluster(self, items: List[Item]) -> Dict[int, List[int]]:
        docs = [f"{it.title}. {it.summary or it.content}" for it in items]
        X = self.embed(docs)
        labels: np.ndarray
        if _HAS_HDBSCAN and X.shape[0] >= 5:
            hdb = HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean', cluster_selection_method='eom')  # type: ignore
            labels = hdb.fit_predict(X)
        elif _HAS_SKLEARN and DBSCAN is not None:
            labels = DBSCAN(eps=0.3, min_samples=2, metric='euclidean').fit_predict(X)  # type: ignore
        else:
            # No clustering libs: simple theme bucketing by theme label
            labels = -1 * np.ones(len(items), dtype=int)
            theme_to_label: Dict[str, int] = {}
            next_lbl = 0
            for i, it in enumerate(items):
                text = it.title + " " + (it.summary or it.content)
                found_theme: Optional[str] = None
                for theme, rx in THEME_REGEX:
                    if re.search(rx, text, flags=re.I):
                        found_theme = theme
                        break
                if found_theme is not None:
                    if found_theme not in theme_to_label:
                        theme_to_label[found_theme] = next_lbl
                        next_lbl += 1
                    labels[i] = theme_to_label[found_theme]
        clusters: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(i)
        return clusters

    def label_cluster(self, items: List[Item], idxs: List[int]) -> str:
        texts = [items[i].title + ". " + (items[i].summary or items[i].content) for i in idxs]
        if _HAS_SKLEARN and TfidfVectorizer is not None:
            vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
            X = vec.fit_transform(texts)
            means = np.asarray(X.mean(axis=0)).ravel()
            vocab = np.array(vec.get_feature_names_out())
            top = vocab[means.argsort()[::-1][:4]]
            joined = " ".join(top)
            for theme, rx in THEME_REGEX:
                if re.search(rx, joined, flags=re.I):
                    return f"{theme}: {', '.join(top)}"
            return ", ".join(top)
        # fallback
        joined = " ".join(texts)[:2000]
        for theme, rx in THEME_REGEX:
            if re.search(rx, joined, flags=re.I):
                return theme
        return "Misc"

# ---------------- Summarization (extractive + optional LLM) -------------------

@dataclass
class LLMConfig:
    provider: str = "openai"  # one of {openai, anthropic, groq}
    model: str = "gpt-4o-mini"
    max_tokens: int = 500

def _simple_sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]

def summarize_text_extractive(text: str, sentences: int = 5) -> str:
    sents = _simple_sentence_split(text)
    return " ".join(sents[:max(1, sentences)])

def _hashing_embed_sents(sents: List[str], dim: int = 1024) -> np.ndarray:
    M = np.zeros((len(sents), dim), dtype=float)
    for i, t in enumerate(sents):
        for w in re.findall(r"[a-z]{2,}", t.lower()):
            h = hash(w) % dim
            M[i, h] += 1.0
        n = np.linalg.norm(M[i]) or 1.0
        M[i] /= n
    return M

def summarize_text_textrank(text: str, max_sentences: int = 6) -> str:
    sents = _simple_sentence_split(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)
    try:
        if _HAS_SKLEARN and TfidfVectorizer is not None:
            vec = TfidfVectorizer(max_features=4096, ngram_range=(1,2), stop_words='english')
            X = vec.fit_transform(sents).toarray()
        else:
            X = _hashing_embed_sents(sents, dim=1024)
    except Exception:
        X = _hashing_embed_sents(sents, dim=1024)
    eps = 1e-8
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    S = np.clip(Xn @ Xn.T, 0.0, 1.0)
    np.fill_diagonal(S, 0.0)
    row_sums = S.sum(axis=1, keepdims=True) + eps
    P = S / row_sums
    N = S.shape[0]
    d = 0.85
    r = np.full((N,), 1.0 / N)
    for _ in range(30):
        r = d * (P.T @ r) + (1 - d) / N
    idxs = np.argsort(-r)[:max_sentences]
    idxs_sorted = sorted(idxs.tolist())
    return " ".join([sents[i] for i in idxs_sorted])

def summarize_items(items: List[Item], method: str = "textrank", llm_cfg: Optional[LLMConfig] = None) -> str:
    joined = "\n\n".join([f"- {it.title}: {it.summary or it.content}" for it in items])
    if method == "textrank":
        return summarize_text_textrank(joined, max_sentences=6)
    if method != "llm" or llm_cfg is None:
        return summarize_text_extractive(joined, sentences=6)
    prov = (llm_cfg.provider or "openai").lower()
    prompt = textwrap.dedent(f"""
    You are an expert research editor. Summarize the following items into a crisp digest for busy researchers.
    Group related points, avoid redundancy, and surface concrete contributions, methods, and evaluation settings.
    Use 4–7 bullet points. Keep bullets concise (<= 24 words), and include paper/site names when obvious.

    ITEMS:\n{joined}
    """)
    try:
        if prov == "openai":
            import openai
            client = openai.OpenAI()
            rsp = client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "system", "content": "You are a precise research editor."},
                         {"role": "user", "content": prompt}],
                max_tokens=llm_cfg.max_tokens,
                temperature=0.3,
            )
            return rsp.choices[0].message.content.strip()
        elif prov == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=llm_cfg.model,
                max_tokens=llm_cfg.max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        elif prov == "groq":
            from groq import Groq
            client = Groq()
            rsp = client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "system", "content": "You are a precise research editor."},
                         {"role": "user", "content": prompt}],
                max_tokens=llm_cfg.max_tokens,
                temperature=0.3,
            )
            return rsp.choices[0].message.content.strip()
    except Exception:
        pass
    return summarize_text_extractive(joined, sentences=6)

# ---------------- Digest assembly --------------------------------------------

def make_digest(items: List[Item], clusters: Dict[int, List[int]], grouper: Grouper,
                summarizer: str, llm_cfg: Optional[LLMConfig]) -> str:
    ordered_labels = sorted([lbl for lbl in clusters.keys() if lbl != -1]) + ([-1] if -1 in clusters else [])
    lines: List[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"# AI Research Digest — {today}\n")

    by_org: Dict[str, int] = {}
    for it in items:
        by_org[it.org] = by_org.get(it.org, 0) + 1
    org_line = ", ".join(f"{k}:{v}" for k, v in sorted(by_org.items()))
    lines.append(f"**Sources covered:** {org_line}\n\n")

    for lbl in ordered_labels:
        idxs = clusters[lbl]
        if not idxs:
            continue
        group_items = [items[i] for i in idxs]
        header = grouper.label_cluster(items, idxs) if lbl != -1 else "Miscellaneous / Singles"
        lines.append(f"## {header}\n")
        for it in sorted(group_items, key=lambda x: (x.published or datetime(1970,1,1,tzinfo=timezone.utc)), reverse=True):
            date_str = it.published.strftime("%Y-%m-%d") if it.published else "undated"
            lines.append(f"- **{it.title}** ({it.org}{' · '+date_str if date_str else ''}) — {it.url}")
        summary = summarize_items(group_items, method=summarizer, llm_cfg=llm_cfg)
        if summary:
            lines.append("\n**Cluster summary:**\n")
            lines.append(summary.strip() + "\n")
    return "\n".join(lines) + "\n"

# ---------------- Source loading & fetch orchestration ------------------------

def load_sources(yaml_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if yaml_path and os.path.exists(yaml_path):
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
        base = dict(DEFAULT_SOURCES)
        base.update(user or {})
        return base
    return DEFAULT_SOURCES


def fetch_all(sources: Dict[str, Dict[str, Any]], max_per_source: int, since_days: int,
              org_filter: Optional[List[str]] = None, enrich: bool = True,
              ai_only: bool = False) -> List[Item]:
    since = datetime.now(timezone.utc) - timedelta(days=since_days)
    out: List[Item] = []
    for key, cfg in sources.items():
        org = ORG_MAP.get(key, key)
        if org_filter and org not in org_filter and key not in org_filter:
            continue
        kind = cfg.get("kind")
        url = cfg["url"]
        try:
            if key == "princeton_nlp_blog":
                items = fetch_princeton_nlp(key, url)
            elif key == "google_deepmind":
                items = fetch_deepmind_blog(key, url)
            elif key == "anthropic_news":
                items = fetch_anthropic(key, url)
            elif key == "openai_news":
                items = fetch_openai_news(key, url)
            elif key == "xai_news":
                items = fetch_xai(key, url)
            elif key == "meta_fair_blog":
                items = fetch_meta_fair(key, url)
            elif kind == "rss":
                items = fetch_rss(key, url)
            else:
                # try RSS first then scrape
                try:
                    items = fetch_rss(key, url.rstrip("/") + "/feed/")
                    if not items:
                        items = fetch_rss(key, url.rstrip("/") + "/feed.xml")
                except Exception:
                    items = []
                if not items:
                    r = http_get(url)
                    soup = BeautifulSoup(r.text, "html.parser")
                    items = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        title = a.get_text(" ", strip=True)
                        if not title or len(title) < 6:
                            continue
                        if href.startswith("/"):
                            href = url.rstrip("/") + href
                        # For generic fallback, prefer blog/news/research paths
                        if href.startswith("http") and (not ai_only or re.search(r"/(blog|news|research|article)/", href)):
                            items.append(Item(source=key, title=title, url=href, published=None, summary="", content="", org=org))
            # window and cap per source
            items = [it for it in items if within_window(it, since)]
            items = items[:max_per_source]
            if enrich:
                enriched = []
                for it in items:
                    enriched.append(enrich_item(it))
                    time.sleep(0.2)  # polite
                out.extend(enriched)
            else:
                out.extend(items)
        except Exception as e:
            print(f"[WARN] Failed source {key}: {e}")
    return dedup(out)

# ------------------------- Self-test utilities --------------------------------

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def selftest() -> None:
    rss_sample = """
    <rss version=\"2.0\"><channel>
      <title>Sample RSS</title>
      <item>
        <title>Post A</title>
        <link>https://example.com/a</link>
        <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
        <description>Hello world A</description>
      </item>
      <item>
        <title>Post B</title>
        <link>https://example.com/b</link>
        <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
        <description>Hello world B</description>
      </item>
    </channel></rss>
    """
    atom_sample = """
    <feed xmlns=\"http://www.w3.org/2005/Atom\">
      <title>Example Feed</title>
      <entry>
        <title>Atom A</title>
        <link href=\"https://example.com/atom-a\"/>
        <updated>2024-01-03T00:00:00Z</updated>
        <summary>Atom summary A</summary>
      </entry>
    </feed>
    """
    items_rss = parse_feed_text("stanford_sail", rss_sample)
    items_atom = parse_feed_text("google_ai_blog", atom_sample)
    _assert(len(items_rss) == 2, "RSS should yield 2 items")
    _assert(items_rss[0].title == "Post A", "First RSS title mismatch")
    _assert(items_atom and items_atom[0].title == "Atom A", "Atom parsing failed")

    # Test undated tagging in digest
    undated = Item(source="test", title="No Date", url="https://x/y", published=None, summary="s", content="c", org="stanford")
    dated = Item(source="test", title="With Date", url="https://x/z", published=datetime(2024,1,4,tzinfo=timezone.utc), summary="s", content="c", org="stanford")
    items = [undated, dated]
    grp = Grouper()
    clusters = {0: [0,1]}
    md = make_digest(items, clusters, grp, summarizer="textrank", llm_cfg=None)
    _assert("undated" in md.lower(), "Digest should tag missing dates as 'undated'")

    print("[SELFTEST] feed parsing + undated tagging OK")

# ------------------------------- Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Collect, group, and summarize recent AI research/news from selected orgs.")
    ap.add_argument("--days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--max-per-source", type=int, default=40)
    ap.add_argument("--out", type=str, default="digests")
    ap.add_argument("--orgs", type=str, default="", help="Comma list of orgs or source keys to include")
    ap.add_argument("--summarizer", type=str, choices=["textrank", "llm"], default="textrank")
    ap.add_argument("--llm-provider", type=str, default="openai")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--max-tokens", type=int, default=700, help="LLM summary max tokens (when --summarizer llm)")
    ap.add_argument("--sources-yaml", type=str, default="", help="Optional YAML to extend/override sources")
    ap.add_argument("--include-openreview", action="store_true", help="Also fetch OpenReview and filter by target affiliations")
    ap.add_argument("--include-scholar", action="store_true", help="Also query Google Scholar via SerpAPI (set SERPAPI_API_KEY)")
    ap.add_argument("--scholar-orgs-limit", type=int, default=12, help="Max orgs to query on Scholar per run (one API call per org)")
    ap.add_argument("--serpapi-key", type=str, default="", help="Optional SerpAPI key (or set SERPAPI_API_KEY)")
    ap.add_argument("--include-arxiv", action="store_true", help="Also fetch arXiv (filtered by affiliations when available)")
    ap.add_argument("--include-crossref", action="store_true", help="Also fetch Crossref by affiliation")
    ap.add_argument("--include-gdelt", action="store_true", help="Also fetch GDELT DOC results for lab domains")
    ap.add_argument("--emit-sources-template", action="store_true", help="Write an example YAML of lab publication pages and exit")
    ap.add_argument("--no-enrich", action="store_true", help="Skip fetching article body/date from item URLs")
    ap.add_argument("--selftest", action="store_true", help="Run internal parser tests and exit")
    ap.add_argument("--filter-ai-only", action="store_true", help="Filter collected items to AI/ML-related topics only")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    sources = load_sources(args.sources_yaml or None)

    # Optionally emit a sources template YAML and exit (unless also fetching providers)
    if args.emit_sources_template:
        tmpl = {
            "stanford_hai_publications": {"kind": "rss_or_scrape", "label": "Stanford HAI Publications", "url": "https://hai.stanford.edu/news/research", "home": "https://hai.stanford.edu/news/research", "tags": ["stanford", "hai", "publications"]},
            "stanford_sail_publications": {"kind": "rss_or_scrape", "label": "Stanford SAIL Publications", "url": "https://ai.stanford.edu/publications/", "home": "https://ai.stanford.edu/publications/", "tags": ["stanford", "sail", "publications"]},
            "berkeley_bair_publications": {"kind": "rss_or_scrape", "label": "BAIR Publications", "url": "https://bair.berkeley.edu/publications/", "home": "https://bair.berkeley.edu/publications/", "tags": ["berkeley", "bair", "publications"]},
            "mit_csail_publications": {"kind": "rss_or_scrape", "label": "CSAIL Publications", "url": "https://www.csail.mit.edu/publications", "home": "https://www.csail.mit.edu/publications", "tags": ["mit", "csail", "publications"]},
            "cmu_mld_publications": {"kind": "rss_or_scrape", "label": "CMU MLD Publications", "url": "https://www.ml.cmu.edu/research/index.html", "home": "https://www.ml.cmu.edu/research/index.html", "tags": ["cmu", "mld", "publications"]},
            "princeton_cs_publications": {"kind": "rss_or_scrape", "label": "Princeton CS Publications", "url": "https://www.cs.princeton.edu/research/areas/ai", "home": "https://www.cs.princeton.edu/", "tags": ["princeton", "cs", "publications"]},
            "deepmind_publications": {"kind": "rss_or_scrape", "label": "DeepMind Publications", "url": "https://deepmind.google/research/publications/", "home": "https://deepmind.google/research/publications/", "tags": ["deepmind", "publications"]},
            "google_research_publications": {"kind": "rss_or_scrape", "label": "Google Research Publications", "url": "https://research.google/publications/", "home": "https://research.google/", "tags": ["google", "publications"]},
            "anthropic_research": {"kind": "rss_or_scrape", "label": "Anthropic Research", "url": "https://www.anthropic.com/research", "home": "https://www.anthropic.com/research", "tags": ["anthropic", "publications"]},
            "microsoft_research_publications": {"kind": "rss_or_scrape", "label": "Microsoft Research Publications", "url": "https://www.microsoft.com/en-us/research/publication/", "home": "https://www.microsoft.com/en-us/research/", "tags": ["microsoft", "publications"]},
            "meta_fair_publications": {"kind": "rss_or_scrape", "label": "FAIR Publications", "url": "https://ai.facebook.com/research/publications/", "home": "https://ai.facebook.com/research/", "tags": ["fair", "meta", "publications"]},
            "xai_publications": {"kind": "rss_or_scrape", "label": "xAI News/Blog", "url": "https://x.ai/news", "home": "https://x.ai/news", "tags": ["xai", "publications"]},
        }
        try:
            import yaml
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, "sources_example.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(tmpl, f, sort_keys=True, allow_unicode=True)
            print(f"[OK] Wrote sources template: {path}")
        except Exception as e:
            print(f"[WARN] Could not write template: {e}")
        if not (args.include_openreview or args.include_scholar or args.include_arxiv or args.include_crossref or args.include_gdelt):
            return

    org_filter = [s.strip().lower() for s in (args.orgs or "").split(',') if s.strip()] or None

    items = fetch_all(sources, max_per_source=args.max_per_source, since_days=args.days, org_filter=org_filter, enrich=not args.no_enrich, ai_only=args.filter_ai_only)

    # Optional external providers
    if args.include_openreview:
        try:
            items.extend(fetch_openreview_recent(since_days=args.days, limit=200))
        except Exception as e:
            print(f"[WARN] OpenReview fetch failed: {e}")
    if args.include_scholar:
        try:
            items.extend(fetch_scholar_serpapi(since_days=args.days, per_org=10, serpapi_key=(args.serpapi_key or None), ai_only=args.filter_ai_only, org_limit=args.scholar_orgs_limit))
        except Exception as e:
            print(f"[WARN] Scholar fetch failed: {e}")
    if args.include_arxiv:
        try:
            items.extend(fetch_arxiv_recent(since_days=args.days, max_results=300))
        except Exception as e:
            print(f"[WARN] arXiv fetch failed: {e}")
    if args.include_crossref:
        try:
            items.extend(fetch_crossref_recent(since_days=args.days, per_org=60, ai_only=args.filter_ai_only))
        except Exception as e:
            print(f"[WARN] Crossref fetch failed: {e}")
    if args.include_gdelt:
        try:
            items.extend(fetch_gdelt_docs(since_days=args.days, per_org=30, ai_only=args.filter_ai_only))
        except Exception as e:
            print(f"[WARN] GDELT fetch failed: {e}")

    if not items:
        print("No items fetched. Try increasing --days or check network access.")
        return

    items = dedup(items)
    if args.filter_ai_only:
        items = [it for it in items if is_relevant_item(it)]

    grouper = Grouper()
    clusters = grouper.cluster(items)

    llm_cfg = None
    if args.summarizer == "llm":
        llm_cfg = LLMConfig(provider=args.llm_provider, model=args.llm_model, max_tokens=args.max_tokens)

    digest_md = make_digest(items, clusters, grouper, summarizer=("textrank" if args.summarizer!="llm" else "llm"), llm_cfg=llm_cfg)

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(args.out, f"ai_research_digest_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(digest_md)
    print(f"[OK] Wrote digest: {out_path}")

    df = pd.DataFrame([{ "org": it.org, "source": it.source, "title": it.title, "url": it.url,
                         "published": it.published.isoformat() if it.published else "" } for it in items])
    csv_path = os.path.join(args.out, f"ai_research_items_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote items CSV: {csv_path}")

    # Final coverage summary by org
    by_org: Dict[str, int] = {}
    for it in items:
        by_org[it.org] = by_org.get(it.org, 0) + 1
    org_line = ", ".join(f"{k}:{v}" for k, v in sorted(by_org.items()))
    print(f"[SUMMARY] Items by org: {org_line}")

if __name__ == "__main__":
    main()
