import re
from typing import List, Optional
from urllib.parse import urljoin, urldefrag, urlparse
import json

from bs4 import BeautifulSoup

from .models import LinkCandidate


INCLUDE_TEXT_PAT = re.compile(r"\b(prof|professor|faculty|chair|lecturer|instructor|assistant|associate|dean|webpage|bio|profile)\b", re.I)
INCLUDE_URL_PAT = re.compile(r"/(faculty|people|profiles?|directory|academics|college|department|schools?)/", re.I)

EXCLUDE_TEXT_PAT = re.compile(r"\b(student|alumni|staff|postdoc|administrator|adjunct|phd\s*candidate|phd\s*student|ms\s*student|undergrad)\b", re.I)
EXCLUDE_URL_PAT = re.compile(r"(events|news|publications?|seminars?|courses?|class|projects?|labs?|groups?|staff|students|alumni|\.pdf|\.docx?)", re.I)
CLOUDFLARE_EMAIL = re.compile(r"/cdn-cgi/l/email-protection", re.I)

NAME_LIKE_PAT = re.compile(r"\b([A-Z][a-z]{1,})(\s+[A-Z][a-z]{1,}){1,3}\b")
PROFILE_URL_PATH_PAT = re.compile(r"/people/[^/]+/[^/]+\.html$", re.I)


def _score_anchor(text: str, href: str) -> float:
	score = 0.0
	if INCLUDE_TEXT_PAT.search(text or ""):
		score += 1.0
	if INCLUDE_URL_PAT.search(href or ""):
		score += 0.6
	if NAME_LIKE_PAT.search(text or ""):
		score += 0.9
	if PROFILE_URL_PATH_PAT.search(href or ""):
		score += 1.2
	# Boost score for links that look like personal webpages
	if re.search(r"(dr\.|webpage|biography|bio)\s", text or "", re.I):
		score += 1.5
	# Boost for personal page URLs
	if re.search(r"/people/[^/]+/?$", href or "", re.I):
		score += 1.5
	if EXCLUDE_TEXT_PAT.search(text or ""):
		score -= 1.0
	if EXCLUDE_URL_PAT.search(href or ""):
		score -= 0.8
	if CLOUDFLARE_EMAIL.search(href or ""):
		score = -999.0
	return score


def _get_anchor_text(a) -> str:
	text = (a.get_text(" ") or "").strip()
	label = (a.get("aria-label") or "").strip()
	if not text and label:
		text = label
	if not text:
		img = a.find("img")
		if img and img.get("alt"):
			text = img.get("alt").strip()
	return text


def _add_candidate(links: List[LinkCandidate], url: str, text: str, base_url: str) -> None:
	score = _score_anchor(text, url)
	if score <= 0:
		return
	context = text[:200]
	links.append(LinkCandidate(url=url, text=text, context=context, source_url=base_url, score=score))


def normalize_text(s: str) -> str:
	return re.sub(r"\s+", " ", (s or "")).strip()


def _maybe_add_onclick(el, base_url: str, links: List[LinkCandidate], same_host: str) -> None:
	onclick = el.get('onclick')
	if not onclick:
		return
	m = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick)
	if not m:
		return
	raw = m.group(1)
	abs_url = urljoin(base_url, raw)
	if not abs_url.lower().startswith(("http://", "https://")):
		return
	if urlparse(abs_url).netloc != same_host:
		return
	_add_candidate(links, abs_url, normalize_text(el.get_text(" ") or ""), base_url)


def _extract_from_asu_isearch_react(soup: BeautifulSoup, source_url: str) -> List[LinkCandidate]:
	"""ASU-specific parser for iSearch React component data."""
	candidates = []
	isearch_divs = soup.select('div#clas-isearch-react[data-config]')
	for div in isearch_divs:
		try:
			config = json.loads(div['data-config'])
			if config.get('type') == 'customList' and 'ids' in config:
				ids = config['ids']
				for asurite_id in ids:
					if not asurite_id or not isinstance(asurite_id, str):
						continue
					profile_url = f"https://isearch.asu.edu/profile/{asurite_id}"
					name = asurite_id.replace('.', ' ').title()
					candidates.append(LinkCandidate(
						url=profile_url,
						text=name,
						source_url=source_url
					))
		except (json.JSONDecodeError, KeyError) as e:
			print(f"Warning: Could not parse ASU iSearch config: {e}")
	return candidates


def extract_link_candidates(html: str, source_url: str) -> List[LinkCandidate]:
	soup = BeautifulSoup(html, "lxml")

	# ASU Human Communication specific parser
	if "humancommunication.asu.edu" in source_url:
		asu_candidates = _extract_from_asu_isearch_react(soup, source_url)
		if asu_candidates:
			return asu_candidates

	candidates: List[LinkCandidate] = []
	seen_urls = set()
	base_defrag, _ = urldefrag(source_url)
	base_host = urlparse(source_url).netloc

	for a in soup.find_all("a", href=True):
		text = _get_anchor_text(a)
		href = a["href"].strip()
		abs_url = urljoin(source_url, href)
		if not abs_url.lower().startswith(("http://", "https://")):
			continue
		if urlparse(abs_url).netloc != base_host:
			continue
		abs_defrag, _ = urldefrag(abs_url)
		if abs_defrag == base_defrag:
			continue
		_add_candidate(candidates, abs_url, text, source_url)
		_maybe_add_onclick(a, source_url, candidates, base_host)

	for attr in ["data-url", "data-href", "data-link", "data-profile-url"]:
		for el in soup.find_all(attrs={attr: True}):
			val = el.get(attr)
			if not val:
				continue
			abs_url = urljoin(source_url, val.strip())
			if not abs_url.lower().startswith(("http://", "https://")):
				continue
			if urlparse(abs_url).netloc != base_host:
				continue
			abs_defrag, _ = urldefrag(abs_url)
			if abs_defrag == base_defrag:
				continue
			text = normalize_text(el.get_text(" ")) if el else ""
			_add_candidate(candidates, abs_url, text, source_url)

	candidates.sort(key=lambda c: c.score, reverse=True)
	unique: List[LinkCandidate] = []
	seen = set()
	for c in candidates:
		if c.url in seen:
			continue
		seen.add(c.url)
		unique.append(c)
	return unique
