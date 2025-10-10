from __future__ import annotations

import re
from typing import Optional
from bs4 import BeautifulSoup

from .models import ProfileResult, ContactInfo, LlmConfig
from .utils import normalize_whitespace, extract_emails, extract_mailto_emails_from_html, select_professor_emails, keywords_from_interest
from .llm import LlmClient, refine_profile_info


PROF_TITLE_PAT = re.compile(r"\b(assistant|associate|full|distinguished|adjunct|visiting|teaching)?\s*professor\b", re.I)
NOT_PROF_PAT = re.compile(r"\b(student|alumni|staff|administrator|coordinator|manager|directory|faculty)\b", re.I)
ACCEPT_YES_PAT = re.compile(r"\b(looking for|recruiting|accepting|seek|seeking|welcome|welcomes|hiring|taking)\b.*\b(phd|doctoral|graduate students?|ph\.?d)\b", re.I)
ACCEPT_NO_PAT = re.compile(r"\b(not\s+accepting|no\s+openings|no\s+positions|not\s+looking|closed|no\s+longer\s+accepting)\b.*\b(phd|students?)\b", re.I)
NAME_LIKE_PAT = re.compile(r"\b([A-Z][a-z]{1,})(\s+[A-Z][a-z]{1,}){1,3}\b")

STOP_PHRASES = (
	"U-M // LSA",
	"U-M LSA",
	"Departments and Units",
	"Majors and Minors",
	"College of LSA",
	"Look To Michigan",
)


def _first_heading_text(soup: BeautifulSoup) -> Optional[str]:
	for tag in ["h1", "h2", "h3"]:
		elem = soup.find(tag)
		if elem:
			return normalize_whitespace(elem.get_text(" "))
	return None


def _page_title(soup: BeautifulSoup) -> Optional[str]:
	t = soup.find("title")
	return normalize_whitespace(t.get_text(" ")) if t else None


def _prune_non_content(soup: BeautifulSoup) -> BeautifulSoup:
	for sel in ["script", "style", "nav", "header", "footer", "aside"]:
		for el in soup.select(sel):
			el.decompose()
	return soup


def _choose_content_root(soup: BeautifulSoup) -> BeautifulSoup:
	for sel in ["main", "article", "#content", "#content-column", ".content", ".profile", "#main-content", ".page-content"]:
		el = soup.select_one(sel)
		if el:
			return el
	return soup


def _extract_research_interest(soup: BeautifulSoup) -> Optional[str]:
	# Special handling for UDel faculty pages
	if "udel.edu" in str(soup):
		# UDel pages have a lot of navigation - be more selective
		main_content = soup.select_one("main, .main-content, #main-content")
		if main_content:
			# Look for specific content sections, avoiding navigation
			content_text = main_content.get_text(" ")
			# Remove common navigation patterns
			nav_patterns = [
				r"Skip to site content.*?Make a Gift",
				r"About Us.*?Research Programs",
				r"Communication.*?Our Alumni",
				r"Resources and Links.*?Citations",
			]
			for pattern in nav_patterns:
				content_text = re.sub(pattern, "", content_text, flags=re.DOTALL)
			
			content_text = normalize_whitespace(content_text)
			
			# Look for actual research content patterns
			research_patterns = [
				r"(my research[^.]*\.)",
				r"(i focus on[^.]*\.)",
				r"(research interests?[^.]*\.)",
				r"(my work[^.]*\.)",
				r"(specializ[a-z]* in[^.]*\.)",
			]
			
			for pattern in research_patterns:
				match = re.search(pattern, content_text, re.I)
				if match:
					result = match.group(1).strip()
					if len(result) > 20 and "communication" not in result.lower()[:50]:  # Avoid nav text
						return result
		
		# If no specific research content found, return None rather than navigation
		return None
	
	# Prune layout chrome then focus on the main content region for other sites
	soup = _prune_non_content(soup)
	root = _choose_content_root(soup)

	# Prefer explicit sections
	candidates = []
	for hdr in root.find_all(["h1", "h2", "h3", "h4", "strong", "b" ]):
		text = normalize_whitespace(hdr.get_text(" "))
		if re.search(r"\bresearch(\s+interests?|areas?|focus|topics?)\b", text, re.I):
			parts = []
			for sib in hdr.find_all_next(["p", "li", "dd"], limit=10):
				paragraph = normalize_whitespace(sib.get_text(" "))
				if paragraph:
					parts.append(paragraph)
					if len(" ".join(parts)) > 800:
						break
			if parts:
				candidates.append(" ".join(parts))

	# Also support definition lists like "Research Interests: ..."
	for dl in root.find_all("dl"):
		dts = dl.find_all("dt")
		dds = dl.find_all("dd")
		for dt, dd in zip(dts, dds):
			if re.search(r"\bresearch(\s+interests?|areas?|focus|topics?)\b", normalize_whitespace(dt.get_text(" ")), re.I):
				val = normalize_whitespace(dd.get_text(" "))
				if val:
					candidates.append(val)

	# Clean and filter candidates
	cleaned = []
	for c in candidates:
		c2 = c
		for s in STOP_PHRASES:
			c2 = c2.replace(s, " ")
		c2 = normalize_whitespace(c2)
		if c2:
			cleaned.append(c2)
	if cleaned:
		return cleaned[0][:800]

	# Fallback: personal-statement style sentences only
	text = normalize_whitespace(root.get_text(" "))
	sentences = re.split(r"(?<=[.!?])\s+", text)
	for s in sentences:
		if re.search(r"\b(my|i|our)\s*(?:current\s*)?research\b", s, re.I) or re.search(r"\b(i\s+study|i\s+focus\s+on|my\s+interests?\s+include)\b", s, re.I):
			s2 = s
			for sp in STOP_PHRASES:
				s2 = s2.replace(sp, " ")
			s2 = normalize_whitespace(s2)
			if s2 and len(s2) >= 12:
				return s2[:500]

	for header in soup.find_all(['h2', 'h3', 'h4']):
		header_text = header.get_text().strip().lower()
		if any(keyword in header_text for keyword in ['research', 'interests', 'expertise', 'specialization']):
			# Found a likely header, now grab the text that follows
			content = []
			for sibling in header.find_next_siblings():
				if sibling.name in ['h2', 'h3', 'h4']:
					break  # Stop at the next header
				if sibling.name in ['p', 'ul', 'div']:
					content.append(normalize_whitespace(sibling.get_text(" ")))
			if content:
				return " ".join(content)

	# Fallback: if no explicit section, look for relevant paragraphs in the main content
	main_content = soup.select_one("main, #main, #content, .main-content, body")
	if main_content:
		text = normalize_whitespace(main_content.get_text(" "))
		# Look for a sentence that contains "research" and is reasonably long
		match = re.search(r"([^.]*?research[^.]*\.)", text, re.I)
		if match:
			return match.group(1).strip()
		
		# Second fallback for short bios like KU's
		bio_match = re.search(r"biography|about", text, re.I)
		if bio_match:
			# Get text around the bio match
			start_index = bio_match.start()
			short_text = text[start_index : start_index + 300]
			return short_text


	return ""


UNIV_CAND_META_KEYS = [
	{"property": "og:site_name"},
	{"name": "application-name"},
	{"name": "site_name"},
]
UNIV_PATTERNS = [
	re.compile(r"\b(University of [A-Z][A-Za-z&'\-\s]+)\b"),
	re.compile(r"\b([A-Z][A-Za-z&'\-\s]+ University)\b"),
	re.compile(r"\b([A-Z][A-Za-z&'\-\s]+ Institute(?: of Technology)?)\b"),
	re.compile(r"\b([A-Z][A-Za-z&'\-\s]+ College)\b"),
	re.compile(r"\b([A-Z][A-Za-z&'\-\s]+ Polytechnic)\b"),
	re.compile(r"\b([A-Z][A-Za-z&'\-\s]+ Academy)\b"),
]


def _infer_university_name(soup: BeautifulSoup, title_text: str, full_text: str) -> Optional[str]:
	for key in UNIV_CAND_META_KEYS:
		m = soup.find("meta", attrs=key)
		if m and m.get("content"):
			val = normalize_whitespace(m.get("content"))
			for pat in UNIV_PATTERNS:
				mm = pat.search(val)
				if mm:
					return mm.group(1)
			if any(tok.lower() in val.lower() for tok in ["university","college","institute","polytechnic","academy"]):
				return val
	for sel in [".site-name", ".site-title", ".navbar-brand", "header .brand", "#site-name", "#logo", "header .site-title"]:
		el = soup.select_one(sel)
		if el:
			val = normalize_whitespace(el.get_text(" "))
			for pat in UNIV_PATTERNS:
				mm = pat.search(val)
				if mm:
					return mm.group(1)
	for t in [title_text, full_text]:
		for pat in UNIV_PATTERNS:
			mm = pat.search(t)
			if mm:
				return mm.group(1)
	return None


PROG_PATTERNS = [
	re.compile(r"\b(Department of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(School of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(College of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(Faculty of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(Institute of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(Program in\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
	re.compile(r"\b(Division of\s+[A-Z][A-Za-z&'\-\s]{2,})\b"),
]


def _infer_program_name(soup: BeautifulSoup, title_text: str, head_text: str, full_text: str) -> Optional[str]:
	for sel in [".breadcrumb li", "nav.breadcrumb li", "#breadcrumbs li", "header h1", "header h2", "main h1", "main h2"]:
		el = soup.select_one(sel)
		if el:
			val = normalize_whitespace(el.get_text(" "))
			for pat in PROG_PATTERNS:
				mm = pat.search(val)
				if mm:
					return mm.group(1)
	for t in [title_text, head_text, full_text]:
		for pat in PROG_PATTERNS:
			mm = pat.search(t)
			if mm:
				return mm.group(1)
	return None


def heuristic_analyze(
	url: str,
	html: str,
	university: Optional[str] = None,
	program: Optional[str] = None,
	llm_config: Optional[LlmConfig] = None,
) -> Optional[ProfileResult]:
	soup = BeautifulSoup(html, "lxml")
	full_text = normalize_whitespace(soup.get_text(" "))
	
	h1_text = soup.h1.get_text(" ") if soup.h1 else ""
	title_text = soup.title.get_text(" ") if soup.title else ""
	
	uni_val = university or _infer_university_name(soup, title_text, full_text)
	prog_val = program or _infer_program_name(soup, title_text, h1_text, full_text)
	
	is_prof_signal = bool(PROF_TITLE_PAT.search(h1_text) or PROF_TITLE_PAT.search(title_text) or re.search(r"\b(professor|faculty)\b", full_text, re.I))
	not_prof_signal = bool(NOT_PROF_PAT.search(h1_text) or NOT_PROF_PAT.search(title_text))
	is_professor_page = is_prof_signal and not not_prof_signal

	name = normalize_whitespace(h1_text) if h1_text else None
	title = normalize_whitespace(title_text) if title_text else None

	if llm_config and llm_config.enabled:
		# Use LLM to refine name and title from the header or start of text
		refinement_text = h1_text
		if len(refinement_text) < 100 and len(full_text) > len(refinement_text):
			refinement_text += "\n" + full_text[:500] # Add first 500 chars of body
		
		llm_result = refine_profile_info(llm_config, refinement_text)
		if llm_result:
			llm_name = llm_result.get("professor_name")
			llm_title = llm_result.get("title")
			if llm_name:
				name = llm_name
			if llm_title:
				title = llm_title


	emails = list({*extract_mailto_emails_from_html(html), *extract_emails(full_text)})
	# Filter emails to find the most likely one for the professor
	selected_emails = select_professor_emails(name, emails, limit=1)

	research = _extract_research_interest(soup)
	keywords = keywords_from_interest(research or "")

	# Stricter check: a profile page should have research interests
	if not research and len(full_text) > 1000:  # Don't penalize very short pages
		is_professor_page = False

	accepting = None
	if ACCEPT_YES_PAT.search(full_text):
		accepting = True
	elif ACCEPT_NO_PAT.search(full_text):
		accepting = False

	return ProfileResult(
		university_name=uni_val,
		program_name=prog_val,
		professor_name=name,
		title=title,
		contact_info=ContactInfo(emails=selected_emails),
		research_interest=research,
		keywords=keywords,
		profile_url=url,
		is_professor_page=is_professor_page,
		is_accepting_phd=accepting,
		source="heuristic",
	)


def refine_with_llm(cfg: LlmConfig, url: str, html_text: str, result: ProfileResult) -> ProfileResult:
	client = LlmClient(cfg)
	if not client.enabled:
		return result
	text = normalize_whitespace(BeautifulSoup(html_text, "lxml").get_text(" "))
	data = client.analyze_profile(url, text)
	if not data:
		return result
	result.is_professor_page = bool(data.get("is_professor_page", result.is_professor_page))
	result.professor_name = data.get("professor_name") or result.professor_name
	result.title = data.get("title") or result.title
	acc = data.get("is_accepting_phd")
	if isinstance(acc, str):
		acc_l = acc.lower()
		result.is_accepting_phd = True if acc_l == "yes" else False if acc_l == "no" else result.is_accepting_phd
	research = data.get("research_interest")
	if research:
		result.research_interest = research
		result.keywords = keywords_from_interest(research)
	em = data.get("emails") or []
	if em:
		result.contact_info.emails = list({*result.contact_info.emails, *em})
	result.source = "mixed"
	return result
