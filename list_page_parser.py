from __future__ import annotations

import re
import json
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, ParseResult, urljoin

from bs4 import BeautifulSoup, FeatureNotFound

from .models import ProfileResult, ContactInfo
from .utils import normalize_whitespace, extract_mailto_emails_from_html, extract_emails, select_professor_emails, keywords_from_interest

NAME_LIKE_PAT = re.compile(r"\b([A-Z][a-zà-ÿ'-]{1,})(\s+[A-Z][a-zà-ÿ'-]{1,}){1,3}\b")
PROF_TITLE_PAT = re.compile(r"\b(assistant|associate|full|distinguished|adjunct|visiting|teaching)?\s*professor\b", re.I)


def _choose_content_root(soup: BeautifulSoup) -> BeautifulSoup:
	for sel in ["main", "article", "#content", "#content-column", ".content", "#block-mainpagecontent", ".region-content", "#region-content"]:
		el = soup.select_one(sel)
		if el:
			return el
	return soup


def _normalize_url_no_fragment(u: str) -> str:
	pr = urlparse(u)
	clean = ParseResult(pr.scheme, pr.netloc, pr.path, pr.params, pr.query, "")
	return urlunparse(clean)


def _slugify(text: str) -> str:
	text = normalize_whitespace(text).lower()
	text = re.sub(r"[^a-z0-9\-\s]", "", text)
	text = re.sub(r"\s+", "-", text).strip("-")
	return text[:64] if text else "entry"


def _is_name(text: str) -> bool:
	"""Check if a string looks like a person's name."""
	if not text or len(text) > 100:
		return False
	if "," in text and len(text) <= 120:
		return True
	return bool(NAME_LIKE_PAT.search(text))


def _detect_syracuse(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	"""Detect the Syracuse Newhouse React directory by its component markup."""
	if soup.select_one('[data-component*="faculty"]'):
		return True
	if soup.select_one('[data-react-class*="Faculty"]'):
		return True
	return bool(soup.select_one('section[class*="faculty-card"], div[class*="faculty-card"]'))


def _extract_from_syracuse(soup: BeautifulSoup, base_url: str) -> List[ProfileResult]:
	"""Syracuse Newhouse-specific parser that handles their React-based faculty page."""
	profiles = []
	
	# Syracuse faculty pages have specific structure after React loads
	# Look for faculty sections and cards
	faculty_containers = soup.find_all(['div', 'section', 'article'])
	
	for container in faculty_containers:
		# Look for professor names and titles in the container
		text_content = container.get_text(" ")
		
		# Check if this container has faculty information
		if not any(term in text_content.lower() for term in ['professor', 'assistant', 'associate', 'instructor']):
			continue
			
		# Extract name and title - Syracuse seems to have specific patterns
		lines = [line.strip() for line in text_content.split('\n') if line.strip()]
		
		name = None
		title = None
		department = None
		
		for i, line in enumerate(lines):
			# Look for title keywords to identify professor lines
			if any(term in line.lower() for term in ['professor', 'assistant', 'associate', 'instructor']):
				# The name is likely in the previous line or at the start of this line
				if i > 0 and _is_name(lines[i-1]):
					name = lines[i-1].strip()
					title = line.strip()
				elif _is_name(line.split()[0] if line.split() else ''):
					# Name and title might be on the same line
					parts = line.split()
					possible_name = ' '.join(parts[:2])  # First two words as name
					if _is_name(possible_name):
						name = possible_name
						title = ' '.join(parts[2:]) if len(parts) > 2 else line
				
				# Look for department in following lines
				if i + 1 < len(lines):
					next_line = lines[i + 1].strip()
					if next_line and 'view bio' not in next_line.lower():
						department = next_line
				break
		
		if name and title:
			# Look for profile links within this container
			profile_url = base_url
			links = container.find_all('a', href=True)
			for link in links:
				if 'bio' in link.get_text().lower() or '/faculty/' in link['href']:
					from urllib.parse import urljoin
					profile_url = urljoin(base_url, link['href'])
					break
			
			# Extract email if present
			email = None
			email_pattern = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text_content)
			if email_pattern:
				email = email_pattern.group(1)
			
			profiles.append(ProfileResult(
				university_name="Syracuse University",
				program_name=department or "Newhouse School",
				professor_name=name,
				title=title,
				profile_url=profile_url,
				contact_info=ContactInfo(emails=[email] if email else []),
				is_professor_page=True,
				source="heuristic"
			))
	
	return profiles


def _detect_udel(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	return bool(root.select_one('li img[alt^="Headshot photo of"]'))


def _extract_from_udel(soup: BeautifulSoup, base_url: str) -> List[ProfileResult]:
	"""UDel-specific parser that handles the card-based faculty page."""
	profiles = []
	
	# Find all faculty cards by looking for the specific UDel structure
	faculty_cards = soup.find_all('li')
	
	for card in faculty_cards:
		# Look for headshot image and name structure
		headshot = card.find('img', alt=re.compile(r'Headshot photo of', re.I))
		if not headshot:
			continue
			
		# Extract name from the alt text
		alt_text = headshot.get('alt', '')
		name_match = re.search(r'Headshot photo of (.+)', alt_text)
		if not name_match:
			continue
		name = name_match.group(1).strip()
		
		# Find title within the card
		title = "Faculty"
		title_patterns = [
			r'<strong>([^<]*(?:Professor|Chair|Instructor|Director)[^<]*)</strong>',
			r'<strong>([^<]*)</strong>'
		]
		card_html = str(card)
		for pattern in title_patterns:
			title_match = re.search(pattern, card_html, re.I)
			if title_match and any(keyword in title_match.group(1).lower() for keyword in ['professor', 'chair', 'instructor', 'director']):
				title = title_match.group(1).strip()
				break
		
		# Extract email
		email_link = card.find('a', href=re.compile(r'^mailto:', re.I))
		email = None
		if email_link:
			email = email_link['href'].replace('mailto:', '').strip()
		
		# Create profile URL - UDel faculty pages might have individual pages
		profile_url = base_url  # Default to the listing page
		name_link = card.find('a', href=True)
		if name_link and name_link.get('href') and not name_link['href'].startswith('mailto:'):
			from urllib.parse import urljoin
			profile_url = urljoin(base_url, name_link['href'])
		
		profiles.append(ProfileResult(
			university_name="University of Delaware",
			program_name="Communication",
			professor_name=name,
			title=title,
			profile_url=profile_url,
			contact_info=ContactInfo(emails=[email] if email else []),
			is_professor_page=True,
			source="heuristic"
		))
	
	return profiles


def _detect_lsu(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	return bool(root.select_one('table.scholarships'))


def _extract_from_lsu(soup: BeautifulSoup, base_url: str) -> List[ProfileResult]:
	"""LSU-specific parser that handles the table-based faculty page."""
	profiles = []
	table = soup.find('table', class_='scholarships')
	if not table:
		return []

	for row in table.find_all('tr'):
		cell = row.find('td')
		if not cell:
			continue

		name_text = ''
		link_url = None

		# Find name and link. The name is usually in a link within a span with a specific style.
		name_span = cell.select_one('span[style*="font-size: 18px"]')
		if name_span:
			if name_span.find('a'):
				name_text = normalize_whitespace(name_span.find('a').get_text(" "))
				link_url = name_span.find('a')['href']
			else:
				name_text = normalize_whitespace(name_span.get_text(" "))

		if not name_text:
			# Fallback for names not in a styled span but in a link
			name_link = cell.select_one('a[href*="/faculty-staff/"]')
			if name_link:
				name_text = normalize_whitespace(name_link.get_text(" "))
				link_url = name_link['href']

		if not name_text or not _is_name(name_text):
			continue

		# Find title from the rest of the cell's text
		all_text = normalize_whitespace(cell.get_text(" "))
		title_text = all_text.replace(name_text, '').strip()

		# Clean up title by removing contact info
		title_text = re.sub(r'(Office|Phone|Email):.*', '', title_text, flags=re.I).strip()

		if link_url:
			profile_url = urljoin(base_url, link_url)
		else:
			slug = _slugify(name_text) or f"entry-{len(profiles)}"
			profile_url = f"{_normalize_url_no_fragment(base_url)}#faculty-{slug}"

		profiles.append(ProfileResult(
			university_name="Louisiana State University",
			program_name="Manship School of Mass Communication",
			professor_name=name_text,
			title=title_text,
			profile_url=profile_url,
			is_professor_page=True, # It's a link to a profile
			source="heuristic",
		))

	return profiles


def _detect_uky(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	for header in root.find_all('h3'):
		if not header.get_text(strip=True):
			continue
		next_el = header.find_next_sibling()
		while next_el and next_el.name not in ('h3', 'hr'):
			if next_el.find('strong'):
				return True
			next_el = next_el.find_next_sibling()
	return False


def _extract_from_uky(soup: BeautifulSoup, base_url: str, university: Optional[str], program: Optional[str]) -> List[ProfileResult]:
	"""UKY-specific parser that handles linkless profiles on the main directory page."""
	profiles = []
	main_content = soup.select_one('main, .main-content, #content')
	if not main_content:
		main_content = soup

	department_headers = main_content.find_all('h3')
	
	for header in department_headers:
		department_name = normalize_whitespace(header.get_text(" "))
		if "department" not in department_name.lower() and "school" not in department_name.lower() and "dean's office" not in department_name.lower():
			continue

		for sibling in header.find_next_siblings():
			if sibling.name == 'h3' or sibling.name == 'hr':
				break
			
			name_el = sibling.find('strong')
			if not name_el:
				continue
				
			name_text = normalize_whitespace(name_el.get_text(" "))
			if not _is_name(name_text):
				continue
			
			title_text = normalize_whitespace(sibling.get_text(" ")).replace(name_text, "").strip()

			slug = _slugify(name_text) or f"entry-{len(profiles)}"
			profile_url = f"{_normalize_url_no_fragment(base_url)}#faculty-{slug}"
			
			profiles.append(ProfileResult(
				university_name="University of Kentucky",
				program_name=department_name,
				professor_name=name_text,
				title=title_text,
				profile_url=profile_url,
				is_professor_page=True,
				source="heuristic",
			))
			
	return profiles


def _detect_unm(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	container = soup.select_one('.main-content .col-sm-12')
	if not container:
		return False
	return bool(container.find('h3'))


def _extract_from_unm(soup: BeautifulSoup, base_url: str, university: Optional[str], program: Optional[str]) -> List[ProfileResult]:
	"""UNM-specific parser that handles linkless profiles on the main directory page."""
	profiles = []
	main_content = soup.select_one('.main-content .col-sm-12')
	if not main_content:
		return []

	name_headings = main_content.find_all('h3')
	for idx, h3 in enumerate(name_headings):
		name_text = normalize_whitespace(h3.get_text(" "))
		if not _is_name(name_text):
			continue

		block_text_parts = []
		block_html_parts = [str(h3)]
		title_val = None

		for sib in h3.next_siblings:
			if getattr(sib, 'name', None) in ('h3', 'h2'):
				break
			
			block_html_parts.append(str(sib))
			
			if getattr(sib, 'name', None) in ('h4', 'p'):
				block_text_parts.append(normalize_whitespace(sib.get_text(" ")))
			
			if getattr(sib, 'name', None) == 'h4' and not title_val:
				title_val = normalize_whitespace(sib.get_text(" "))

		block_text = " ".join(block_text_parts)
		block_html = "\n".join(block_html_parts)
		
		mailto_emails = extract_mailto_emails_from_html(block_html)
		visible_emails = extract_emails(block_text)
		all_emails = list({*mailto_emails, *visible_emails})
		selected_emails = select_professor_emails(name_text, all_emails, limit=1)

		slug = _slugify(name_text) or f"entry-{idx}"
		profile_url = f"{_normalize_url_no_fragment(base_url)}#faculty-{slug}"

		profiles.append(ProfileResult(
			university_name="University of New Mexico",
			program_name=program,
			professor_name=name_text,
			title=title_val,
			contact_info=ContactInfo(emails=selected_emails),
			profile_url=profile_url,
			is_professor_page=True,
			source="heuristic",
		))
	return profiles


def _detect_asu_isearch(soup: BeautifulSoup, root: BeautifulSoup, base_url: str) -> bool:
	return bool(root.select_one('div#clas-isearch-react[data-config]'))


def _extract_from_asu_isearch_react(soup: BeautifulSoup, base_url: str, university: Optional[str], program: Optional[str]) -> List[ProfileResult]:
	"""ASU-specific parser for iSearch React component data."""
	profiles = []
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
					profiles.append(ProfileResult(
						university_name="Arizona State University",
						program_name=program,
						professor_name=asurite_id.replace('.', ' ').title(),
						profile_url=profile_url,
						source="heuristic"
					))
		except (json.JSONDecodeError, KeyError) as e:
			print(f"Warning: Could not parse ASU iSearch config: {e}")
	return profiles


Detector = Callable[[BeautifulSoup, BeautifulSoup, str], bool]
Extractor = Callable[[BeautifulSoup, BeautifulSoup, str, Optional[str], Optional[str]], List[ProfileResult]]


SPECIALIZED_LIST_HANDLERS: List[Tuple[Detector, Extractor]] = [
	(
		_detect_unm,
		lambda soup, root, base_url, university, program: _extract_from_unm(
			root, base_url, university, program
		),
	),
	(
		_detect_uky,
		lambda soup, root, base_url, university, program: _extract_from_uky(
			root, base_url, university, program
		),
	),
	(
		_detect_lsu,
		lambda soup, root, base_url, university, program: _extract_from_lsu(root, base_url),
	),
	(
		_detect_syracuse,
		lambda soup, root, base_url, university, program: _extract_from_syracuse(
			soup, base_url
		),
	),
	(
		_detect_udel,
		lambda soup, root, base_url, university, program: _extract_from_udel(root, base_url),
	),
	(
		_detect_asu_isearch,
		lambda soup, root, base_url, university, program: _extract_from_asu_isearch_react(
			root, base_url, university, program
		),
	),
]


def extract_profiles_from_listing(html: str, base_url: str, university: Optional[str], program: Optional[str]) -> List[ProfileResult]:
	try:
		soup = BeautifulSoup(html, "lxml")
	except FeatureNotFound:
		soup = BeautifulSoup(html, "html.parser")
	root = _choose_content_root(soup)

	for detector, extractor in SPECIALIZED_LIST_HANDLERS:
		if detector(soup, root, base_url):
			profiles = extractor(soup, root, base_url, university, program)
			if profiles:
				return profiles

	profiles: List[ProfileResult] = []

	# Revised strategy: find profile containers and extract info from them
	profile_containers = root.select('.panel-profile.list-view, .list-item.person, .wsu-card-person, .faculty-listing-item, .table-responsive tbody tr, .people-contact-collection__grid--item, table.scholarships tbody tr, .faculty-member, .person-card, .faculty-card, .directory-item, .person-item, .faculty-item')
	if not profile_containers:
		# Fallback to old heading-based method if no containers found
		return _extract_profiles_from_headings(soup, root, base_url, university, program)

	for idx, container in enumerate(profile_containers):
		name_el = container.find('strong')
		if not name_el:
			name_el = container.select_one('a[href*="/people/"]')
		if not name_el:
			name_el = container.select_one('a[href*="/faculty-staff/"]')
		if not name_el:
			# For LSU table layout
			name_el = container.select_one('span[style*="font-size: 18px"]')

		if not name_el:
			name_el = container.find('a', href=True)
		if not name_el:
			continue
		name_text = normalize_whitespace(name_el.get_text(" "))
		# ASU specific name cleanup
		if "ASU" in (university or "") and "[" in name_text:
			name_text = name_text.split("[")[0].strip()
		if not _is_name(name_text):
			continue

		profile_url: Optional[str] = None
		# Look for a specific "view profile" or "see bio" link first
		link_el = container.find('a', string=re.compile(r"see bio|view profile", re.I))
		if not link_el:
			link_el = container.select_one('a.wvu-button, a[href*="/staff/"], a[href*="/faculty-research/"], a.wvu-profile__name-link, a[href*="/faculty-staff/"]')
		if not link_el and name_el and name_el.find_parent('a'):
			# For LSU, the name itself is in the link
			link_el = name_el.find_parent('a')
		if not link_el:
			link_el = container.find('a', href=True)


		if link_el and link_el.get('href') and not str(link_el['href']).lower().startswith('mailto:'):
			profile_url = urljoin(base_url, link_el['href'])
		else:
			# If no profile link is found, create a placeholder URL with a fragment
			slug = _slugify(name_text) or f"entry-{idx}"
			profile_url = f"{_normalize_url_no_fragment(base_url)}#faculty-{slug}"
		
		title_el = container.select_one('.job-title, .title, .wsu-card__person-title, .faculty-listing-bio p, td, .wvu-profile__job-title')
		title_val = normalize_whitespace(title_el.get_text(" ")) if title_el else None
		if title_val and name_text:
			title_val = title_val.replace(name_text, "").strip()


		container_html = str(container)
		mailto_emails = extract_mailto_emails_from_html(container_html)
		visible_emails = extract_emails(container.get_text(" "))
		all_emails = list({*mailto_emails, *visible_emails})
		selected_emails = select_professor_emails(name_text, all_emails, limit=1)
		
		is_prof_page = (title_val and PROF_TITLE_PAT.search(title_val)) or PROF_TITLE_PAT.search(container.get_text(" "))

		# For linkless profiles, we can try to extract research interests from the container
		research_interest = None
		if not link_el:
			# A bit of a guess: the specialization is often in a nearby column or div.
			# For a table row, we might look in other `td` elements.
			if container.name == 'tr':
				cells = container.find_all('td')
				if len(cells) > 2: # Assuming name is in one, contact in another, research in a third
					research_interest = normalize_whitespace(cells[2].get_text(" "))

		profiles.append(ProfileResult(
			university_name=university,
			program_name=program,
			professor_name=name_text,
			title=title_val,
			contact_info=ContactInfo(emails=selected_emails, phones=[]),
			research_interest=research_interest,
			keywords=keywords_from_interest(research_interest) if research_interest else [],
			profile_url=profile_url,
			is_professor_page=bool(is_prof_page),
			is_accepting_phd=None,
			source="heuristic",
		))
	
	return profiles


def _extract_profiles_from_headings(soup: BeautifulSoup, root: BeautifulSoup, base_url: str, university: Optional[str], program: Optional[str]) -> List[ProfileResult]:
	profiles: List[ProfileResult] = []
	
	# Enhanced fallback specifically for UMaine directory
	if "cmj.umaine.edu" in base_url:
		# Find all faculty entries by looking for h2 headings with names
		faculty_headings = soup.find_all('h2')
		for heading in faculty_headings:
			heading_text = normalize_whitespace(heading.get_text(" "))
			if not _is_name(heading_text):
				continue
				
			# Look for webpage link in the following content
			next_element = heading.find_next_sibling()
			profile_url = base_url
			while next_element and next_element.name not in ['h2', 'h1']:
				if next_element.name == 'p':
					link = next_element.find('a', string=re.compile(r'webpage', re.I))
					if link and link.get('href'):
						profile_url = urljoin(base_url, link['href'])
						break
				next_element = next_element.find_next_sibling()
			
			# Extract title and email from the heading context
			title = "Faculty"
			email = None
			parent_section = heading.find_parent('div') or heading.find_parent('section')
			if parent_section:
				text_content = parent_section.get_text(" ")
				# Look for title patterns
				title_match = re.search(r'(Professor|Associate Professor|Assistant Professor|Chair|Lecturer|Instructor)', text_content, re.I)
				if title_match:
					title = title_match.group(1)
				# Look for email
				email_match = re.search(r'(\S+@\S+\.\S+)', text_content)
				if email_match:
					email = email_match.group(1)
			
			profiles.append(ProfileResult(
				university_name=university or "University of Maine",
				program_name=program or "Communication and Journalism",
				professor_name=heading_text,
				title=title,
				profile_url=profile_url,
				contact_info=ContactInfo(emails=[email] if email else []),
				is_professor_page=True,
				source="heuristic"
			))
		
		if profiles:
			return profiles
	
	# Enhanced fallback specifically for UNL directory
	if "comm.unl.edu" in base_url:
		# Find all person items with structured data
		person_items = soup.select('li[itemtype="https://schema.org/Person"]')
		for item in person_items:
			name_el = item.select_one('a[itemprop="name"]')
			if not name_el:
				continue
				
			name = normalize_whitespace(name_el.get_text(" "))
			if not name or len(name) < 2:
				continue
				
			# Get title
			title_el = item.select_one('[itemprop="jobTitle"]')
			title = normalize_whitespace(title_el.get_text(" ")) if title_el else "Faculty"
			
			# Get email
			email_el = item.select_one('a[itemprop="email"]')
			email = email_el['href'].replace('mailto:', '') if email_el and email_el.get('href') else None
			
			# Get profile URL
			profile_url = urljoin(base_url, name_el['href']) if name_el.get('href') else base_url
			
			profiles.append(ProfileResult(
				university_name=university or "University of Nebraska-Lincoln",
				program_name=program or "Communication Studies",
				professor_name=name,
				title=title,
				profile_url=profile_url,
				contact_info=ContactInfo(emails=[email]) if email else ContactInfo(),
				is_professor_page=True,
				source="heuristic"
			))
		
		if profiles:
			return profiles
	
	idx = 0
	for heading_tag in ["h2", "h3", "h4"]:
		for h in root.find_all(heading_tag):
			name_text = normalize_whitespace(h.get_text(" "))
			if not _is_name(name_text):
				continue
			idx += 1
			buff_parts: List[str] = []
			for sib in h.next_siblings:
				if getattr(sib, 'name', None) in ("h2", "h3", "h4"):
					break
				if getattr(sib, 'name', None) in ("p", "div", "span", "ul", "li", "dd", "dt"):
					buff_parts.append(normalize_whitespace(sib.get_text(" ")))
				elif isinstance(sib, str):
					buff_parts.append(normalize_whitespace(sib))
			block_text = normalize_whitespace(" ".join(p for p in buff_parts if p))

			title_val: Optional[str] = None
			m = re.search(r"([A-Za-z ,\-/]*Professor[ A-Za-z,\-/]*|Chair|Lecturer|Instructor)", block_text, re.I)
			if m:
				title_val = m.group(1).strip()

			block_html = str(h) + "\n" + "\n".join(str(s) for s in h.next_siblings)
			mailto_emails = extract_mailto_emails_from_html(block_html)
			visible_emails = extract_emails(block_text)
			all_emails = list({*mailto_emails, *visible_emails})
			selected_emails = select_professor_emails(name_text, all_emails, limit=1)

			# Prefer a profile link inside the heading or immediate siblings
			profile_url: Optional[str] = None
			link = h.find('a', href=True)
			if link and link.get('href') and not str(link['href']).lower().startswith('mailto:'):
				profile_url = urljoin(base_url, link['href'])
			if not profile_url:
				# search a few next siblings for a link
				count = 0
				for sib in h.next_siblings:
					if count > 5:
						break
					count += 1
					if getattr(sib, 'name', None):
						lnk = sib.find('a', href=True)
						if lnk and lnk.get('href') and not str(lnk['href']).lower().startswith('mailto:'):
							profile_url = urljoin(base_url, lnk['href'])
							break

			if not profile_url:
				anchor_id = h.get('id')
				if anchor_id:
					profile_url = f"{_normalize_url_no_fragment(base_url)}#{anchor_id}"
				else:
					slug = _slugify(name_text) or f"entry-{idx}"
					profile_url = f"{_normalize_url_no_fragment(base_url)}#name-{slug}-{idx}"

			is_prof_page = bool(title_val and PROF_TITLE_PAT.search(title_val)) or bool(PROF_TITLE_PAT.search(block_text))

			profiles.append(ProfileResult(
				university_name=university,
				program_name=program,
				professor_name=name_text,
				title=title_val,
				contact_info=ContactInfo(emails=selected_emails, phones=[]),
				research_interest=None,
				keywords=[],
				profile_url=profile_url,
				is_professor_page=is_prof_page,
				is_accepting_phd=None,
				source="heuristic",
			))

	return profiles
