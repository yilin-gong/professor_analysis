import re
from typing import Iterable, List, Optional, Tuple


WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})")
MAILTO_RE = re.compile(r"href=[\"']mailto:([^\"'\?\s>]+)", re.I)

GENERIC_LOCAL_PART_BAD = {
	"info","contact","contacts","admissions","admission","enroll","enrolment","help","support","service","services","office","privacy","press","media","webmaster","no-reply","noreply","newsletter","events","event","alumni","students","student","recruit","career","advising","advisor","department","dept","school","college","faculty","comm","communication","marketing","hr","finance","billing","accounts","account","news"
}


def normalize_whitespace(text: str) -> str:
	return WHITESPACE_RE.sub(" ", (text or "")).strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
	seen = set()
	result: List[str] = []
	for item in items:
		if item and item not in seen:
			seen.add(item)
			result.append(item)
	return result


def extract_emails(text: str) -> List[str]:
	return dedupe_preserve_order(m.group(0).lower() for m in EMAIL_RE.finditer(text or ""))


def extract_mailto_emails_from_html(html: str) -> List[str]:
	return dedupe_preserve_order(m.group(1).lower() for m in MAILTO_RE.finditer(html or ""))


def extract_phones(text: str) -> List[str]:
	return dedupe_preserve_order(m.group(0) for m in PHONE_RE.finditer(text or ""))


def _split_name(full_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
	if not full_name:
		return None, None
	parts = [p for p in re.split(r"\s+", full_name.strip()) if p]
	if not parts:
		return None, None
	first = parts[0]
	last = parts[-1] if len(parts) > 1 else None
	return first.lower(), (last.lower() if last else None)


def score_email_for_name(full_name: Optional[str], email: str) -> int:
	local = email.split('@', 1)[0].lower()
	first, last = _split_name(full_name)
	score = 0
	if last and last in local:
		score += 3
	if first and local.startswith(first[0]):
		score += 1
	if first and first in local:
		score += 1
	# penalize generic addresses
	if local in GENERIC_LOCAL_PART_BAD:
		score -= 4
	for bad in GENERIC_LOCAL_PART_BAD:
		if f"-{bad}" in local or f"_{bad}" in local:
			score -= 2
	return score


def select_professor_emails(full_name: Optional[str], emails: List[str], limit: int = 1) -> List[str]:
	if not emails:
		return []
	scored = sorted(emails, key=lambda e: score_email_for_name(full_name, e), reverse=True)
	best = [e for e in scored if score_email_for_name(full_name, e) >= 0]
	if not best:
		best = [scored[0]]
	return best[:limit]


def keywords_from_interest(interest: str, limit: int = 15) -> List[str]:
	if not interest:
		return []
	tokens = re.split(r"[^A-Za-z0-9\-\+]+", interest.lower())
	tokens = [t for t in tokens if len(t) >= 3 and not t.isdigit()]
	return dedupe_preserve_order(tokens)[:limit]
