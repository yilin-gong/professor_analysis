import time
from typing import Optional, Tuple

import requests
from urllib.parse import urldefrag
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


DEFAULT_HEADERS = {
	"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	"Accept-Language": "en-US,en;q=0.9",
	"Connection": "keep-alive",
}


SESSION = requests.Session()


class HttpError(Exception):
	pass


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.6, min=0.6, max=5), retry=retry_if_exception_type((requests.RequestException,)) )
def _do_get(url: str, timeout: float, referer: Optional[str]) -> requests.Response:
	clean_url, _ = urldefrag(url)
	headers = DEFAULT_HEADERS.copy()
	if referer:
		ref_clean, _ = urldefrag(referer)
		headers["Referer"] = ref_clean
	else:
		headers.setdefault("Referer", clean_url)
	resp = SESSION.get(clean_url, headers=headers, timeout=timeout, allow_redirects=True)
	resp.raise_for_status()
	return resp


def fetch_html(url: str, timeout: float = 15.0, delay: float = 1.0, referer: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
	try:
		resp = _do_get(url, timeout, referer)
		time.sleep(delay)
		content_type = resp.headers.get("Content-Type", "")
		if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
			return None, resp.url
		return resp.text, resp.url
	except requests.RequestException as e:
		print(f"  [Requests Error] Failed to fetch {url}: {e}")
		return None, None


def fetch_html_browser(url: str, timeout: float = 60.0) -> Tuple[Optional[str], Optional[str]]:
	try:
		from playwright.sync_api import sync_playwright
	except Exception:
		return None, None
	try:
		with sync_playwright() as p:
			browser = p.chromium.launch(headless=True)
			context = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], java_script_enabled=True)
			page = context.new_page()
			page.set_default_timeout(timeout * 1000)
			
			# For React/SPA sites, use a more lenient approach
			if "newhouse.syracuse.edu" in url:
				try:
					page.goto(url, wait_until="networkidle", timeout=30000)
				except:
					# Fallback if networkidle times out
					page.goto(url, wait_until="domcontentloaded")
				# Additional wait for React content to render
				page.wait_for_timeout(5000)
			else:
				page.goto(url, wait_until="domcontentloaded")
			
			html = page.content()
			final_url = page.url
			context.close()
			browser.close()
			return html, final_url
	except Exception as e:
		print(f"  [Playwright Error] Failed to fetch {url}: {e}")
		return None, None


def fetch_html_auto(url: str, *, use_browser: bool = False, referer: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
	if use_browser:
		html, final = fetch_html_browser(url)
		# Do not fall back if browser is explicitly requested and fails; it's likely intentional
		return html, final
	return fetch_html(url, referer=referer)
