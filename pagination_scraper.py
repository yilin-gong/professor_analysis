"""Simple pagination scraper with optional browser support and OpenAI summaries."""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI

try:  # Optional import if user wants to fetch pages with a headless browser
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - playwright is optional
    sync_playwright = None


def fetch_html(url, use_browser=False):
    """Return raw HTML for a URL.

    If ``use_browser`` is True and Playwright is available, a headless
    browser is used to render the page. Otherwise ``requests`` is used.
    """
    if use_browser and sync_playwright is not None:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=10000)
            html = page.content()
            browser.close()
            return html

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def extract_professor_links(url, use_browser=False):
    """Return professor links and next page url from a page."""
    html = fetch_html(url, use_browser=use_browser)
    soup = BeautifulSoup(html, 'html.parser')

    professor_links = []
    for a in soup.find_all('a', href=True):
        text = a.get_text(strip=True).lower()
        href = a['href']
        if 'professor' in text or 'professor' in href.lower():
            professor_links.append(urljoin(url, href))

    # Find next page link
    next_link = None
    for a in soup.find_all('a', href=True):
        link_text = a.get_text(strip=True).lower()
        if link_text in {'next', 'next page', '>', '»'} or 'page=' in a['href'].lower():
            next_link = urljoin(url, a['href'])
            break

    return professor_links, next_link


def fetch_page_summary(url, client, use_browser=False):
    """Fetch page text and return an OpenAI-generated summary."""
    html = fetch_html(url, use_browser=use_browser)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)[:3000]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that extracts a professor's "
                    "research interests and a brief description from a webpage."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Extract the research interests and a short description "
                    f"from the following webpage text:\n\n{text}"
                ),
            },
        ],
    )

    return response.choices[0].message.content.strip()


def crawl(start_url, api_key, max_pages=3, use_browser=False):
    """Crawl pages starting from ``start_url`` using pagination.

    For each professor link found, the page is summarized with the OpenAI API.
    """

    client = OpenAI(api_key=api_key)
    current = start_url
    page_num = 1
    results = []

    while current and page_num <= max_pages:
        print(f"\n== Page {page_num}: {current} ==")
        try:
            links, next_page = extract_professor_links(current, use_browser)
            for link in links:
                print(f"Found professor link: {link}")
                summary = fetch_page_summary(link, client, use_browser)
                print(f"  Summary: {summary[:100]}\n")
                results.append({"url": link, "summary": summary})
            current = next_page
        except Exception as exc:
            print(f"Error scraping {current}: {exc}")
            break
        page_num += 1

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Professor scraper with optional browser rendering")
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--pages', type=int, default=3, help='Maximum pages to traverse')
    parser.add_argument('--browser', action='store_true', help='Use headless browser for fetching')
    args = parser.parse_args()

    crawl(args.url, args.api_key, max_pages=args.pages, use_browser=args.browser)
