import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def extract_professor_links(url):
    """Return professor links and next page url from a page."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

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


def fetch_page_info(url):
    """Fetch basic text info from a professor page."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)[:500]


def crawl(start_url, max_pages=3):
    """Crawl pages starting from start_url following pagination."""
    current = start_url
    page = 1
    while current and page <= max_pages:
        print(f"\n== Page {page}: {current} ==")
        try:
            links, next_page = extract_professor_links(current)
            for link in links:
                print(f"Found professor link: {link}")
                info = fetch_page_info(link)
                print(f"  Info snippet: {info[:100]}\n")
            current = next_page
        except Exception as exc:
            print(f"Error scraping {current}: {exc}")
            break
        page += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Simple professor scraper with pagination")
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument('--pages', type=int, default=3, help='Maximum pages to traverse')
    args = parser.parse_args()

    crawl(args.url, max_pages=args.pages)
