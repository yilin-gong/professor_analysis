import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import concurrent.futures
from openai import OpenAI
import urllib.parse
import logging
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 创建OpenAI客户端函数
def get_client(api_key):
    """使用提供的API密钥创建OpenAI客户端"""
    if not api_key:
        raise ValueError("必须提供API密钥")
    return OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")


# OpenAI客户端将在调用时初始化
client = None


# Create a session with retry capability
def create_session():
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def detect_next_page(soup, base_url):
    """Return the URL of the next page if a pagination link is found."""
    next_link = soup.find("a", rel=lambda x: x and "next" in x.lower())
    if next_link and next_link.get("href"):
        href = next_link["href"]
        if not href.startswith(("http://", "https://")):
            return urllib.parse.urljoin(base_url, href)
        return href

    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(strip=True).lower()
        if text in ("next", "next page", ">", "»", "下一页"):
            href = anchor["href"]
            if not href.startswith(("http://", "https://")):
                return urllib.parse.urljoin(base_url, href)
            return href
    return None


def get_all_links(url, session=None, follow_pagination=False, max_pages=3):
    """Extract links from a webpage, optionally following pagination."""
    if session is None:
        session = create_session()

    to_visit = [url]
    visited = set()
    collected = []
    page_count = 0

    while to_visit and page_count < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            logger.info(f"Extracting links from {current_url}")
            response = session.get(
                current_url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            logger.error(f"Error scraping {current_url}: {e}")
            continue

        # Parse the URL for resolving relative links
        parsed_url = urllib.parse.urlparse(current_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Check for base tag
        base_tag = soup.find("base", href=True)
        if base_tag and base_tag["href"]:
            base_href = base_tag["href"]
            if base_href.startswith(("http://", "https://")):
                base_url = base_href
            else:
                base_url = urllib.parse.urljoin(base_url, base_href)

        links = []
        professor_keywords = [
            "faculty",
            "professor",
            "staff",
            "people",
            "members",
            "researchers",
            "team",
        ]

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            anchor_text = anchor.get_text(strip=True).lower()

            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            if not href.startswith(("http://", "https://")):
                href = urllib.parse.urljoin(base_url, href)

            if href.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip")):
                continue

            priority = 0
            if any(keyword in anchor_text for keyword in professor_keywords):
                priority = 2
                logger.info(
                    f"Found potential professor-related link: {href} - Text: {anchor_text}"
                )
            elif any(keyword in href.lower() for keyword in professor_keywords):
                priority = 1

            links.append((href, priority))

        collected.extend(links)

        if follow_pagination:
            next_url = detect_next_page(soup, base_url)
            if next_url and next_url not in visited:
                to_visit.append(next_url)

        page_count += 1

    # Remove duplicates while prioritizing professor-related links
    unique_links = []
    seen = set()
    for link, priority in sorted(collected, key=lambda x: x[1], reverse=True):
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    logger.info(
        f"Found {len(unique_links)} unique links from {url} across {page_count} page(s)"
    )
    return unique_links


def is_professor_webpage(url, session=None, client=None):
    """Use LLM to determine if a webpage is a professor's personal page."""
    if session is None:
        session = create_session()

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        logger.info(f"Checking if {url} is a professor webpage")
        response = session.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title and meta description
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_desc = meta_tag["content"]

        # Extract text content
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)
        # Limit text length for API efficiency
        text = text[:3000]

        # Look for professor-specific keywords
        professor_keywords = [
            "professor",
            "faculty",
            "dr.",
            "ph.d",
            "phd",
            "research interests",
            "publications",
            "curriculum vitae",
            "cv",
            "teaching",
            "associate professor",
            "assistant professor",
            "scholar",
        ]
        keyword_matches = [
            kw
            for kw in professor_keywords
            if kw.lower() in text.lower() or kw.lower() in title.lower()
        ]

        # Combine important elements for LLM analysis
        important_elements = f"URL: {url}\nTitle: {title}\nMeta Description: {meta_desc}\n\nContent Preview: {text[:500]}\n\nDetected Keywords: {', '.join(keyword_matches)}"

        logger.info(f"Title: {title}")
        logger.info(f"Keyword matches: {keyword_matches}")

        # Ask LLM to classify the page
        llm_response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that determines if a webpage belongs to a professor at a university or research institution. Look for indicators like academic titles (Professor, Dr., Ph.D.), research interests, publications, courses taught, and academic affiliations.",
                },
                {
                    "role": "user",
                    "content": f"Based on the following webpage details, determine if this is a personal or professional page of a professor, faculty member, or academic researcher at a university or research institution. Answer only YES or NO.\n\n{important_elements}",
                },
            ],
        )

        answer = llm_response.choices[0].message.content.strip().upper()
        logger.info(f"LLM classification for {url}: {answer}")
        return "YES" in answer
    except Exception as e:
        logger.error(f"Error analyzing {url}: {e}")
        return False


def get_research_interests(url, session=None, client=None):
    """Use LLM to summarize a professor's research interests from their webpage."""
    if session is None:
        session = create_session()

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Get title
        title = soup.title.string if soup.title else ""

        # Extract text content
        for script in soup(["script", "style"]):
            script.extract()

        # Try to find research-specific sections
        research_sections = []
        research_keywords = [
            "research",
            "interests",
            "projects",
            "expertise",
            "publications",
        ]

        # Look for sections with research-related headers
        for header in soup.find_all(["h1", "h2", "h3", "h4"]):
            header_text = header.get_text(strip=True).lower()
            if any(keyword in header_text for keyword in research_keywords):
                section_content = []
                # Collect paragraphs until the next header
                for sibling in header.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4"]:
                        break
                    if sibling.name in ["p", "ul", "ol", "div"]:
                        section_content.append(sibling.get_text(strip=True))

                if section_content:
                    research_sections.append(
                        {"header": header_text, "content": " ".join(section_content)}
                    )

        # Combine all found sections
        research_content = ""
        if research_sections:
            research_content = "\n\n".join(
                f"{section['header']}: {section['content'][:500]}"
                for section in research_sections
            )
        else:
            # If no specific research sections found, use general page content
            research_content = soup.get_text(separator=" ", strip=True)[:3000]

        # Ask LLM to extract research interests
        llm_response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that extracts and summarizes a professor's research interests from their webpage. Your task is to provide detailed summaries that capture specific research areas, methodologies, and even subtle differences in research directions.",
                },
                {
                    "role": "user",
                    "content": f"Based on the following professor's webpage content, provide a detailed summary of their research interests. Include specific topics, methodologies, and capture any nuanced differences in research directions. Make the summary comprehensive while remaining focused on their core research areas:\n\nPage Title: {title}\n\n{research_content}",
                },
            ],
        )

        return llm_response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error extracting research interests from {url}: {e}")
        return "Unable to determine research interests"


def process_link(link, session, client=None):
    """Process a single link - to be used with ThreadPoolExecutor."""
    logger.info(f"Analyzing: {link}")

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        is_professor = is_professor_webpage(link, session, client)
        if is_professor:
            research_interests = get_research_interests(link, session, client)
            return {
                "URL": link,
                "Is Professor Page": "Yes",
                "Research Interests": research_interests,
            }
        else:
            return {"URL": link, "Is Professor Page": "No", "Research Interests": ""}
    except Exception as e:
        logger.error(f"Error processing {link}: {e}")
        return {
            "URL": link,
            "Is Professor Page": "Error",
            "Research Interests": f"Error: {str(e)}",
        }


def analyze_webpage_links(start_url, api_key, max_links=50, max_workers=5, max_pages=3):
    """Analyze a website to find professor pages and research interests.

    Parameters
    ----------
    start_url : str
        The starting page URL for scraping.
    api_key : str
        API key used to access the LLM service.
    max_links : int, optional
        Maximum number of links to analyze.
    max_workers : int, optional
        Number of threads used for processing.
    max_pages : int, optional
        Maximum pages to follow when detecting pagination.
    """
    logger.info(f"Starting analysis from: {start_url}")

    # 使用提供的API密钥初始化客户端
    if not api_key:
        raise ValueError("必须提供API密钥才能执行分析")

    client = get_client(api_key)

    # Create a session for consistent connections
    session = create_session()

    # Get all links from the starting URL, following pagination if present
    all_links = get_all_links(
        start_url, session, follow_pagination=True, max_pages=max_pages
    )[:max_links]
    logger.info(f"Found {len(all_links)} links to analyze.")

    if not all_links:
        logger.warning(
            "No links found to analyze. Check if the website is accessible or has changed structure."
        )
        return pd.DataFrame(columns=["URL", "Is Professor Page", "Research Interests"])

    # Print some of the links we're going to check
    for i, link in enumerate(all_links[:10]):
        logger.info(f"Link {i+1} to check: {link}")

    results = []

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {
            executor.submit(process_link, link, session, client): link
            for link in all_links
        }
        for future in concurrent.futures.as_completed(future_to_link):
            link = future_to_link[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # Log professor pages as we find them
                    if result["Is Professor Page"] == "Yes":
                        logger.info(f"FOUND PROFESSOR PAGE: {link}")
                        logger.info(
                            f"Research Interests: {result['Research Interests']}"
                        )
            except Exception as e:
                logger.error(f"Exception processing {link}: {e}")
                results.append(
                    {
                        "URL": link,
                        "Is Professor Page": "Error",
                        "Research Interests": f"Error: {str(e)}",
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Filter only professor pages if desired
    professors_df = df[df["Is Professor Page"] == "Yes"]

    logger.info(
        f"Analysis complete. Found {len(professors_df)} professor pages out of {len(results)} total links analyzed."
    )

    # Print summary statistics
    yes_count = sum(1 for r in results if r["Is Professor Page"] == "Yes")
    no_count = sum(1 for r in results if r["Is Professor Page"] == "No")
    error_count = sum(1 for r in results if r["Is Professor Page"] == "Error")

    logger.info(
        f"Results Summary: {yes_count} professor pages, {no_count} non-professor pages, {error_count} errors"
    )

    return df


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a website to find professor pages and research interests."
    )
    parser.add_argument("url", help="Starting URL to analyze")
    parser.add_argument("--api-key", required=True, help="API key for OpenAI services")
    parser.add_argument(
        "--max-links", type=int, default=30, help="Maximum number of links to analyze"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum number of pages to follow when detecting pagination",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of worker threads"
    )
    parser.add_argument(
        "--output", default="professor_analysis_results.csv", help="Output CSV filename"
    )
    args = parser.parse_args()

    # Run the analysis
    results_df = analyze_webpage_links(
        args.url,
        args.api_key,
        max_links=args.max_links,
        max_workers=args.workers,
        max_pages=args.max_pages,
    )

    # Save full results to CSV
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Display professor pages
    professor_pages = results_df[results_df["Is Professor Page"] == "Yes"]
    if not professor_pages.empty:
        # Save professor-only results to a separate file
        professor_output = args.output.replace(".csv", "_professors_only.csv")
        professor_pages.to_csv(professor_output, index=False)

        # Print professor pages to console
        print("\nProfessor Pages Found:")
        print(professor_pages.to_string())
    else:
        print("\nNo professor pages found.")
"""
python main.py https://journalism.uiowa.edu/people --max-links 500 --workers 5
"""
