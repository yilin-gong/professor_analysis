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
import re
from typing import Dict, List, Tuple, Optional, Union
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 创建OpenAI客户端函数
def get_client(api_key: str) -> OpenAI:
    """使用提供的API密钥创建OpenAI客户端
    
    Args:
        api_key: API密钥字符串
        
    Returns:
        OpenAI客户端实例
        
    Raises:
        ValueError: 如果API密钥为空
    """
    if not api_key:
        raise ValueError("必须提供API密钥")
    return OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")


# OpenAI客户端将在调用时初始化
client = None


# Create a session with retry capability
def create_session() -> requests.Session:
    """创建具有重试功能的HTTP会话
    
    Returns:
        配置了重试策略的requests.Session对象
    """
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def detect_next_page(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """检测分页中的下一页链接
    
    Args:
        soup: 当前页面的BeautifulSoup对象
        base_url: 基础URL，用于解析相对链接
        
    Returns:
        下一页的URL，如果没有找到则返回None
    """
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


def get_all_links(
    url: str, 
    session: Optional[requests.Session] = None, 
    follow_pagination: bool = False, 
    max_pages: int = 3
) -> List[str]:
    """从网页中提取链接，可选择跟随分页
    
    Args:
        url: 起始网页URL
        session: HTTP会话对象，如果为None则创建新的
        follow_pagination: 是否跟随分页链接
        max_pages: 最多跟随的页面数量
        
    Returns:
        高质量链接的列表，按评分排序
        
    Raises:
        Exception: 网络请求或解析错误时
    """
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

        # 分析页面结构
        page_structure = analyze_page_structure(soup)

        links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            anchor_text = anchor.get_text(strip=True).lower()

            # 基础过滤
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            if not href.startswith(("http://", "https://")):
                href = urllib.parse.urljoin(base_url, href)

            # 文件类型过滤
            if href.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx", ".ppt", ".pptx")):
                continue

            # 使用新的评分系统
            score = calculate_link_score(anchor, href, page_structure, base_url)
            
            # 检测页面类型并动态调整阈值
            is_academic_page = detect_academic_page_type(base_url, page_structure)
            threshold = 1.5 if is_academic_page else 2  # 学术页面使用更低阈值
            
            # 只保留评分大于阈值的链接
            if score > threshold:
                links.append((href, score))
                if score >= 5:  # 高分链接记录日志
                    logger.info(f"High-score professor link found: {href} (Score: {score}) - Text: {anchor_text}")
                elif is_academic_page and score > 3:
                    logger.info(f"Academic page link found: {href} (Score: {score}) - Text: {anchor_text}")

        collected.extend(links)

        if follow_pagination:
            next_url = detect_next_page(soup, base_url)
            if next_url and next_url not in visited:
                to_visit.append(next_url)

        page_count += 1

    # Remove duplicates while prioritizing higher-scored links
    unique_links = []
    seen = set()
    # 按分数从高到低排序，优先处理高分链接
    for link, score in sorted(collected, key=lambda x: x[1], reverse=True):
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    logger.info(
        f"Found {len(unique_links)} high-quality links from {url} across {page_count} page(s)"
    )
    return unique_links


def is_professor_webpage(url, session=None, client=None):
    """Use LLM to determine if a webpage is a professor's personal page and extract basic info."""
    if session is None:
        session = create_session()

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        logger.info(f"Checking if {url} is a professor webpage")
        response = robust_web_request(session, url)
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

        # 寻找H1标签作为可能的姓名来源
        h1_text = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            h1_text = h1_tag.get_text(strip=True)

        # Combine important elements for LLM analysis
        important_elements = f"URL: {url}\nTitle: {title}\nH1 Tag: {h1_text}\nMeta Description: {meta_desc}\n\nContent Preview: {text[:500]}\n\nDetected Keywords: {', '.join(keyword_matches)}"

        logger.info(f"Title: {title}")
        logger.info(f"Keyword matches: {keyword_matches}")

        # Ask LLM to classify the page and extract structured information
        response_text = robust_llm_call(client, [
            {
                "role": "system",
                "content": """你是一个专业的学术网页分析专家。你的任务是：
1. 判断这是否是大学或研究机构的教授、教员或学术研究人员的个人页面
2. 如果是，提取以下结构化信息：姓名、职位/头衔、院系/部门

请以JSON格式返回结果：
{
    "is_professor": true/false,
    "confidence": 0.0-1.0,
    "name": "教授姓名",
    "title": "职位头衔",
    "department": "院系部门"
}

注意：
- 姓名应该是完整的人名，不包含职位头衔
- 职位包括：Professor, Associate Professor, Assistant Professor, Dr., etc.
- 院系是所属的学院或部门
- 如果不是教授页面，将除is_professor和confidence外的字段设为空字符串""",
            },
            {
                "role": "user",
                "content": f"请分析以下网页信息：\n\n{important_elements}",
            },
        ])

        logger.info(f"LLM response for {url}: {response_text}")
        
        # 尝试解析JSON响应
        try:
            import json
            result = json.loads(response_text)
            is_prof = result.get("is_professor", False)
            
            if is_prof:
                prof_info = {
                    "name": result.get("name", "Unknown"),
                    "title": result.get("title", ""),
                    "department": result.get("department", ""),
                    "confidence": result.get("confidence", 0.5)
                }
                return True, prof_info
            else:
                return False, {
                    "name": "",
                    "title": "",
                    "department": "",
                    "confidence": result.get("confidence", 0.0)
                }
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用原有的简单判断逻辑
            is_prof = "true" in response_text.lower() or "yes" in response_text.lower()
            if is_prof:
                # 尝试从标题中提取姓名
                name = h1_text if h1_text else title
                return True, {
                    "name": name,
                    "title": "",
                    "department": "",
                    "confidence": 0.7
                }
            else:
                return False, {
                    "name": "",
                    "title": "",
                    "department": "",
                    "confidence": 0.3
                }
            
    except Exception as e:
        logger.error(f"Error analyzing {url}: {e}")
        return False, {
            "name": "",
            "title": "",
            "department": "",
            "confidence": 0.0
        }


def get_research_interests(url, session=None, client=None):
    """Use LLM to summarize a professor's research interests from their webpage and extract keywords."""
    if session is None:
        session = create_session()

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        response = robust_web_request(session, url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get title and meta information
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_desc = meta_tag["content"]

        # Extract all text content and remove scripts/styles
        for script in soup(["script", "style"]):
            script.extract()

        # 1. 提取主要标题信息
        main_heading = ""
        h1_tag = soup.find("h1")
        if h1_tag:
            main_heading = h1_tag.get_text(strip=True)

        # 2. 提取所有段落内容
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 20:  # 过滤过短的段落
                paragraphs.append(text)

        # 3. 查找研究相关的特定区域
        research_sections = []
        research_keywords = [
            "research", "interests", "projects", "expertise", "publications",
            "areas", "focus", "specialty", "work", "studies", "investigation"
        ]

        # Look for sections with research-related headers
        for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            header_text = header.get_text(strip=True).lower()
            if any(keyword in header_text for keyword in research_keywords):
                section_content = []
                
                # 收集该标题下的内容，直到下一个同级或更高级标题
                current_level = int(header.name[1])  # h1=1, h2=2, etc.
                
                for sibling in header.find_next_siblings():
                    if (sibling.name and sibling.name.startswith('h') and 
                        int(sibling.name[1]) <= current_level):
                        break
                    if sibling.name in ["p", "ul", "ol", "div", "section"]:
                        content = sibling.get_text(strip=True)
                        if len(content) > 10:
                            section_content.append(content)

                if section_content:
                    research_sections.append({
                        "header": header_text,
                        "content": " ".join(section_content)
                    })

        # 4. 提取列表信息（可能包含研究领域）
        list_items = []
        for ul in soup.find_all(["ul", "ol"]):
            items = []
            for li in ul.find_all("li"):
                item_text = li.get_text(strip=True)
                if len(item_text) > 5:
                    items.append(item_text)
            if items and len(" ".join(items)) < 1000:  # 避免过长的列表
                list_items.extend(items)

        # 5. 查找相关链接并获取额外信息
        related_links = find_related_professor_links(url, session)
        additional_info = integrate_professor_info(url, related_links, session, client)

        # 6. 构建全面的分析内容
        comprehensive_content = {
            "page_title": title,
            "meta_description": meta_desc,
            "main_heading": main_heading,
            "research_sections": research_sections,
            "key_paragraphs": paragraphs[:10],  # 限制段落数量
            "relevant_lists": list_items[:20],   # 限制列表项数量
            "additional_cv_info": additional_info.get('cv_info', ''),
            "additional_publications": additional_info.get('publications_info', ''),
            "additional_research": additional_info.get('additional_research_info', '')
        }

        # 7. 构建给LLM的综合提示词
        analysis_prompt = _build_comprehensive_analysis_prompt(comprehensive_content)

        # Ask LLM to extract research interests and keywords
        response_text = robust_llm_call(client, [
            {
                "role": "system",
                "content": """你是一个专业的学术信息提取专家。你的任务是从教授的网页内容中综合分析并提取研究兴趣和关键词。

请返回JSON格式的结果：
{
    "research_interests": "清洁简洁的研究兴趣描述",
    "keywords": ["关键词1", "关键词2", "关键词3"]
}

分析要求：
1. 研究兴趣描述应该综合所有提供的信息，包括页面标题、研究区域、出版物信息等
2. 不要包含"该教授的研究兴趣是"等描述性语言，直接描述研究内容
3. 不要使用格式化标记（粗体、斜体等）
4. 关键词应该是5-10个最重要的研究领域术语
5. 关键词应该按重要性排序
6. 优先关注研究专业术语和方法论
7. 如果有多个研究领域，应该都涵盖到""",
            },
            {
                "role": "user",
                "content": analysis_prompt,
            },
        ])

        # 尝试解析JSON响应
        try:
            import json
            result = json.loads(response_text)
            
            # 获取研究兴趣并进行额外清理
            raw_interests = result.get("research_interests", "")
            clean_interests = clean_research_text(raw_interests)
            
            # 获取关键词
            keywords = result.get("keywords", [])
            
            # 确保关键词是列表且不超过10个
            if isinstance(keywords, list):
                keywords = keywords[:10]
            else:
                keywords = []
            
            return {
                "interests": clean_interests,
                "keywords": keywords,
                "additional_urls": additional_info.get('additional_urls', []),
            }
            
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用原始文本并尝试简单清理
            clean_interests = clean_research_text(response_text)
            return {
                "interests": clean_interests,
                "keywords": [],
                "additional_urls": additional_info.get('additional_urls', []),
            }
            
    except Exception as e:
        logger.error(f"Error extracting research interests from {url}: {e}")
        return {
            "interests": "Unable to determine research interests",
            "keywords": [],
            "additional_urls": [],
        }


def _build_comprehensive_analysis_prompt(content_dict):
    """构建综合分析提示词"""
    prompt_parts = []
    
    prompt_parts.append("请综合分析以下教授网页的所有信息，提取研究兴趣和关键词：\n")
    
    if content_dict["page_title"]:
        prompt_parts.append(f"页面标题: {content_dict['page_title']}")
    
    if content_dict["main_heading"]:
        prompt_parts.append(f"主要标题: {content_dict['main_heading']}")
    
    if content_dict["meta_description"]:
        prompt_parts.append(f"页面描述: {content_dict['meta_description']}")
    
    if content_dict["research_sections"]:
        prompt_parts.append("\n研究相关区域:")
        for section in content_dict["research_sections"]:
            prompt_parts.append(f"- {section['header']}: {section['content'][:800]}")
    
    if content_dict["key_paragraphs"]:
        prompt_parts.append(f"\n主要段落内容: {' '.join(content_dict['key_paragraphs'][:5])[:1500]}")
    
    if content_dict["relevant_lists"]:
        prompt_parts.append(f"\n相关列表信息: {' | '.join(content_dict['relevant_lists'][:15])}")
    
    if content_dict["additional_cv_info"]:
        prompt_parts.append(f"\nCV信息: {content_dict['additional_cv_info'][:800]}")
    
    if content_dict["additional_publications"]:
        prompt_parts.append(f"\n出版物信息: {content_dict['additional_publications'][:800]}")
    
    if content_dict["additional_research"]:
        prompt_parts.append(f"\n额外研究信息: {content_dict['additional_research'][:800]}")
    
    return "\n".join(prompt_parts)


def process_link(link, session, client=None):
    """Process a single link - to be used with ThreadPoolExecutor."""
    logger.info(f"Analyzing: {link}")

    if client is None:
        raise ValueError("必须提供API客户端")

    try:
        is_professor, prof_info = is_professor_webpage(link, session, client)
        if is_professor:
            # 只有确认是教授个人网页才进行详细分析
            research_data = get_research_interests(link, session, client)
            return {
                "URL": link,
                "Professor Name": prof_info.get("name", "Unknown"),
                "Title": prof_info.get("title", ""),
                "Department": prof_info.get("department", ""),
                "Is Professor Page": "Yes",
                "Research Interests": research_data.get("interests", ""),
                "Keywords": research_data.get("keywords", []),
                "Additional URLs": research_data.get("additional_urls", []),
                "Confidence Score": prof_info.get("confidence", 0.0)
            }
        else:
            # 非教授页面不返回结果，不在最终结果中列举
            return None
    except Exception as e:
        logger.error(f"Error processing {link}: {e}")
        return None


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
        return pd.DataFrame(columns=["URL", "Professor Name", "Title", "Department", "Is Professor Page", "Research Interests", "Keywords", "Additional URLs", "Confidence Score"])

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
                if result:  # 只添加非None的结果（即教授页面）
                    results.append(result)
                    # Log professor pages as we find them
                    logger.info(f"FOUND PROFESSOR PAGE: {link}")
                    logger.info(
                        f"Research Interests: {result['Research Interests']}"
                    )
            except Exception as e:
                logger.error(f"Exception processing {link}: {e}")
                # 错误情况也不再添加到结果中

    # Create DataFrame
    df = pd.DataFrame(results)

    logger.info(
        f"Analysis complete. Found {len(results)} professor pages out of {len(all_links)} total links analyzed."
    )

    # Print summary statistics
    professor_count = len(results)  # 现在results中只包含教授页面
    total_links = len(all_links)
    
    logger.info(
        f"Results Summary: {professor_count} professor pages found from {total_links} links analyzed"
    )

    return df


def analyze_page_structure(soup):
    """分析页面结构，识别导航、内容、页脚等区域"""
    structure = {
        'navigation_elements': [],
        'content_elements': [],
        'footer_elements': [],
        'sidebar_elements': []
    }
    
    # 识别导航区域
    nav_selectors = ['nav', 'header', '.navigation', '.nav', '.menu', '.navbar', '#nav', '#navigation', '#menu']
    for selector in nav_selectors:
        elements = soup.select(selector)
        structure['navigation_elements'].extend(elements)
    
    # 识别页脚区域
    footer_selectors = ['footer', '.footer', '#footer', '.site-footer']
    for selector in footer_selectors:
        elements = soup.select(selector)
        structure['footer_elements'].extend(elements)
    
    # 识别侧边栏区域
    sidebar_selectors = ['aside', '.sidebar', '.side-nav', '#sidebar', '.widget-area']
    for selector in sidebar_selectors:
        elements = soup.select(selector)
        structure['sidebar_elements'].extend(elements)
    
    # 识别主要内容区域
    content_selectors = ['main', '.content', '.main-content', '#content', '#main', 'article', '.post-content']
    for selector in content_selectors:
        elements = soup.select(selector)
        structure['content_elements'].extend(elements)
    
    return structure


def is_link_in_non_content_area(anchor, page_structure):
    """判断链接是否在非内容区域（导航、页脚、侧边栏）"""
    for area_name in ['navigation_elements', 'footer_elements', 'sidebar_elements']:
        for element in page_structure[area_name]:
            if element and anchor in element.find_all('a'):
                return True
    return False


def calculate_link_score(anchor, href, page_structure, base_url):
    """计算链接的综合评分 (0-10分制)"""
    score = 0
    anchor_text = anchor.get_text(strip=True).lower()
    original_anchor_text = anchor.get_text(strip=True)  # 保留原始大小写
    
    # 基础分数
    score += 1
    
    # 检测页面类型 - 学术机构特殊处理
    is_academic_page = detect_academic_page_type(base_url, page_structure)
    
    # 教授相关关键词匹配 (最高4分) - 扩展关键词列表
    professor_keywords = [
        "faculty", "professor", "staff", "people", "members", 
        "researchers", "team", "dr.", "ph.d", "phd", "associate professor",
        "assistant professor", "clinical professor", "emeritus", "lecturer",
        "instructor", "scholar", "academic", "researcher", "scientist",
        "chair", "director", "dean", "postdoc", "fellow"
    ]
    strong_keywords = ["professor", "faculty", "dr.", "ph.d", "phd", "scholar"]
    
    keyword_matches = sum(1 for kw in professor_keywords if kw in anchor_text)
    strong_matches = sum(1 for kw in strong_keywords if kw in anchor_text)
    
    score += min(keyword_matches * 0.5 + strong_matches, 4)
    
    # URL路径分析 (最高4分) - 扩展URL模式识别
    url_keywords = ["faculty", "professor", "people", "staff", "members", "user", "profile", "directory"]
    academic_paths = ["/faculty/", "/people/", "/staff/", "/directory/", "/profiles/", "/bio/", "/academic/"]
    
    # 基础URL关键词匹配
    if any(kw in href.lower() for kw in url_keywords):
        score += 2
    
    # 学术路径模式匹配
    if any(path in href.lower() for path in academic_paths):
        score += 1
        
    # 特殊处理：个人页面模式识别 (更通用的模式)
    # 模式1: /user/数字 或 /profile/数字
    if re.search(r'/(user|profile|member)/\d+/?$', href):
        score += 3
    # 模式2: /people/name 或 /faculty/name
    elif re.search(r'/(people|faculty|staff|directory)/[a-zA-Z\-_]+/?$', href.lower()):
        score += 3
    # 模式3: 包含姓名的URL模式 (/firstname-lastname, /f.lastname等)
    elif re.search(r'/[a-zA-Z]+[\-_\.][a-zA-Z]+(?:[\-_\.][a-zA-Z]+)*/?$', href):
        # 检查是否像是人名模式
        url_parts = href.split('/')[-1].replace('-', ' ').replace('_', ' ').replace('.', ' ')
        if len(url_parts.split()) >= 2:  # 至少两个部分，可能是姓和名
            score += 2
    
    # 增强的人名识别 (支持更多格式)
    name_score = calculate_name_likelihood(original_anchor_text, is_academic_page)
    score += name_score
    
    # 学术页面特殊加分 - 更灵活的识别
    if is_academic_page:
        # 在学术页面中，简单的人名链接应该得到更高分数
        words = anchor_text.split()
        if (len(words) == 2 and 
            not any(kw in anchor_text for kw in professor_keywords) and
            all(len(word) > 1 for word in words)):  # 确保不是缩写
            # 可能是纯人名链接
            score += 2
            logger.debug(f"Academic page name bonus: {original_anchor_text} -> +2")
        
        # 如果链接文本包含学位信息，额外加分
        if re.search(r'\b(ph\.?d\.?|m\.?d\.?|m\.?s\.?|m\.?a\.?|b\.?a\.?|b\.?s\.?)\b', anchor_text, re.I):
            score += 1
            logger.debug(f"Degree information bonus: {original_anchor_text} -> +1")
    
    # 页面位置分析 (扣分机制) - 保持较低扣分
    if is_link_in_non_content_area(anchor, page_structure):
        score -= 1  # 轻微扣分，因为有些学校的教授链接可能在导航区
    
    # 黑名单关键词 (扣分) - 更精确的过滤
    blacklist_keywords = [
        'home', 'about us', 'contact us', 'news', 'events', 'login', 'search',
        'admin', 'privacy', 'terms', 'cookie', 'sitemap', 'rss', 'subscribe',
        'programs', 'admissions', 'tuition', 'apply', 'campus', 'library',
        'calendar', 'alumni', 'give', 'donate', 'career', 'job'
    ]
    blacklist_matches = sum(1 for kw in blacklist_keywords if kw in anchor_text)
    if blacklist_matches > 0:
        score -= min(blacklist_matches * 1.5, 3)  # 渐进式扣分
    
    # URL模式过滤 (扣分) - 扩展不良模式
    bad_patterns = [
        '/admin/', '/login/', '/search/', '/api/', '/static/', '/css/', '/js/',
        '/assets/', '/images/', '/downloads/', '/resources/', '/forms/', '/application/'
    ]
    if any(pattern in href.lower() for pattern in bad_patterns):
        score -= 3
    
    # 调试日志 - 提高日志质量
    if score > 2:  # 降低日志阈值，记录更多潜在链接
        logger.debug(f"Link scoring: '{original_anchor_text}' -> {href} = {score:.1f} points (academic: {is_academic_page})")
    
    # 确保分数在0-10范围内
    return max(0, min(10, score))


def detect_academic_page_type(base_url: str, page_structure: Dict) -> bool:
    """检测是否为学术机构页面"""
    # URL模式检测 - 扩展支持更多学术机构
    academic_url_patterns = [
        # 通用学术关键词
        'faculty', 'people', 'staff', 'edu/', 'university', 'college', 'school', 'department',
        'professor', 'academic', 'research', 'scholar',
        
        # 顶级域名
        '.edu', '.ac.', 
        
        # 知名大学域名
        'stanford.edu', 'harvard.edu', 'mit.edu', 'berkeley.edu', 'ucla.edu', 'columbia.edu',
        'yale.edu', 'princeton.edu', 'uchicago.edu', 'upenn.edu', 'cornell.edu', 'brown.edu',
        'dartmouth.edu', 'duke.edu', 'northwestern.edu', 'vanderbilt.edu', 'rice.edu',
        'emory.edu', 'georgetown.edu', 'cmu.edu', 'caltech.edu', 'nyu.edu', 'steinhardt.nyu.edu',
        'asc.upenn.edu', 'wharton.upenn.edu', 'seas.upenn.edu',
        
        # 州立大学系统
        'uc.edu', 'csu.edu', 'suny.edu', 'cuny.edu', 'ufl.edu', 'fsu.edu', 'uf.edu',
        'umich.edu', 'msu.edu', 'osu.edu', 'psu.edu', 'rutgers.edu', 'umd.edu', 'vt.edu',
        'unc.edu', 'ncsu.edu', 'clemson.edu', 'sc.edu', 'uga.edu', 'gsu.edu', 'fiu.edu',
        'ucf.edu', 'usf.edu', 'famu.edu', 'fgcu.edu', 'nova.edu', 'barry.edu', 'lynn.edu',
        
        # 国际大学
        'ox.ac.uk', 'cam.ac.uk', 'imperial.ac.uk', 'ucl.ac.uk', 'kcl.ac.uk', 'lse.ac.uk',
        'ed.ac.uk', 'manchester.ac.uk', 'bristol.ac.uk', 'warwick.ac.uk', 'bath.ac.uk',
        'utoronto.ca', 'ubc.ca', 'mcgill.ca', 'sfu.ca', 'uvic.ca', 'ualberta.ca',
        'anu.edu.au', 'unsw.edu.au', 'sydney.edu.au', 'melbourne.edu.au', 'monash.edu.au',
        'nus.edu.sg', 'ntu.edu.sg', 'hku.hk', 'cuhk.edu.hk', 'ust.hk',
        
        # 社区学院和其他教育机构
        'cc.edu', 'edu.', 'academic', 'institute', 'consortium'
    ]
    
    # 检查URL是否匹配学术模式
    url_lower = base_url.lower()
    if any(pattern in url_lower for pattern in academic_url_patterns):
        return True
    
    # 页面结构检测 - 查找学术相关元素
    if page_structure:
        # 检查页面是否包含学术相关的结构元素
        academic_indicators = []
        
        # 检查导航和内容区域是否包含学术关键词
        for area_name in ['navigation_elements', 'content_elements']:
            for element in page_structure.get(area_name, []):
                if element:
                    text = element.get_text().lower()
                    if any(keyword in text for keyword in ['faculty', 'professor', 'research', 'academic', 'department', 'college']):
                        academic_indicators.append(area_name)
                        break
        
        # 如果多个区域都包含学术关键词，认为是学术页面
        if len(academic_indicators) >= 2:
            return True
    
    return False


def calculate_name_likelihood(text: str, is_academic_page: bool) -> int:
    """计算文本是人名的可能性评分 (0-4分)"""
    if not text or len(text) > 50:  # 过长的文本不太可能是人名
        return 0
    
    score = 0
    words = text.split()
    
    # 基本人名模式
    if len(words) == 2:
        # 两个词的组合，很可能是姓名
        first, last = words
        if (first[0].isupper() and last[0].isupper() and 
            len(first) > 1 and len(last) > 1):
            score += 3
            
    elif len(words) == 3:
        # 三个词的组合，可能是 "First Middle Last" 或 "Dr. First Last"
        if all(word[0].isupper() for word in words if len(word) > 1):
            score += 2
            
    # 学术页面上的简化名称格式
    if is_academic_page and len(words) >= 2:
        # 在学术页面上，即使格式不标准也给予一定分数
        if all(word[0].isupper() for word in words if len(word) > 1):
            score += 1
    
    # 常见学术称谓检测
    academic_titles = ['dr.', 'prof.', 'professor', 'dr', 'prof']
    if any(title in text.lower() for title in academic_titles):
        score += 1
    
    # 特殊字符检测（人名中不太可能出现的字符）
    if any(char in text for char in ['@', '#', '$', '%', '&', '*', '(', ')', '[', ']']):
        score -= 2
    
    return max(0, min(4, score))


def clean_research_text(raw_text):
    """清理研究兴趣文本，去除格式化标记和描述性语言"""
    if not raw_text:
        return ""
    
    # 去除markdown格式化
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', raw_text)  # 去除粗体
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # 去除斜体
    text = re.sub(r'`(.*?)`', r'\1', text)  # 去除代码标记
    text = re.sub(r'#{1,6}\s*', '', text)  # 去除标题标记
    
    # 去除HTML标签残留
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除描述性开头语句
    descriptive_patterns = [
        r'^.*?教授.*?研究兴趣.*?[:：]',
        r'^.*?professor.*?research interests.*?[:：]',
        r'^.*?主要研究.*?[:：]',
        r'^.*?研究领域.*?[:：]',
        r'^.*?focuses on.*?[:：]',
        r'^.*?specializes in.*?[:：]',
        r'^.*?的研究兴趣包括.*?[:：]',
        r'^.*?research area.*?[:：]',
    ]
    
    for pattern in descriptive_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 去除结尾的描述性语句
    ending_patterns = [
        r'等相关领域.*$',
        r'等研究方向.*$',
        r'and related areas.*$',
        r'among others.*$',
    ]
    
    for pattern in ending_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 清理多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 去除开头的冒号或句号
    text = re.sub(r'^[：:\.]+\s*', '', text)
    
    return text


def extract_keywords(research_text):
    """从研究兴趣文本中提取3-10个关键词"""
    if not research_text:
        return []
    
    # 这个函数将由LLM来实现，返回关键词列表
    return []  # 占位符，将在get_research_interests中通过LLM实现


def find_related_professor_links(url, session=None):
    """在教授页面内查找相关的CV、出版物、研究等链接"""
    if session is None:
        session = create_session()
    
    related_links = []
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 解析URL以便处理相对链接
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # 相关链接的关键词模式
        related_keywords = {
            'cv': ['cv', 'curriculum vitae', 'resume', '简历'],
            'publications': ['publications', 'papers', 'articles', 'published', '发表', '论文', '出版物'],
            'research': ['research', 'projects', 'ongoing', 'current', '研究', '项目'],
            'teaching': ['teaching', 'courses', 'syllabi', '教学', '课程'],
            'bio': ['biography', 'bio', 'about', '简介', '个人介绍']
        }
        
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            anchor_text = anchor.get_text(strip=True).lower()
            
            # 跳过无效链接
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
                
            # 处理相对链接
            if not href.startswith(("http://", "https://")):
                href = urllib.parse.urljoin(base_url, href)
            
            # 跳过外部链接（不在同一域名下）
            if not href.startswith(base_url):
                continue
                
            # 检查是否匹配相关关键词
            for category, keywords in related_keywords.items():
                if any(keyword in anchor_text for keyword in keywords) or \
                   any(keyword in href.lower() for keyword in keywords):
                    related_links.append({
                        'url': href,
                        'text': anchor_text,
                        'category': category
                    })
                    break
        
        # 去重
        seen_urls = set()
        unique_links = []
        for link in related_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)
        
        logger.info(f"Found {len(unique_links)} related links on {url}")
        return unique_links
        
    except Exception as e:
        logger.error(f"Error finding related links on {url}: {e}")
        return []


def integrate_professor_info(main_url, related_links, session=None, client=None):
    """整合教授主页面和相关页面的信息"""
    if not client:
        return {}
    
    integrated_info = {
        'additional_research_info': '',
        'cv_info': '',
        'publications_info': '',
        'additional_urls': []
    }
    
    # 记录所有相关URL
    integrated_info['additional_urls'] = [link['url'] for link in related_links]
    
    # 尝试获取CV和出版物页面的额外信息
    high_value_categories = ['cv', 'publications', 'research']
    
    for link in related_links:
        if link['category'] in high_value_categories:
            try:
                if session is None:
                    session = create_session()
                    
                response = session.get(link['url'], timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 提取页面内容
                for script in soup(["script", "style"]):
                    script.extract()
                
                content = soup.get_text(separator=" ", strip=True)[:2000]  # 限制长度
                
                # 根据类别存储信息
                if link['category'] == 'cv':
                    integrated_info['cv_info'] += f" {content}"
                elif link['category'] == 'publications':
                    integrated_info['publications_info'] += f" {content}"
                elif link['category'] == 'research':
                    integrated_info['additional_research_info'] += f" {content}"
                    
            except Exception as e:
                logger.warning(f"Could not fetch additional info from {link['url']}: {e}")
                continue
    
    return integrated_info


def robust_llm_call(client, messages, max_retries=3, backoff_factor=1.0):
    """稳健的LLM调用，带有重试机制"""
    import time
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="doubao-1-5-pro-32k-250115",
                messages=messages,
                timeout=30  # 增加超时时间
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # 指数退避
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts")
                raise e


def robust_web_request(session, url, max_retries=3, timeout=15):
    """稳健的网络请求，带有重试机制"""
    import time
    
    for attempt in range(max_retries):
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.warning(f"Web request attempt {attempt + 1} for {url} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # 线性退避
            else:
                logger.error(f"Web request to {url} failed after {max_retries} attempts")
                raise e


def intelligent_parameter_estimation(url: str, session: Optional[requests.Session] = None) -> Dict[str, int]:
    """智能估计最优的分析参数
    
    Args:
        url: 起始页面URL
        session: HTTP会话对象
        
    Returns:
        包含推荐参数的字典 {'max_links': int, 'max_pages': int, 'reasoning': str}
    """
    if session is None:
        session = create_session()
    
    try:
        # 获取首页内容
        response = robust_web_request(session, url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. 分析页面规模和类型
        page_analysis = analyze_page_characteristics(soup, url)
        
        # 2. 检测分页结构
        pagination_info = analyze_pagination_structure(soup, url)
        
        # 3. 计算教授链接密度
        professor_density = calculate_professor_link_density(soup, url)
        
        # 4. 基于分析结果推荐参数
        recommendations = generate_parameter_recommendations(
            page_analysis, pagination_info, professor_density
        )
        
        logger.info(f"智能参数推荐: {recommendations}")
        return recommendations
        
    except Exception as e:
        logger.warning(f"智能参数估计失败，使用默认值: {e}")
        return {
            'max_links': 30,
            'max_pages': 3,
            'reasoning': '参数估计失败，使用默认值'
        }


def analyze_page_characteristics(soup: BeautifulSoup, url: str) -> Dict[str, any]:
    """分析页面特征和规模"""
    analysis = {
        'page_type': 'unknown',
        'estimated_scale': 'medium',
        'has_search_filters': False,
        'total_links': 0,
        'content_depth': 'shallow'
    }
    
    # 分析URL路径，判断页面类型 - 扩展支持更多学校
    url_lower = url.lower()
    
    # 检查是否为教师/教授页面
    if any(keyword in url_lower for keyword in ['/faculty', '/people', '/staff', '/directory', '/profiles']):
        # 进一步细分页面类型
        if any(dept_keyword in url_lower for dept_keyword in ['department', 'dept', 'division', 'program']):
            analysis['page_type'] = 'department'  # 系级页面
        elif any(college_keyword in url_lower for college_keyword in ['college', 'school', 'institute']):
            analysis['page_type'] = 'college'     # 学院级页面
        elif 'graduate' in url_lower or 'phd' in url_lower:
            analysis['page_type'] = 'graduate_faculty'  # 研究生院教师
        else:
            analysis['page_type'] = 'faculty_list'  # 通用教授列表
    
    # 特殊检测知名学校页面类型
    special_patterns = {
        'steinhardt.nyu.edu': 'nyu_steinhardt',
        'asc.upenn.edu': 'upenn_annenberg',
        'seas.upenn.edu': 'upenn_engineering',
        'wharton.upenn.edu': 'upenn_wharton',
        'harvard.edu': 'harvard',
        'mit.edu': 'mit',
        'stanford.edu': 'stanford',
        'berkeley.edu': 'uc_berkeley',
        'columbia.edu': 'columbia'
    }
    
    for pattern, page_type in special_patterns.items():
        if pattern in url_lower and 'faculty' in url_lower:
            analysis['page_type'] = page_type
            analysis['estimated_scale'] = 'large'  # 知名大学通常规模较大
            break
    
    # 检测搜索和过滤功能 - 更精确的检测
    search_indicators = soup.find_all(['input', 'select', 'form', 'button'])
    filter_forms = [elem for elem in search_indicators if 
                   any(keyword in elem.get('class', []) + [elem.get('id', '')] + [elem.get('name', '')] 
                       for keyword in ['search', 'filter', 'sort', 'category', 'department'])]
    analysis['has_search_filters'] = len(filter_forms) > 1
    
    # 统计链接总数
    all_links = soup.find_all('a', href=True)
    analysis['total_links'] = len(all_links)
    
    # 评估内容深度 - 更准确的评估
    text_content = soup.get_text()
    content_length = len(text_content)
    
    # 计算教授相关内容的密度
    professor_keywords = ['professor', 'faculty', 'ph.d', 'phd', 'dr.', 'research', 'department']
    professor_mentions = sum(text_content.lower().count(keyword) for keyword in professor_keywords)
    
    if content_length > 15000 or professor_mentions > 20:
        analysis['content_depth'] = 'deep'
        analysis['estimated_scale'] = 'large'
    elif content_length > 5000 or professor_mentions > 10:
        analysis['content_depth'] = 'medium'
    else:
        analysis['content_depth'] = 'shallow'
        if analysis['page_type'] != 'unknown':
            analysis['estimated_scale'] = 'small'
    
    return analysis


def analyze_pagination_structure(soup: BeautifulSoup, url: str) -> Dict[str, any]:
    """分析分页结构"""
    import re  # 将导入移到函数开头
    
    pagination_info = {
        'has_pagination': False,
        'pagination_type': 'none',
        'estimated_total_pages': 1,
        'items_per_page': 0
    }
    
    # 查找分页指示器
    pagination_selectors = [
        '.pagination', '.pager', '.page-numbers', 
        '[class*="page"]', '[class*="pagination"]'
    ]
    
    pagination_elements = []
    for selector in pagination_selectors:
        elements = soup.select(selector)
        pagination_elements.extend(elements)
    
    if pagination_elements:
        pagination_info['has_pagination'] = True
        
        # 尝试找到页码数字 - 改进过滤逻辑
        page_numbers = []
        for element in pagination_elements:
            # 查找数字，但要过滤掉无关数字
            text_content = element.get_text()
            numbers = re.findall(r'\b\d+\b', text_content)
            
            for num_str in numbers:
                num = int(num_str)
                # 过滤条件：
                # - 大于1（页码从2开始有意义）
                # - 小于1000（页码不太可能超过1000）
                # - 不是常见的邮政编码模式（5位数）
                # - 不是年份（1900-2100）
                if (1 < num < 1000 and 
                    not (10000 <= num <= 99999) and  # 5位邮政编码
                    not (1900 <= num <= 2100)):  # 年份
                    page_numbers.append(num)
        
        if page_numbers:
            pagination_info['estimated_total_pages'] = max(page_numbers)
            pagination_info['pagination_type'] = 'numbered'
            logger.info(f"检测到分页: {max(page_numbers)} 页")
        else:
            # 检查是否有"下一页"类型的分页
            next_indicators = ['next', 'more', '>', '»', '下一页']
            has_next = any(indicator in soup.get_text().lower() for indicator in next_indicators)
            if has_next:
                pagination_info['estimated_total_pages'] = 5  # 保守估计
                pagination_info['pagination_type'] = 'next_only'
    
    # 估计每页条目数（基于当前页面的教授数量）
    professor_like_elements = soup.find_all(['div', 'li', 'tr'], 
                                          string=re.compile(r'professor|dr\.|ph\.d', re.I))
    pagination_info['items_per_page'] = len(professor_like_elements)
    
    return pagination_info


def calculate_professor_link_density(soup: BeautifulSoup, url: str) -> float:
    """计算教授链接密度"""
    total_links = len(soup.find_all('a', href=True))
    if total_links == 0:
        return 0.0
    
    # 使用我们的评分系统快速评估教授相关链接
    page_structure = analyze_page_structure(soup)
    professor_links = 0
    is_academic_page = detect_academic_page_type(url, page_structure)
    
    for anchor in soup.find_all('a', href=True):
        href = anchor['href'].strip()
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
            continue
            
        score = calculate_link_score(anchor, href, page_structure, url)
        # 对学术页面使用更低的阈值来判断教授链接
        threshold = 3 if is_academic_page else 4
        if score > threshold:
            professor_links += 1
    
    density = professor_links / total_links
    logger.info(f"教授链接密度: {professor_links}/{total_links} = {density:.2%}")
    return density


def generate_parameter_recommendations(
    page_analysis: Dict, 
    pagination_info: Dict, 
    professor_density: float
) -> Dict[str, any]:
    """基于分析结果生成参数推荐"""
    
    # 基础参数
    max_links = 30
    max_pages = 3
    reasoning_parts = []
    
    # 1. 根据页面类型调整
    if page_analysis['page_type'] == 'department':
        max_links = 20  # 系级页面通常教授较少
        reasoning_parts.append("系级页面，减少链接数")
    elif page_analysis['page_type'] == 'college':
        max_links = 50  # 学院级页面教授较多
        reasoning_parts.append("学院级页面，增加链接数")
    elif page_analysis['page_type'] == 'nyu_steinhardt':
        max_links = 60  # NYU Steinhardt页面教授很多
        reasoning_parts.append("NYU Steinhardt页面，增加链接数以确保覆盖所有教授")
    
    # 2. 根据教授密度调整
    if professor_density > 0.5:  # 极高密度，说明这是专门的教授页面
        max_links = max(max_links, 60)  # 增加链接数，因为几乎都是教授
        reasoning_parts.append(f"极高教授密度({professor_density:.1%})，增加链接数以覆盖所有教授")
    elif professor_density > 0.3:  # 高密度
        max_links = max(max_links, 40)  # 适度增加链接数
        reasoning_parts.append(f"高教授密度({professor_density:.1%})，增加链接数")
    elif professor_density < 0.1:  # 低密度
        max_links = max(max_links, 50)  # 增加链接数，寻找更多教授
        reasoning_parts.append(f"低教授密度({professor_density:.1%})，增加链接数")
    
    # 3. 根据分页情况调整
    if pagination_info['has_pagination']:
        estimated_pages = pagination_info['estimated_total_pages']
        if estimated_pages <= 3:
            max_pages = estimated_pages
            reasoning_parts.append(f"检测到{estimated_pages}页，跟随所有页面")
        elif estimated_pages <= 10:
            max_pages = min(5, estimated_pages)
            reasoning_parts.append(f"检测到{estimated_pages}页，限制跟随{max_pages}页")
        else:
            max_pages = 3
            reasoning_parts.append(f"检测到{estimated_pages}页，保守跟随{max_pages}页")
    else:
        max_pages = 1
        reasoning_parts.append("未检测到分页，仅分析首页")
    
    # 4. 根据页面规模调整
    if page_analysis['total_links'] > 200:
        # 对于学术页面，尤其是教授密度高的页面，不限制链接数
        if page_analysis['page_type'] in ['nyu_steinhardt', 'faculty_list', 'college'] and professor_density > 0.5:
            reasoning_parts.append("大型学术页面且教授密度高，不限制分析范围")
        else:
            max_links = min(max_links, 40)  # 普通大型页面，控制分析范围
            reasoning_parts.append("大型页面，控制分析范围")
    elif page_analysis['total_links'] < 50:
        max_links = max(max_links, 15)  # 小型页面，确保足够分析
        reasoning_parts.append("小型页面，确保足够分析")
    
    # 5. 安全边界
    max_links = max(10, min(100, max_links))  # 限制在10-100之间
    max_pages = max(1, min(10, max_pages))    # 限制在1-10之间
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "使用默认参数"
    
    return {
        'max_links': max_links,
        'max_pages': max_pages,
        'reasoning': reasoning,
        'page_type': page_analysis['page_type'],
        'professor_density': professor_density,
        'pagination_detected': pagination_info['has_pagination']
    }


def clean_faculty_url(url: str) -> str:
    """清理教师页面URL，去除筛选参数"""
    import urllib.parse
    
    # 解析URL
    parsed = urllib.parse.urlparse(url)
    
    # 特殊处理各种学校的URL模式
    special_cleaning_patterns = [
        # NYU系列
        'steinhardt.nyu.edu',
        'nyu.edu',
        # 宾夕法尼亚大学系列
        'upenn.edu',
        'asc.upenn.edu',
        'wharton.upenn.edu',
        'seas.upenn.edu',
        # 其他知名大学
        'harvard.edu',
        'mit.edu',
        'stanford.edu',
        'berkeley.edu',
        'columbia.edu',
        'yale.edu',
        'princeton.edu'
    ]
    
    # 检查是否为需要特殊处理的学校
    needs_cleaning = any(pattern in parsed.netloc for pattern in special_cleaning_patterns)
    
    if needs_cleaning and any(keyword in parsed.path for keyword in ['faculty', 'people', 'staff', 'directory']):
        # 去除查询参数，保留基础的faculty页面
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # 对于带有筛选参数的URL，尝试简化到基础路径
        if parsed.query:
            # 检查路径是否以这些关键词结尾
            faculty_endpoints = ['faculty', 'people', 'staff', 'directory', 'profiles']
            
            for endpoint in faculty_endpoints:
                if clean_url.endswith(endpoint):
                    return clean_url
                    
            # 如果URL包含这些路径，保留到这些路径为止
            path_parts = parsed.path.split('/')
            for i, part in enumerate(path_parts):
                if part in faculty_endpoints:
                    clean_path = '/'.join(path_parts[:i+1])
                    return f"{parsed.scheme}://{parsed.netloc}{clean_path}"
        
        return clean_url
    
    # 对于其他URL，只是去除明显的筛选参数
    if parsed.query:
        # 保留重要的查询参数，去除筛选参数
        query_params = urllib.parse.parse_qs(parsed.query)
        important_params = {}
        
        # 保留这些参数，因为它们可能是页面结构的一部分
        preserve_params = ['page', 'p', 'dept', 'department', 'college', 'school']
        for param in preserve_params:
            if param in query_params:
                important_params[param] = query_params[param]
        
        if important_params:
            new_query = urllib.parse.urlencode(important_params, doseq=True)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
        else:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    return url


def adaptive_analysis_with_intelligent_params(
    start_url: str, 
    api_key: str, 
    max_workers: int = 5,
    use_intelligent_params: bool = True
) -> List[Dict[str, any]]:
    """使用智能参数推荐的自适应分析
    
    Args:
        start_url: 起始URL
        api_key: OpenAI API密钥
        max_workers: 最大工作线程数
        use_intelligent_params: 是否使用智能参数推荐
        
    Returns:
        分析结果列表
    """
    logger.info(f"开始自适应分析: {start_url}")
    
    # 清理URL，去除筛选参数
    clean_url = clean_faculty_url(start_url)
    if clean_url != start_url:
        logger.info(f"URL已清理: {start_url} -> {clean_url}")
        start_url = clean_url
    
    # 1. 获取智能参数推荐
    if use_intelligent_params:
        logger.info("🧠 获取智能参数推荐...")
        param_recommendations = intelligent_parameter_estimation(start_url)
        max_links = param_recommendations['max_links']
        max_pages = param_recommendations['max_pages']
        logger.info(f"智能推荐 - 链接数: {max_links}, 页面数: {max_pages}")
        logger.info(f"推荐原因: {param_recommendations['reasoning']}")
    else:
        max_links = 30
        max_pages = 3
        logger.info("使用默认参数")
    
    # 2. 执行初步分析
    logger.info(f"🚀 开始分析 (链接: {max_links}, 页面: {max_pages})")
    df_results = analyze_webpage_links(start_url, api_key, max_links, max_workers, max_pages)
    results = df_results.to_dict('records')
    
    # 3. 分析初步结果并决定是否需要调整
    if use_intelligent_params and results:
        adjustment = analyze_results_and_adjust_params(results, max_links, max_pages)
        
        if adjustment['should_adjust']:
            logger.info(f"📊 结果分析: {adjustment['reason']}")
            logger.info(f"调整参数 - 新链接数: {adjustment['new_max_links']}, 新页面数: {adjustment['new_max_pages']}")
            
            # 执行调整后的分析
            additional_df = analyze_webpage_links(
                start_url, 
                api_key, 
                adjustment['new_max_links'], 
                max_workers,
                adjustment['new_max_pages']
            )
            additional_results = additional_df.to_dict('records')
            
            # 合并结果并去重
            all_results = results + additional_results
            seen_urls = set()
            deduplicated_results = []
            
            for result in all_results:
                url = result.get('URL', '')  # 修正字段名：URL而不是url
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_results.append(result)
            
            logger.info(f"合并结果: {len(results)} + {len(additional_results)} = {len(deduplicated_results)} (去重后)")
            results = deduplicated_results
    
    return results


def analyze_results_and_adjust_params(
    results: List[Dict[str, any]], 
    current_max_links: int, 
    current_max_pages: int
) -> Dict[str, any]:
    """分析初步结果并决定参数调整策略"""
    
    professor_results = [r for r in results if r.get('Is Professor Page') == 'Yes']
    professor_count = len(professor_results)
    total_results = len(results)
    
    adjustment = {
        'should_adjust': False,
        'reason': '',
        'new_max_links': current_max_links,
        'new_max_pages': current_max_pages
    }
    
    # 计算教授发现率
    professor_rate = professor_count / total_results if total_results > 0 else 0
    
    # 分析是否需要调整参数
    if professor_count == 0:
        # 没有找到任何教授，大幅增加搜索范围
        adjustment.update({
            'should_adjust': True,
            'reason': f'未发现教授，增加搜索范围',
            'new_max_links': min(current_max_links * 2, 100),
            'new_max_pages': min(current_max_pages + 2, 8)
        })
    elif professor_count < 20 and professor_rate > 0.4:
        # 教授发现率高但数量少，可能是参数太保守，增加搜索范围
        adjustment.update({
            'should_adjust': True,
            'reason': f'教授发现率高({professor_rate:.1%})但数量少({professor_count}个)，增加搜索范围',
            'new_max_links': min(current_max_links * 2, 100),
            'new_max_pages': min(current_max_pages + 1, 6)
        })
    elif professor_count < 5 and professor_rate < 0.2:
        # 教授数量较少且发现率低，适度增加搜索
        adjustment.update({
            'should_adjust': True,
            'reason': f'教授数量少({professor_count}个)且发现率低({professor_rate:.1%})，增加搜索',
            'new_max_links': min(current_max_links + 20, 80),
            'new_max_pages': min(current_max_pages + 1, 6)
        })
    elif professor_rate > 0.8 and current_max_links > 60:
        # 发现率很高且已经搜索了很多链接，参数已足够
        adjustment.update({
            'should_adjust': False,
            'reason': f'发现率很高({professor_rate:.1%})且搜索范围充足，当前参数已足够'
        })
    
    logger.info(f"结果分析 - 教授: {professor_count}/{total_results} ({professor_rate:.1%})")
    
    return adjustment


def test_nyu_steinhardt_fixes():
    """测试NYU Steinhardt页面的修复效果"""
    # 创建测试会话
    session = create_session()
    
    # 测试URL
    test_url = "https://steinhardt.nyu.edu/about/faculty"
    
    logger.info("🧪 测试NYU Steinhardt修复效果...")
    
    try:
        # 获取页面内容
        response = robust_web_request(session, test_url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 分析页面结构
        page_structure = analyze_page_structure(soup)
        
        # 测试几个样本链接
        test_cases = [
            "Dianna Heldman",
            "Dave Pietro", 
            "Juan Pablo Bello",
            "Marilyn Nonken"
        ]
        
        logger.info("测试教授姓名链接评分:")
        for name in test_cases:
            # 创建模拟的anchor元素
            mock_html = f'<a href="/people/{name.lower().replace(" ", "-")}">{name}</a>'
            mock_soup = BeautifulSoup(mock_html, 'html.parser')
            mock_anchor = mock_soup.find('a')
            href = f"https://steinhardt.nyu.edu/people/{name.lower().replace(' ', '-')}"
            
            score = calculate_link_score(mock_anchor, href, page_structure, test_url)
            logger.info(f"  {name}: {score} 分")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
    
    logger.info("✅ NYU Steinhardt修复测试完成")


def test_multi_university_compatibility():
    """测试多个大学的兼容性"""
    logger.info("🌍 测试多学校兼容性...")
    
    # 测试用的大学URL列表
    test_universities = [
        {
            'name': '宾夕法尼亚大学传播学院',
            'url': 'https://www.asc.upenn.edu/people/faculty',
            'expected_professors': 10
        },
        {
            'name': 'NYU Steinhardt',
            'url': 'https://steinhardt.nyu.edu/about/faculty',
            'expected_professors': 25
        },
        {
            'name': '爱荷华大学新闻学院',
            'url': 'https://journalism.uiowa.edu/people',
            'expected_professors': 15
        },
        {
            'name': '哥伦比亚大学新闻学院',
            'url': 'https://journalism.columbia.edu/faculty',
            'expected_professors': 20
        },
        {
            'name': '斯坦福大学传播系',
            'url': 'https://comm.stanford.edu/people/faculty',
            'expected_professors': 15
        }
    ]
    
    session = create_session()
    results = []
    
    for university in test_universities:
        try:
            logger.info(f"📊 测试 {university['name']}...")
            
            # 测试页面访问
            try:
                response = session.get(university['url'], timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                page_accessible = response.status_code == 200
                page_size = len(response.text) if page_accessible else 0
            except Exception:
                page_accessible = False
                page_size = 0
            
            # 如果页面可访问，测试智能参数推荐
            if page_accessible:
                try:
                    recommendations = intelligent_parameter_estimation(university['url'], session)
                    
                    # 测试链接提取
                    links = get_all_links(university['url'], session, follow_pagination=False, max_pages=1)
                    
                    # 统计教授相关链接
                    professor_links = []
                    for link in links:
                        if any(pattern in link.lower() for pattern in ['/faculty/', '/people/', '/profile', 'professor']):
                            professor_links.append(link)
                    
                    result = {
                        'university': university['name'],
                        'url': university['url'],
                        'page_accessible': True,
                        'page_size': page_size,
                        'recommended_max_links': recommendations['max_links'],
                        'recommended_max_pages': recommendations['max_pages'],
                        'page_type': recommendations.get('page_type', 'unknown'),
                        'professor_density': recommendations.get('professor_density', 0),
                        'total_links_found': len(links),
                        'professor_links_found': len(professor_links),
                        'expected_professors': university['expected_professors'],
                        'success_rate': len(professor_links) / university['expected_professors'] if university['expected_professors'] > 0 else 0
                    }
                    
                except Exception as e:
                    result = {
                        'university': university['name'],
                        'url': university['url'],
                        'page_accessible': True,
                        'page_size': page_size,
                        'error': str(e)
                    }
            else:
                result = {
                    'university': university['name'],
                    'url': university['url'],
                    'page_accessible': False,
                    'error': 'Page not accessible'
                }
            
            results.append(result)
            logger.info(f"✅ {university['name']} 测试完成")
            
        except Exception as e:
            logger.error(f"❌ {university['name']} 测试失败: {e}")
            results.append({
                'university': university['name'],
                'url': university['url'],
                'error': f'Test failed: {str(e)}'
            })
    
    # 输出测试结果摘要
    logger.info("📋 多学校测试结果摘要:")
    logger.info("=" * 60)
    
    accessible_count = sum(1 for r in results if r.get('page_accessible', False))
    total_count = len(results)
    
    logger.info(f"页面可访问性: {accessible_count}/{total_count} ({accessible_count/total_count:.1%})")
    
    for result in results:
        if result.get('page_accessible'):
            logger.info(f"✅ {result['university']}")
            logger.info(f"   页面类型: {result.get('page_type', 'unknown')}")
            logger.info(f"   教授密度: {result.get('professor_density', 0):.1%}")
            logger.info(f"   推荐参数: {result.get('recommended_max_links', 'N/A')} 链接, {result.get('recommended_max_pages', 'N/A')} 页面")
            logger.info(f"   找到链接: {result.get('total_links_found', 0)} 总计, {result.get('professor_links_found', 0)} 教授")
            if 'success_rate' in result:
                logger.info(f"   成功率: {result['success_rate']:.1%}")
        else:
            logger.info(f"❌ {result['university']}: {result.get('error', 'Unknown error')}")
        logger.info("")
    
    return results


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a website to find professor pages and research interests."
    )
    parser.add_argument("url", nargs='?', help="Starting URL to analyze")
    parser.add_argument("--api-key", help="API key for OpenAI services")
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
    parser.add_argument(
        "--test-fixes", action="store_true", help="Test NYU Steinhardt fixes"
    )
    parser.add_argument(
        "--test-multi-university", action="store_true", help="Test multi-university compatibility"
    )
    args = parser.parse_args()
    
    # 如果是测试模式，运行测试并退出
    if args.test_fixes:
        test_nyu_steinhardt_fixes()
        exit(0)
    
    # 如果是多学校测试模式，运行多学校测试并退出
    if args.test_multi_university:
        test_multi_university_compatibility()
        exit(0)

    # 对于正常运行模式，检查必需参数
    if not args.url:
        parser.error("URL is required for normal analysis mode")
    if not args.api_key:
        parser.error("API key is required for normal analysis mode")

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
