import asyncio
import random
import logging
import aiohttp
from urllib.robotparser import RobotFileParser
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

# Language detection
from langdetect import detect, LangDetectException

# crawl4ai imports - assuming standard API based on plan description
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from utils import (
    normalize_text,
    truncate_text,
    get_url_variations,
    clean_url_for_deduplication,
)

logger = logging.getLogger("startup_scraper")


class ScrapeResult:
    def __init__(self):
        self.full_text: str = ""
        self.success: int = 0  # 0 or 1
        self.failure_reason: str = "success"
        self.num_pages_scraped: int = 0
        self.urls_visited: List[str] = []
        self.text_length: int = 0


async def check_robots_txt(
    base_url: str, user_agent: str = "*"
) -> Tuple[bool, Optional[RobotFileParser]]:
    """
    Checks robots.txt for the given base_url.
    Returns (is_allowed, parser_object).
    If robots.txt is missing or unreachable, defaults to True (allowed).
    """
    robots_url = urljoin(base_url, "/robots.txt")
    parser = RobotFileParser()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    parser.parse(content.splitlines())
                    return parser.can_fetch(user_agent, base_url), parser
                elif response.status in [401, 403]:
                    # If robots.txt is forbidden, standard practice is to assume FULL DISALLOW
                    return False, None
                else:
                    # 404 or other implies allowed
                    return True, None
    except Exception:
        # Network error on robots.txt usually means we can't reach it,
        # but the main scrape might also fail.
        # We'll assume allowed for now and let the main scraper hit the error if the site is truly down.
        return True, None


def is_captcha_or_blocked(text: str) -> bool:
    """
    Checks for common CAPTCHA or blocking messages in the text.
    """
    block_keywords = [
        "verify you are human",
        "please complete the security check",
        "access denied",
        "access to this page has been denied",
        "security challenge",
        "cloudflare ray id",
        "enable javascript and cookies",
        "attention required!",
        "pardon our interruption",
    ]
    text_lower = text.lower()
    for keyword in block_keywords:
        if keyword in text_lower:
            return True
    return False


async def process_company(company_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrapes a single company's website (homepage + up to 9 subpages).

    Args:
        company_row: Dictionary containing 'companyid', 'website', etc.

    Returns:
        Dictionary with scraping results added/updated.
    """
    company_id = company_row.get("companyid")
    raw_domain = company_row.get("website")

    result = ScrapeResult()

    # 1. URL Normalization & Connection Attempt
    target_urls = get_url_variations(raw_domain)
    valid_homepage_url = None

    # Configuration for the crawler
    run_config = CrawlerRunConfig(verbose=False, cache_mode=CacheMode.BYPASS)

    # User Agent
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    browser_config = BrowserConfig(user_agent=user_agent, headless=True, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # --- Step 1: Find Valid Homepage ---
        for url in target_urls:
            try:
                # Check robots.txt first
                is_allowed, robot_parser = await check_robots_txt(url, user_agent)

                if not is_allowed:
                    result.failure_reason = "robots_disallowed"
                    result.success = 0
                    return _compile_result(company_row, result)

                # Attempt to load the page
                scrape_result = await crawler.arun(url=url, config=run_config)

                if scrape_result.success:
                    # Check for empty text
                    if not scrape_result.markdown:
                        # Potentially empty
                        pass

                    cleaned_text = normalize_text(scrape_result.markdown)

                    if not cleaned_text:
                        # Empty text on homepage -> soft failure, try next prefix
                        result.failure_reason = "empty_text"
                        continue

                    # Check for CAPTCHA / Block
                    if is_captcha_or_blocked(cleaned_text):
                        result.failure_reason = "blocked_captcha"
                        # If blocked, likely blocked on all prefixes, but we can abort or continue.
                        # Usually a hard block.
                        result.success = 0
                        return _compile_result(company_row, result)

                    # Check Language
                    try:
                        # Detect on a substring to save time/memory, e.g. first 1000 chars
                        lang = detect(cleaned_text[:1000])
                        if lang != "en":
                            result.failure_reason = "non_english"
                            result.success = 0
                            return _compile_result(company_row, result)
                    except LangDetectException:
                        # Could not detect language (too short? numbers?)
                        # We'll assume it's okay or mark as empty/other.
                        # If text is very short, langdetect fails.
                        # If it passed empty check, let's allow it but warn.
                        pass

                    # If we passed all checks:
                    valid_homepage_url = url
                    result.full_text += cleaned_text + " "
                    result.num_pages_scraped += 1
                    result.urls_visited.append(url)

                    # Extract internal links
                    internal_links = scrape_result.links.get("internal", [])
                    subpages = _filter_subpages(internal_links, valid_homepage_url)

                    # --- Step 2: Process Subpages ---
                    count = 0
                    for link in subpages:
                        if count >= 9:
                            break

                        # Check robots for subpage if we have a parser
                        if robot_parser and not robot_parser.can_fetch(
                            user_agent, link
                        ):
                            logger.info(f"Skipping subpage {link} due to robots.txt")
                            continue

                        # Random delay 2-5s
                        await asyncio.sleep(random.uniform(2, 5))

                        try:
                            sub_result = await crawler.arun(url=link, config=run_config)
                            if sub_result.success:
                                sub_text = normalize_text(sub_result.markdown)
                                if sub_text and not is_captcha_or_blocked(sub_text):
                                    result.full_text += sub_text + " "
                                    result.num_pages_scraped += 1
                                    result.urls_visited.append(link)
                                    count += 1
                            else:
                                logger.warning(
                                    f"Failed to scrape subpage {link} for {company_id}"
                                )

                        except Exception as e:
                            logger.warning(
                                f"Error scraping subpage {link} for {company_id}: {e}"
                            )

                    # Success!
                    result.success = 1
                    result.failure_reason = "success"
                    break  # Stop trying other prefixes

            except asyncio.TimeoutError:
                result.failure_reason = "timeout"
            except aiohttp.ClientConnectorError:
                result.failure_reason = "dns_error"  # or connection error
            except Exception as e:
                # Log specific error if needed, try next prefix
                # Map some common SSL errors if possible, otherwise generic
                err_str = str(e).lower()
                if "ssl" in err_str or "certificate" in err_str:
                    result.failure_reason = "tls_error"
                else:
                    result.failure_reason = "http_error"  # generic fallback
                pass

    # --- Finalize Result ---

    if not valid_homepage_url:
        # If we exhausted the loop without success, 'result.failure_reason'
        # holds the reason for the LAST attempt.
        result.success = 0

    # Global text limit
    result.full_text = truncate_text(result.full_text.strip())
    result.text_length = len(result.full_text)

    return _compile_result(company_row, result)


def _filter_subpages(links: List[Dict], homepage_url: str) -> List[str]:
    """
    Filters and prioritizes subpages:
    - Same registrable domain (treating www and root as same).
    - Up to 9.
    - Deduplicate using clean_url_for_deduplication.
    """
    # Normalize homepage domain (strip www.)
    parsed_home = urlparse(homepage_url)
    home_domain = parsed_home.netloc.lower().replace("www.", "")

    candidates = []
    seen = set()

    # Add homepage to seen to avoid loops
    seen.add(clean_url_for_deduplication(homepage_url))

    for link in links:
        href = link.get("href", "")
        if not href:
            continue

        # Ensure absolute URL (Crawl4AI usually provides this)
        if not href.startswith("http"):
            continue  # skip relative if not resolved

        # Check domain
        parsed_link = urlparse(href)
        link_domain = parsed_link.netloc.lower().replace("www.", "")

        # Strict match on domain (excluding www prefix)
        # This allows example.com <-> www.example.com
        # But excludes blog.example.com (since 'blog.example.com' != 'example.com')
        if link_domain != home_domain:
            continue

        clean_link = clean_url_for_deduplication(href)

        if clean_link not in seen:
            candidates.append(href)
            seen.add(clean_link)

    return candidates[:9]


def _compile_result(company_row: Dict, result: ScrapeResult) -> Dict:
    """
    Merges scraping result back into company row.
    """
    row = company_row.copy()
    row["text"] = result.full_text if result.success else None
    row["failure"] = 0 if result.success else 1
    row["failure_reason"] = result.failure_reason
    row["num_pages_scraped"] = result.num_pages_scraped
    row["text_length"] = result.text_length
    # 'similarity_score' and 'has_change' will be calculated in main.py
    return row
