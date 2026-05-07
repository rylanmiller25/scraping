import asyncio
import random
import logging
import os
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
    log_error,
)

logger = logging.getLogger("startup_scraper")

def _brightdata_proxy_url() -> str:
    """
    Builds an authenticated proxy URL from environment variables (GitHub secrets).

    Expected env vars:
      - BRIGHTDATA_RES_PROXY_HOST
      - BRIGHTDATA_RES_PROXY_PORT
      - BRIGHTDATA_RES_PROXY_USERNAME
      - BRIGHTDATA_RES_PROXY_PASSWORD
    """
    host = os.getenv("BRIGHTDATA_RES_PROXY_HOST", "").strip()
    port = os.getenv("BRIGHTDATA_RES_PROXY_PORT", "").strip()
    username = os.getenv("BRIGHTDATA_RES_PROXY_USERNAME", "").strip()
    password = os.getenv("BRIGHTDATA_RES_PROXY_PASSWORD", "").strip()

    if not (host and port and username and password):
        return ""

    # Bright Data proxies are typically HTTP proxies, even for HTTPS targets.
    return f"http://{username}:{password}@{host}:{port}"


def _crawl4ai_proxy_config() -> object:
    """
    Returns a Crawl4AI-compatible proxy_config value.

    Crawl4AI supports:
      - a ProxyConfig instance (newer versions), or
      - a dict with {server, username, password}, or
      - a string URL.
    We avoid importing ProxyConfig directly to stay compatible with different versions.
    """
    proxy_url = _brightdata_proxy_url()
    if not proxy_url:
        return None

    # Prefer dict format (works across documented versions).
    # server should include scheme and host:port.
    # Note: we pass credentials separately even though they are also embedded in proxy_url.
    host = os.getenv("BRIGHTDATA_RES_PROXY_HOST", "").strip()
    port = os.getenv("BRIGHTDATA_RES_PROXY_PORT", "").strip()
    username = os.getenv("BRIGHTDATA_RES_PROXY_USERNAME", "").strip()
    password = os.getenv("BRIGHTDATA_RES_PROXY_PASSWORD", "").strip()
    server = f"http://{host}:{port}"
    return {"server": server, "username": username, "password": password}


class ScrapeResult:
    def __init__(self):
        self.full_text: str = ""
        self.success: int = 0  # 0 or 1
        self.failure_reason: str = "success"
        self.failure_detail: str = ""
        self.num_pages_scraped: int = 0
        self.urls_visited: List[str] = []
        self.text_length: int = 0


def _classify_network_exception(exc: BaseException) -> Tuple[str, str]:
    """
    Attempts to classify common network/proxy errors into stable reason codes.
    Returns (reason_code, detail_string).
    """
    detail = repr(exc)

    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", detail

    # Proxy-specific errors
    if isinstance(exc, aiohttp.ClientHttpProxyError):
        # Often indicates auth problems or proxy refusing connection.
        status = getattr(exc, "status", None)
        if status == 407:
            return "proxy_auth_error", detail
        return "proxy_http_error", detail
    if isinstance(exc, aiohttp.ClientProxyConnectionError):
        return "proxy_connect_error", detail

    # TLS / cert
    if isinstance(exc, aiohttp.ClientConnectorCertificateError):
        return "tls_error", detail
    if isinstance(exc, aiohttp.ClientSSLError):
        return "tls_error", detail

    # DNS / connect
    if isinstance(exc, aiohttp.ClientConnectorError):
        return "connect_error", detail

    # HTTP-level
    if isinstance(exc, aiohttp.ClientResponseError):
        return "http_error", detail

    if isinstance(exc, aiohttp.InvalidURL):
        return "invalid_url", detail

    if isinstance(exc, OSError):
        return "os_error", detail

    return "other_error", detail


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
    proxy_url = _brightdata_proxy_url()
    ssl_param = False if proxy_url else None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                robots_url,
                timeout=10,
                proxy=proxy_url or None,
                ssl=ssl_param,
            ) as response:
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
    except Exception as e:
        # Network error on robots.txt usually means we can't reach it,
        # but the main scrape might also fail.
        # We'll assume allowed for now and let the main scraper hit the error if the site is truly down.
        reason, detail = _classify_network_exception(e)
        logger.info(f"robots.txt check error for {robots_url}: {reason} {detail}")
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
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    proxy_url = _brightdata_proxy_url()
    ssl_param = False if proxy_url else None

    try:
        # 1. URL Normalization
        target_urls = get_url_variations(raw_domain)
        valid_homepage_url = None

        # 2. Pre-checks BEFORE launching Playwright: skip browser if we'd scrape nothing
        # 2a. Robots: if first URL is disallowed, fail without opening browser
        if target_urls:
            is_allowed, _ = await check_robots_txt(target_urls[0], user_agent)
            if not is_allowed:
                result.failure_reason = "robots_disallowed"
                result.success = 0
                return _compile_result(company_row, result)

        # 2b. Quick connectivity check (10s): if site doesn't respond, fail without opening browser
        if target_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        target_urls[0],
                        timeout=aiohttp.ClientTimeout(total=10),
                        proxy=proxy_url or None,
                        ssl=ssl_param,
                    ) as resp:
                        pass  # any response means we might get content
            except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
                reason, detail = _classify_network_exception(e)
                logger.warning(
                    f"Pre-check failed for {target_urls[0]}: {reason} {detail}. Skipping Playwright."
                )
                result.failure_reason = reason
                result.failure_detail = detail
                result.success = 0
                return _compile_result(company_row, result)

        # 3. Configuration and browser launch (only if pre-checks passed)
        # Use Crawl4AI's built-in navigation timeout in addition to our asyncio-level timeout.
        proxy_config = _crawl4ai_proxy_config()
        run_config = CrawlerRunConfig(
            verbose=False,
            cache_mode=CacheMode.BYPASS,
            page_timeout=30000,  # ms
            wait_until="domcontentloaded",
            proxy_config=proxy_config,
        )
        # Crawl4AI's BrowserConfig does not expose launch timeout; Playwright default (180s) is used.
        browser_config = BrowserConfig(
            user_agent=user_agent,
            headless=True,
            verbose=False,
        )

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

                    # Attempt to load the page with retries
                    scrape_result = None
                    for attempt in range(3):
                        try:
                            logger.info(
                                f"Attempting to scrape {url} (Attempt {attempt + 1}/3)..."
                            )
                            # Enforce a hard per-page timeout of 30 seconds
                            scrape_result = await asyncio.wait_for(
                                crawler.arun(url=url, config=run_config), timeout=30
                            )
                            if scrape_result.success:
                                logger.info(f"Successfully scraped {url}")
                                break
                            else:
                                logger.warning(
                                    f"Failed to scrape {url} (Attempt {attempt + 1})"
                                )
                        except asyncio.CancelledError:
                            raise
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout scraping {url} (Attempt {attempt + 1}) after 30 seconds."
                            )
                            if attempt == 2:
                                result.failure_reason = "timeout"
                        except Exception as e:
                            logger.error(f"Exception scraping {url}: {e}")
                            # Only wait if we are going to retry this specific URL
                            if attempt < 2:
                                wait_time = 15  # Long wait between retries as requested
                                logger.info(
                                    f"Homepage retry {attempt + 1}/3 for {url} after {wait_time}s..."
                                )
                                await asyncio.sleep(wait_time)

                    if scrape_result and scrape_result.success:
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
                            # If text is very short, langdetect fails.
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

                            # Random delay 2-5s between pages
                            await asyncio.sleep(random.uniform(2, 5))

                            try:
                                # Retry logic for subpages
                                sub_result = None
                                for attempt in range(3):
                                    try:
                                        logger.info(
                                            f"Attempting to scrape subpage {link} (Attempt {attempt + 1}/3)..."
                                        )
                                        # Enforce a hard per-page timeout of 30 seconds
                                        sub_result = await asyncio.wait_for(
                                            crawler.arun(url=link, config=run_config),
                                            timeout=30,
                                        )
                                        if sub_result.success:
                                            logger.info(
                                                f"Successfully scraped subpage {link}"
                                            )
                                            break
                                        else:
                                            logger.warning(
                                                f"Failed subpage {link} (Attempt {attempt + 1})"
                                            )
                                    except asyncio.CancelledError:
                                        raise
                                    except asyncio.TimeoutError:
                                        logger.warning(
                                            f"Timeout scraping subpage {link} (Attempt {attempt + 1}) after 30 seconds."
                                        )
                                    except Exception as e:
                                        if attempt < 2:
                                            wait_time = 15  # Long wait between retries
                                            logger.info(
                                                f"Subpage retry {attempt + 1}/3 for {link} after {wait_time}s..."
                                            )
                                            await asyncio.sleep(wait_time)
                                        else:
                                            logger.warning(
                                                f"Failed to scrape subpage {link} after 3 attempts: {e}"
                                            )

                                if sub_result and sub_result.success:
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

                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                logger.warning(
                                    f"Error scraping subpage {link} for {company_id}: {e}"
                                )

                        # Success!
                        result.success = 1
                        result.failure_reason = "success"
                        break  # Stop trying other prefixes

                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    result.failure_reason = "timeout"
                except aiohttp.ClientConnectorError:
                    result.failure_reason = "dns_error"  # or connection error
                except Exception as e:
                    # Map some common SSL errors if possible, otherwise generic
                    err_str = str(e).lower()
                    if "ssl" in err_str or "certificate" in err_str:
                        result.failure_reason = "tls_error"
                        result.failure_detail = repr(e)
                    else:
                        result.failure_reason = "http_error"  # generic fallback
                        result.failure_detail = repr(e)

        # --- Finalize Result ---

        if not valid_homepage_url:
            # If we exhausted the loop without success, 'result.failure_reason'
            # holds the reason for the LAST attempt.
            result.success = 0

        # Global text limit
        result.full_text = truncate_text(result.full_text.strip())
        result.text_length = len(result.full_text)

        return _compile_result(company_row, result)
    except asyncio.CancelledError:
        # Allow upstream timeouts/cancellation to interrupt promptly.
        raise
    except Exception as e:
        # Any unexpected error in the scraping flow gets logged with file and line info.
        log_error(e, __file__)
        logger.error(f"Unhandled error while processing company {company_id}: {e}")
        result.success = 0
        result.failure_reason = "unhandled_exception"
        result.failure_detail = repr(e)
        result.full_text = ""
        result.text_length = 0
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
    row["failure_detail"] = result.failure_detail
    row["num_pages_scraped"] = result.num_pages_scraped
    row["text_length"] = result.text_length
    # 'similarity_score' and 'has_change' will be calculated in main.py
    return row
