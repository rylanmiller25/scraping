import re
import logging
import sys
from typing import List

def setup_logging(log_file: str = "scraper.log"):
    """
    Sets up logging to both console and file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("startup_scraper")

def normalize_text(text: str) -> str:
    """
    Normalizes text for robust change detection:
    1. Lowercase.
    2. Replace all sequences of whitespace with a single space.
    3. Trim leading/trailing whitespace.
    """
    if not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Whitespace Collapsing (regex \s+ matches space, tab, newline, etc.)
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Trimming
    text = text.strip()
    
    return text

def truncate_text(text: str, max_chars: int = 500000) -> str:
    """
    Enforces a hard limit on text length to prevent memory issues.
    """
    if not text:
        return ""
    return text[:max_chars]

def get_url_variations(domain: str) -> List[str]:
    """
    Generates the prioritized list of URL prefixes for a given domain
    as specified in the implementation details.
    
    Order:
    1. https://www.
    2. https://
    3. http://www.
    4. http://
    """
    # Remove any existing protocol or www if inadvertently passed, 
    # though input is expected to be clean domain.
    clean_domain = domain.lower().replace("http://", "").replace("https://", "").replace("www.", "")
    
    return [
        f"https://www.{clean_domain}",
        f"https://{clean_domain}",
        f"http://www.{clean_domain}",
        f"http://{clean_domain}"
    ]

def clean_url_for_deduplication(url: str) -> str:
    """
    Removes common tracking parameters for deduplication purposes.
    Retains other query parameters.
    """
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    
    # List of tracking parameters to remove
    tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'gclid', 'fbclid'}
    
    # Filter out tracking params
    new_query_params = {k: v for k, v in query_params.items() if k.lower() not in tracking_params}
    
    # Reconstruct query string
    new_query = urlencode(new_query_params, doseq=True)
    
    # Reconstruct URL
    return urlunparse(parsed._replace(query=new_query))

