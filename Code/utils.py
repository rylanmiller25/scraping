import re
import logging
import sys
import os
import traceback
import linecache
from typing import List, Optional


def setup_logging(log_file: str = "scraper.log"):
    """
    Sets up logging to both console and file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("startup_scraper")


# --- Error Logging Helpers ---

ERROR_DIR = os.path.join(os.path.dirname(__file__), "..", "Error")
ERROR_FILE_PATH: Optional[str] = None


def init_error_file_from_input_path(input_path: str) -> str:
    """
    Initializes the error log file for a given input parquet path.
    The file is created in the Error/ folder and named:
        pb_MM_YYYY_error.txt
    based on the input file name: pb_MM_YYYY.parquet.
    """
    global ERROR_FILE_PATH

    os.makedirs(ERROR_DIR, exist_ok=True)

    base_name = os.path.basename(input_path)
    # Expect pattern pb_MM_YYYY.parquet
    match = re.match(r"pb_(\d{2})_(\d{4})\.parquet$", base_name)
    if match:
        month_str, year_str = match.groups()
    else:
        # Fallback: if pattern unexpected, don't block execution; use a generic name.
        month_str, year_str = "00", "0000"

    error_filename = f"pb_{month_str}_{year_str}_error.txt"
    ERROR_FILE_PATH = os.path.join(ERROR_DIR, error_filename)

    # Start fresh each run for the given month/year.
    with open(ERROR_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"Error log for input file: {base_name}\n\n")

    return ERROR_FILE_PATH


def log_error(exc: BaseException, py_file: str) -> None:
    """
    Logs an error to the monthly error file, if initialized.

    The log entry includes:
      1) The error message.
      2) The .py file where logging is invoked.
      3) The specific line of code that raised the error (line number and source).
    """
    if not ERROR_FILE_PATH:
        # If the error file was never initialized, we do not attempt to create it here
        # to avoid masking the original control flow; just return silently.
        return

    # Walk to the deepest traceback frame
    tb = exc.__traceback__
    last_tb = tb
    while last_tb and last_tb.tb_next:
        last_tb = last_tb.tb_next

    line_no = last_tb.tb_lineno if last_tb else -1
    code_filename = last_tb.tb_frame.f_code.co_filename if last_tb else ""
    code_line = linecache.getline(code_filename, line_no).strip() if line_no > 0 else ""

    with open(ERROR_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("=== ERROR ===\n")
        f.write(f"Error: {repr(exc)}\n")
        f.write(f"Py File: {py_file}\n")
        if line_no > 0:
            f.write(f"Code Line: {line_no}: {code_line}\n")
        else:
            f.write("Code Line: <unavailable>\n")
        f.write("\n")


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
    text = re.sub(r"\s+", " ", text)

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
    # Remove any existing protocol.
    # Input is expected to be in 'www.example.com' format, but we clean thoroughly just in case.
    clean_domain = domain.lower().replace("http://", "").replace("https://", "")

    # We strip www. to build the base, then re-add it in variations.
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]

    return [
        f"https://www.{clean_domain}",
        f"https://{clean_domain}",
        f"http://www.{clean_domain}",
        f"http://{clean_domain}",
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
    tracking_params = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
    }

    # Filter out tracking params
    new_query_params = {
        k: v for k, v in query_params.items() if k.lower() not in tracking_params
    }

    # Reconstruct query string
    new_query = urlencode(new_query_params, doseq=True)

    # Reconstruct URL
    return urlunparse(parsed._replace(query=new_query))
