# Summary

This markdown file includes instructions and guidelines for a monthly comprehensive scraping project of all startups founded since January 2026, which we source from PitchBook. Taking the websites for each of these startups, this project involves scraping the text from the homepages of these websites at the beginning of each month. The eventual goal is to create a firm-month panel measuring changes in website text, firm-level entries and exits, and name changes. Since each month we will be pulling from PitchBook all startups founded since January 2026, the scraping will encapsulate both startups existing in that sample from the beginning as well as startups founded after the scraping project has begun. Additionally, this project will utilize Natural Language Processing (NLP) to quantify the similarity of a firm's website text from month to month, providing a robust measure of change over time.

# Formatting and Final Scraping Output 

The project should produce **two separate output files** for each run to optimize storage and analysis:

1.  **Text Archive (Heavy Storage)**:
    -   **Format**: Parquet (`Output Data/Raw Text Datasets/raw_{month}_{year}.parquet`).
    -   **Content**: `companyid`, `year`, `month`, and the full `text`.
    -   **Purpose**: Long-term storage of the raw/cleaned text for NLP or auditing.
    -   **Note on Naming**: `{month}` is 2-digit.

2.  **Analysis Panel (Light Analysis)**:
    -   **Format**: Stata (`Output Data/Analysis Datasets/analysis_{month}_{year}.dta`).
    -   **Content**:
        -   `companyid`: Unique identifier.
        -   `companyname`: Company name.
        -   `companyformername`: Former company name (may be missing).
        -   `website`: Domain.
        -   `yearfounded`: Year founded (may be missing).
        -   `year`, `month`: Time of scraping.
        -   `failure`: Binary indicator (0/1).
        -   `failure_reason`: Reason code (e.g., `robots_disallowed`).
        -   `similarity_score`: Cosine similarity to previous month (1.0 = no change).
        -   `has_change`: Binary flag (1 if text changed, 0 if same).
        -   `num_pages_scraped`: Count of usable-text pages successfully scraped (1-10).
        -   `text_length`: Total number of characters in the aggregated text.

**Text Deduplication & State Management**: 
- **Goal**: To save space, if the text for a company has not changed since the last *valid* text entry, store a single dash `'-'`.
- **Mechanism**: The scraper must maintain a **"Reference State"** (e.g., a local Parquet file `latest_state.parquet`) that stores the *last seen full text* for every company.
- **Logic**:
    1. Scrape the current text for a company (`current_text`).
    2. Look up the company's text in the **Reference State** file (`ref_text`).
    3. **Comparison**:
        - If `current_text == ref_text`: Store `'-'` in the *monthly output file*. Do *not* update the Reference State.
        - If `current_text != ref_text` (or if company is new): Store `current_text` in the *monthly output file*. **Update** the Reference State with this new `current_text`.
- **Outcome**: This ensures that even if months `t`, `t+1`, and `t+2` are all dashes, month `t+3` is still compared against the text from month `t-1` (the last time it changed), satisfying the requirement to compare against the last known actual text.

# Guidelines

- **Language & Tooling**: 
    - The scraper must be written in Python.
    - It must use **Crawl4AI** (which utilizes Playwright under the hood) as the primary scraping library. Crawl4AI is chosen for its ability to produce clean, LLM-ready markdown/text and handle dynamic content efficiently.

- **Content Extraction & Crawling**:
    - The goal is to capture all visible text from the website in a clean format.
    - **Scope**: The scraper must visit the homepage of each startup. It should then identify and visit up to **9 other first-level links** (links that point to pages within the same domain, e.g., "About Us", "Team", "Product") found on the homepage.
        - **Implementation note**: With Crawl4AI (Playwright), the scraper does **not** need a separate, manual "download raw HTML" step. Playwright loads/renders the page; link discovery can be done from the rendered DOM, and plain visible text extraction can be done directly via Crawl4AI on the loaded page.
    - **Order & Aggregation**: 
        1. **Order**: Scrape the homepage first, then proceed to the subpages.
            - Only proceed to subpages if the homepage yields usable text.
        2. **Joining**: The text extracted from all pages (homepage + up to 9 subpages) must be **joined together into a single continuous string**, separated only by spaces. Do not use delimiters (like "---Page Break---") or structured formats (like JSON lists). The result should be one unified block of text representing the company's entire web presence.
    - **Total Pages**: A maximum of 10 pages per company (1 homepage + 9 subpages) should be scraped.
    - **Extraction**: For each visited page, use Crawl4AI to extract the **clean plain text**. Do not use Markdown output, as the formatting symbols (like `**`, `##`, `[]`) are unnecessary for the NLP analysis and could interfere with strict text comparison. The text from all pages should be aggregated for that company.
    - **Normalization**: Before storage or comparison, the aggregated text must be normalized to ensure robust change detection:
        1.  **Lowercasing**: Convert all text to lowercase.
        2.  **Whitespace Collapsing**: Replace all sequences of whitespace (spaces, tabs, newlines) with a single space.
        3.  **Trimming**: Remove leading and trailing whitespace.
    - **Sanity Limit**: To prevent memory issues with outlier websites, enforce a hard limit of **500,000 characters** per company. If the aggregated text exceeds this limit, truncate it to the first 500,000 characters.
    - **Filtering**:
        - Exclude pages that are empty, represent errors (404s), or are non-English.
        - Ensure the extracted text is free of HTML tags (Crawl4AI handles this by default).

- **Etiquette & Performance**:
    - **Robots.txt Policy**:
        - **User Agent**: The scraper should use a standard browser User-Agent string (e.g., Chrome) to mimic a real user visiting the site.
        - **Compliance**: Check `robots.txt` for the target domain.
            - If the **homepage** is disallowed: Abort scraping for this company and mark `failure_reason` as `robots_disallowed`.
            - If a **subpage** is disallowed: Skip that specific link but continue scraping allowed pages.
    - Speed: The scraping process should be deliberate and not overly aggressive. While concurrency can be used since targets are distinct domains, avoiding high-volume bursts is preferred to prevent network issues or blocking. A limit of 5 concurrent browsers is recommended for local execution on a standard machine.
    - Timing: Implement a randomized delay of 2 to 5 seconds between navigation actions on the same domain (i.e., between the homepage and subsequent subpages). This variation makes the traffic pattern look less robotic and helps avoid rate limits.
    - Retries: Implement a retry mechanism (e.g., 3 attempts) with backoff for transient network errors.

- **Error Handling**: 
    - The scraper must implement robust error handling.
    - If a request fails (e.g., connection timeout, DNS error, non-200 status code), the script should not terminate.
    - Instead, it should log the specific error for the corresponding `companyid` and `website`.
    - The output dataframe should contain a record for the failed attempt with a null value for `text` and a value of 1 under `failure`.
    - **Failure Reasons**: A `failure_reason` column must be populated with one of the following statuses to indicate the cause:
        - `success`: No failure.
        - `robots_disallowed`: Access blocked by robots.txt.
        - `timeout`: Page load exceeded 30 seconds.
        - `dns_error`: Domain name could not be resolved.
        - `tls_error`: SSL/TLS handshake failed.
        - `http_error`: Non-200 HTTP status code (e.g., 404, 500).
        - `blocked_captcha`: Request blocked by a CAPTCHA or anti-bot challenge.
        - `empty_text`: Page loaded but contained no visible text.
        - `non_english`: Detected language was not English.
        - `other_error`: Any other unclassified exception.
    - **Success definition (critical)**:
        - A firm is considered a **success** if and only if the **homepage yields usable text**.
        - If the homepage does **not** yield usable text, the firm is a failure and subpages should not be attempted.
    - **`num_pages_scraped` definition (critical)**:
        - `num_pages_scraped` counts **usable-text pages** (after extraction + normalization + filtering). It includes the homepage if it yields usable text.

# Input Data

- **Format**: Parquet.
- **Location**: Input datasets will be placed in the `Input Data/` folder (a subdirectory of the overall `Scraping/` directory).
- **Naming convention**: `pb_{month}_{year}.parquet` (there will be many input datasets, and they will all follow this convention).
    - `{month}` should be a **2-digit** month (e.g., `02` for February).
- **Columns**:
    - `companyid` (**dtype: object**): Unique identifier for the company. **There will be no duplicates of `companyid`.**
    - `website` (**dtype: object**): The domain of the company's homepage. Domains are provided **without any prefixes** (e.g., `example.com`). They will not include `www.`, `http://`, or `https://`.
    - `yearfounded` (**dtype: float64**): Year founded (may be missing).
    - `companyname` (**dtype: object**): Company name.
    - `companyformername` (**dtype: object**): Former company name (may be missing).
- **Context**: Each monthly file represents a cross-section of startups founded since January 2026 (the scraper begins running in February 2026).

# Implementation Details

- **URL Normalization & Smart Fallback**: Since input URLs lack prefixes, the scraper must attempt to find the correct accessible URL by trying a prioritized list of prefixes.
    - **Logic**: For a given domain (e.g., `example.com`), the scraper should attempt to connect in the following order:
        1. `https://www.example.com`
        2. `https://example.com`
        3. `http://www.example.com`
        4. `http://example.com`
    - **Success Condition**: As soon as one prefix returns a valid response (status 200 OK), the scraper should proceed with that URL and skip the remaining prefixes for that domain.
    - **Failure Condition**: If all 4 variations fail (or time out), record the attempt as a failure.
    - **Query parameter handling (default)**:
        - When selecting/deduplicating homepage-discovered links, strip only common tracking parameters (e.g., `utm_*`, `gclid`, `fbclid`) but retain other query parameters by default.
- **Date Handling**: The `year` and `month` columns in the output dataframe should be populated based on the current system date when the script is run. This ensures the "firm-month" panel structure tracks when the data was actually observed.
- **Timeouts**: To prevent the scraper from hanging on broken or slow websites, a strict timeout of 30 seconds should be enforced for each page load. If a page fails to load within this window, it should be recorded as a failure.
- **Execution & Concurrency Strategy**:
    - **Mode**: The scraper must use Python's `asyncio` for asynchronous execution.
    - **Sorting**: Process companies in deterministic order (sorted by `companyid`).
    - **Concurrency Control**:
        - **Global**: Max 5 concurrent companies (browsers) active at once.
        - **Per-Domain**: Strictly **1 page at a time** (sequential scraping of homepage -> subpages). Do *not* scrape subpages in parallel.
    - **Domain restriction (critical)**:
        - Subpages must be restricted to the same **registrable domain** as the homepage (e.g., allow `www.example.com` but exclude `blog.example.com`).
    - **Reproducibility**: Set a fixed seed (e.g., `42`) for the random number generator to ensure delay patterns are reproducible for debugging.
- **Resumability & Checkpointing**:
    - The script should support stopping and restarting without losing progress.
    - **Logic**:
        1. On startup, check if the output file (or a checkpoint file) exists.
        2. Read the list of `companyid`s already attempted (successes and failures).
        3. Filter the input Parquet dataset to exclude these attempted IDs.
        4. Resume scraping only the remaining companies.
    - **Write Strategy**: Write results incrementally (e.g., append to Parquet or write batch files) to ensure data is saved even if the script crashes.

# Environment Setup

- The project requires the following Python libraries:
    - `crawl4ai`: The primary library for scraping and cleaning web content.
    - `playwright`: Required by Crawl4AI for browser automation.
    - `pandas`: For data manipulation and creating the output dataframe.
    - `pyarrow` (or `fastparquet`): For saving the output dataframe as a Parquet file.
    - `aiohttp`: May be needed for efficient asynchronous handling.
    - `sentence-transformers`: For loading the Jina embedding model.
    - `transformers`, `torch`: Dependencies for the embedding pipeline.
- The environment must have the Playwright browsers installed (run `playwright install`).
- Despite all of these libraries, prioritize feasibility and operability over fanciness. I want this scraper to work correctly. Don't do too much such that it doesn't work.

# Natural Language Processing

- **Goal**: To measure the semantic similarity of a startup's website text between the current month (`t`) and the prior month (`t-1`), creating a metric for website change.
- **Model**: The project will use the **Jina Embeddings** model `jinaai/jina-embeddings-v2-base-en` (available on Hugging Face).
    - **Reasoning**: This model supports a long context window of **8,192 tokens** (approx. 6,000 words), which allows it to embed the entire aggregated website content in a single pass without complex chunking.
- **Metric**: The primary metric will be the **Cosine Similarity** between the embedding of the current month's text and the embedding of the previous month's text.
- **Logic**:
    1.  **Token Limit & Truncation**: Before embedding, check the token length of the aggregated text.
        - **Tokenizer**: Use `AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)` to ensure accurate token counts matching the model.
        - If it exceeds **8,192 tokens**, **deterministically truncate** the text to the first 8,192 tokens. This preserves the homepage and top subpages (since scraping order is homepage-first).
    2.  **Generate Embedding**: For each company, generate a vector embedding of the (potentially truncated) text using `jinaai/jina-embeddings-v2-base-en`.
    3.  **Comparison**: Calculate the cosine similarity score between the embedding for month `t` and the embedding for month `t-1`.
    4.  **Dash Handling**: The process must be robust to the deduplication logic (where `'-'` represents no change).
        - If the text for month `t` is a dash `'-'`, the similarity score is automatically **1.0** (perfect similarity).
        - If the text for month `t` is different (i.e., not a dash), compare it against the **Reference State** (the last valid text) to compute the actual similarity score.
