# Summary

This markdown file includes instructions and guidelines for a monthly comprehensive scraping project of all startups founded since January 2026, which we source from PitchBook. Taking the websites for each of these startups, this project involves scraping the text from the homepages of these websites at the beginning of each month. The eventual goal is to create a firm-month panel measuring changes in website text, firm-level entries and exits, and name changes. Since each month we will be pulling from PitchBook all startups founded since January 2026, the scraping will encapsulate both startups existing in that sample from the beginning as well as startups founded after the scraping project has begun. Additionally, this project will utilize Natural Language Processing (NLP) to quantify the similarity of a firm's website text from month to month, providing a robust measure of change over time.

# Formatting and Final Scraping Output 

The scraper employed here should extract all of the text from each given website domain. I should be able to give the scraper a list of website domains, and the scraper should be able to take each of those, scrape the raw HTML from the homepage and up to 9 additional first-level links (same domain), and then store the combined visible text (deleting the HTML after storing the text) in a dataframe where a column exists for the company ID called 'companyid', for the company name called 'company', for the domain called 'website', for the year called 'year', for the month called 'month', for the text called 'text', and an indicator for failures called 'failure'. The final output should be saved as a Parquet file.

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
    - **Order & Aggregation**: 
        1. **Order**: Scrape the homepage first, then proceed to the subpages.
        2. **Joining**: The text extracted from all pages (homepage + up to 9 subpages) must be **joined together into a single continuous string**, separated only by spaces. Do not use delimiters (like "---Page Break---") or structured formats (like JSON lists). The result should be one unified block of text representing the company's entire web presence.
    - **Total Pages**: A maximum of 10 pages per company (1 homepage + 9 subpages) should be scraped.
    - **Extraction**: For each visited page, use Crawl4AI to extract the **clean plain text**. Do not use Markdown output, as the formatting symbols (like `**`, `##`, `[]`) are unnecessary for the NLP analysis and could interfere with strict text comparison. The text from all pages should be aggregated for that company.
    - **Normalization**: Before storage or comparison, the aggregated text must be normalized to ensure robust change detection:
        1.  **Lowercasing**: Convert all text to lowercase.
        2.  **Whitespace Collapsing**: Replace all sequences of whitespace (spaces, tabs, newlines) with a single space.
        3.  **Trimming**: Remove leading and trailing whitespace.
    - **Filtering**:
        - Exclude pages that are empty, represent errors (404s), or are non-English.
        - Ensure the extracted text is free of HTML tags (Crawl4AI handles this by default).

- **Etiquette & Performance**:
    - Robots.txt: The scraper must check and respect the `robots.txt` file for each domain before attempting to scrape.
    - Speed: The scraping process should be deliberate and not overly aggressive. While concurrency can be used since targets are distinct domains, avoiding high-volume bursts is preferred to prevent network issues or blocking. A limit of 5 concurrent browsers is recommended for local execution on a standard machine.
    - Timing: Implement a randomized delay of 2 to 5 seconds between navigation actions on the same domain (i.e., between the homepage and subsequent subpages). This variation makes the traffic pattern look less robotic and helps avoid rate limits.
    - Retries: Implement a retry mechanism (e.g., 3 attempts) with backoff for transient network errors.

- **Error Handling**: 
    - The scraper must implement robust error handling.
    - If a request fails (e.g., connection timeout, DNS error, non-200 status code), the script should not terminate.
    - Instead, it should log the specific error for the corresponding `companyid` and `website`.
    - The output dataframe should contain a record for the failed attempt with a null value for `text` and a value of 1 under `failure`.

# Input Data

- The input to the scraper will be a CSV file containing at least the following columns:
    - `company`: The company's unique name in the given month and year, stored as a string.
    - `companyid`: A unique identifier for the company.
    - `website`: The domain of the company's homepage. The websites in this file will be formatted **without any prefixes** (e.g., `example.com`). They will not include `www.`, `http://`, or `https://`.
- **Context**: This file represents a cross-section of US startups founded since January 2026 (the scraper will begin running in February 2026).

# Implementation Details

- **URL Normalization & Smart Fallback**: Since input URLs lack prefixes, the scraper must attempt to find the correct accessible URL by trying a prioritized list of prefixes.
    - **Logic**: For a given domain (e.g., `example.com`), the scraper should attempt to connect in the following order:
        1. `https://www.example.com`
        2. `https://example.com`
        3. `http://www.example.com`
        4. `http://example.com`
    - **Success Condition**: As soon as one prefix returns a valid response (status 200 OK), the scraper should proceed with that URL and skip the remaining prefixes for that domain.
    - **Failure Condition**: If all 4 variations fail (or time out), record the attempt as a failure.
- **Date Handling**: The `year` and `month` columns in the output dataframe should be populated based on the current system date when the script is run. This ensures the "firm-month" panel structure tracks when the data was actually observed.
- **Timeouts**: To prevent the scraper from hanging on broken or slow websites, a strict timeout of 30 seconds should be enforced for each page load. If a page fails to load within this window, it should be recorded as a failure.

# Environment Setup

- The project requires the following Python libraries:
    - `crawl4ai`: The primary library for scraping and cleaning web content.
    - `playwright`: Required by Crawl4AI for browser automation.
    - `pandas`: For data manipulation and creating the output dataframe.
    - `pyarrow` (or `fastparquet`): For saving the output dataframe as a Parquet file.
    - `aiohttp`: May be needed for efficient asynchronous handling.
- The environment must have the Playwright browsers installed (run `playwright install`).
- Despite all of these libraries, prioritize feasibility and operability over fanciness. I want this scraper to work correctly. Don't do too much such that it doesn't work.

# Natural Language Processing

- **Goal**: To measure the semantic similarity of a startup's website text between the current month (`t`) and the prior month (`t-1`), creating a metric for website change.
- **Model**: The project will use the **BERT** embeddings model `intfloat/e5-base-v2` (available on Hugging Face).
- **Metric**: The primary metric will be the **Cosine Similarity** between the embedding of the current month's text and the embedding of the previous month's text.
- **Logic**:
    1.  **Generate Embedding**: For each company, generate a vector embedding of the scraped text using `intfloat/e5-base-v2`.
    2.  **Comparison**: Calculate the cosine similarity score between the embedding for month `t` and the embedding for month `t-1`.
    3.  **Dash Handling**: The process must be robust to the deduplication logic (where `'-'` represents no change).
        - If the text for month `t` is a dash `'-'`, the similarity score is automatically **1.0** (perfect similarity).
        - If the text for month `t` is different (i.e., not a dash), compare it against the **Reference State** (the last valid text) to compute the actual similarity score.
