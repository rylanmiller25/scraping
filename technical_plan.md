# Summary

This markdown file includes instructions and guidelines for a monthly comprehensive scraping project of all startups founded since January 2026, which we source from PitchBook. Taking the websites for each of these startups, this project involves scraping the text from the homepages of these websites at the beginning of each month. The eventual goal is to create a firm-month panel measuring changes in website text, firm-level entries and exits, and name changes. Since each month we will be pulling from PitchBook all startups founded since January 2026, the scraping will encapsulate both startups existing in that sample from the beginning as well as startups founded after the scraping project has begun.

# Formatting and Final Scraping Output 

The scraper employed here should extract all of the text from each given website domain. I should be able to give the scraper a list of website domains, and the scraper should be able to take each of those, scrape the raw HTML from the homepage and up to 9 additional first-level links (same domain), and then store the combined visible text (deleting the HTML after storing the text) in a dataframe where a column exists for the company ID called 'companyid', for the company name called 'company', for the domain called 'website', for the year called 'year', for the month called 'month', for the text called 'text', and an indicator for failures called 'failure'. The final output should be saved as a Parquet file.

**Text Deduplication**: To save space and highlight changes, the scraper should compare the scraped text for month `t` against the text from month `t-1` (if available).
- If the scraped text for a company is **identical** to the text scraped in the previous month, the `text` column for the current month should contain a single dash `'-'`.
- If the text has changed, or if there is no record for the previous month (e.g., new entry), store the full scraped text.
- This requires the scraper to accept an optional input for the "previous month's data file" to perform the comparison.

# Guidelines

- **Language & Tooling**: 
    - The scraper must be written in Python.
    - It must use Playwright to handle dynamic content (Single Page Applications, React/Next.js sites) and ensure all visible text is captured.

- **Content Extraction & Crawling**:
    - The goal is to capture all visible text from the website.
    - **Scope**: The scraper must visit the homepage of each startup. It should then identify and visit up to **9 other first-level links** (links that point to pages within the same domain, e.g., "About Us", "Team", "Product") found on the homepage.
    - **Total Pages**: A maximum of 10 pages per company (1 homepage + 9 subpages) should be scraped.
    - **Extraction**: For each visited page, render the full HTML, extract all visible text, and then discard the HTML. The text from all pages should be aggregated for that company.
    - **Filtering**:
        - Exclude pages that are empty, contain only boilerplate (e.g., just a nav bar), represent errors (404s), or are non-English.
        - Ensure the extracted text is cleaned of HTML tags, scripts, and styles.

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

- **URL Normalization**: Since input URLs lack any prefixes, the scraper **must prepend the appropriate protocols and subdomains** (e.g., `https://www.`) to the `website` string before attempting navigation (e.g., `example.com` becomes `https://www.example.com`). The scraper handles potential redirects or protocol fallbacks as needed.
- **Date Handling**: The `year` and `month` columns in the output dataframe should be populated based on the current system date when the script is run. This ensures the "firm-month" panel structure tracks when the data was actually observed.
- **Timeouts**: To prevent the scraper from hanging on broken or slow websites, a strict timeout of 30 seconds should be enforced for each page load. If a page fails to load within this window, it should be recorded as a failure.

# Environment Setup

- The project requires the following Python libraries:
    - `playwright`: For browser automation and rendering dynamic content.
    - `pandas`: For data manipulation and creating the output dataframe.
    - `pyarrow` (or `fastparquet`): For saving the output dataframe as a Parquet file.
    - `aiohttp` or similar: May be needed for efficient asynchronous handling if not fully covered by Playwright's native async capabilities.
- The environment must have the Playwright browsers installed (run `playwright install`).
- Despite all of these libraries, prioritize feasibility and operability over fanciness. I want this scraper to work correctly. Don't do too much such that it doesn't work.
