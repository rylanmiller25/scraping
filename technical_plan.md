# Summary

This markdown file includes instructions and guidelines for a monthly comprehensive scraping project of all startups founded since 2024, which we source from PitchBook. Taking the websites for each of these startups, this project involves scraping the text from the homepages of these websites at the beginning of each month. The eventual goal is to create a firm-month panel measuring changes in website text, firm-level entries and exits, and name changes. Since we will be pulling from PitchBook all startups founded since 2024 each month, the scraping will encapsulate both startups existing in that sample from the beginning as well as startups founded after the scraping project has begun.

# Formatting

The scraper employed here should extract all of the text (and only the text, not the raw HTML) from each given website domain given. I should be able to give the scraper a list of website domains, and the scraper should be able to take each of those, scrape the text alone from each website, and then store that text in a dataframe where a column exists for the company called 'companyid', for the domain called 'website', for the year called 'year', for the month called 'month', for the text called 'text', and an indicator for failures called 'failure'. The final output should be saved as a Parquet file.

# Guidelines

- **Language & Tooling**: 
    - The scraper must be written in Python.
    - It must use Playwright to handle dynamic content (Single Page Applications, React/Next.js sites) and ensure all visible text is captured.

- **Content Extraction**:
    - The goal is to capture any text whatsoever that would appear when you go to a website.
    - The scraper should render the full homepage and extract all visible text content, ignoring HTML tags, scripts, and styles.

- **Etiquette & Performance**:
    - Robots.txt: The scraper must check and respect the `robots.txt` file for each domain before attempting to scrape.
    - Speed: The scraping process should be deliberate and not overly aggressive. While concurrency can be used since targets are distinct domains, avoiding high-volume bursts is preferred to prevent network issues or blocking. A limit of 5 concurrent browsers is recommended for local execution on a standard machine.
    - Retries: Implement a retry mechanism (e.g., 3 attempts) with backoff for transient network errors.

- **Error Handling**: 
    - The scraper must implement robust error handling.
    - If a request fails (e.g., connection timeout, DNS error, non-200 status code), the script should not terminate.
    - Instead, it should log the specific error for the corresponding `companyid` and `website`.
    - The output dataframe should contain a record for the failed attempt with a null value for `text` and a value of 1 under `failure`.

# Input Data

- The input to the scraper will be a CSV file containing at least the following columns:
    - `companyid`: A unique identifier for the company.
    - `website`: The full URL or domain of the company's homepage. The websites in this file will be formatted with the `www.` prefix (e.g., `www.example.com`) but will *not* include the protocol (`http://` or `https://`).

# Implementation Details

- **URL Normalization**: Since input URLs lack a protocol, the scrape must prepend `https://` to the `website` string before attempting navigation (e.g., `www.example.com` becomes `https://www.example.com`).
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
