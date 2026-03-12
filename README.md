# Startup Website Scraper

This project scrapes the websites of startups founded since January 2026, analyzes the text content, and tracks changes month-over-month using NLP embeddings.

## Project Structure

-   `Code/`: Contains all Python source code.
    -   `main.py`: The entry point for the scraper.
    -   `scraper.py`: Core scraping logic using Crawl4AI.
    -   `nlp.py`: NLP engine for calculating similarity scores.
    -   `utils.py`: Helper functions for normalization and URL handling.
    -   `requirements.txt`: Python dependencies.
-   `Input Data/`: Place monthly input Parquet files here (e.g., `pb_02_2026.parquet`).
-   `Output Data/`:
    -   `Raw Text Datasets/`: Stores the full extracted text (Parquet).
    -   `Analysis Datasets/`: Stores the analysis metrics (Stata `.dta`).
-   `Scraping Guidelines/`: Contains the technical plan and documentation.

## Setup

1.  **Install Dependencies**:
    Navigate to the `Code` directory and install the required packages:
    ```bash
    cd Code
    pip install -r requirements.txt
    playwright install
    ```

2.  **Prepare Input Data**:
    Ensure the input Parquet file for the current month is placed in the `Input Data/` folder.
    -   Naming convention: `pb_{month}_{year}.parquet` (e.g., `pb_02_2026.parquet`).
    -   Required columns: `companyid`, `website`, `yearfounded`, `companyname`, `companyformername`.

## Usage

To run the scraper for the current month:

1.  Open a terminal.
2.  Navigate to the `Code` directory:
    ```bash
    cd "/Users/rylanmiller/Desktop/Startup Positioning/scraping/Code"
    ```
3.  Run the main script:
    ```bash
    python main.py
    ```

## Tracking Errors

-   Before processing, all variable names are normalized by lowercase such that there should not be any discrepancies in naming conventions.
-   **Error folder**: Any unexpected errors during a run are logged to text files in the `Error/` folder at the project root.
-   **Per-month error file**: When `main.py` starts and finds the monthly input file (for example, `pb_03_2026.parquet`), it creates or overwrites a matching error file named `pb_03_2026_error.txt` inside `Error/`.
-   **Recorded information**: Each time an unhandled exception occurs in `main.py`, `scraper.py`, or `nlp.py`, a new entry is appended to that month’s error file, including:
    1. What the error is (the Python exception message),
    2. Which `.py` file was running when it happened,
    3. The specific line of code that raised the error (line number and source text).
-   Additionally, any individual page (homepage or subpage) that takes longer than roughly 30 seconds to load is treated as a timeout and skipped for that run; this behavior will be reflected in the error logs when relevant.

## Logic Overview

-   **Frequency**: The scraper is designed to be run once at the beginning of each month.
-   **URL Handling**: It attempts `https://www.`, `https://`, `http://www.`, and `http://` in that order.
-   **Robots.txt**: Respects `robots.txt` rules.
-   **Filtering**: Skips non-English pages and blocked/CAPTCHA pages.
-   **Deduplication**: If a company's text hasn't changed since the last scrape, it stores a `-` instead of duplicating the text, but calculates a similarity score of 1.0.
-   **Resume Capability**: The script checks for existing output files and resumes where it left off if interrupted.
