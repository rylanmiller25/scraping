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
    cd Code
    ```
3.  Run the main script:
    ```bash
    python main.py
    ```

## Logic Overview

-   **Frequency**: The scraper is designed to be run once at the beginning of each month.
-   **URL Handling**: It attempts `https://www.`, `https://`, `http://www.`, and `http://` in that order.
-   **Robots.txt**: Respects `robots.txt` rules.
-   **Filtering**: Skips non-English pages and blocked/CAPTCHA pages.
-   **Deduplication**: If a company's text hasn't changed since the last scrape, it stores a `-` instead of duplicating the text, but calculates a similarity score of 1.0.
-   **Resume Capability**: The script checks for existing output files and resumes where it left off if interrupted.
