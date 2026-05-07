import asyncio
import datetime
import os
from typing import Any, Dict, List

import pandas as pd

from scraper import process_company
from utils import init_error_file_from_input_path, log_error, setup_logging

INPUT_DIR = "../Input Data"
OUTPUT_RAW_DIR = "../Output Data/Raw Text Datasets"
BATCH_SIZE = 50
COMPANY_TIMEOUT_SECONDS = 180
MAX_CONCURRENCY = 2

logger = setup_logging()


def get_input_file(year: int, month: int) -> str:
    month_str = f"{month:02d}"
    filename = f"pb_{month_str}_{year}.parquet"
    filepath = os.path.join(INPUT_DIR, filename)
    return filepath if os.path.exists(filepath) else ""


def get_raw_output_path(year: int, month: int) -> str:
    return os.path.join(OUTPUT_RAW_DIR, f"raw_{month:02d}_{year}.parquet")


def _append_raw_rows(rows: List[Dict[str, Any]], raw_path: str) -> None:
    new_df = pd.DataFrame(rows)
    if os.path.exists(raw_path):
        existing_raw = pd.read_parquet(raw_path)
        combined_raw = pd.concat([existing_raw, new_df], ignore_index=True)
        combined_raw.to_parquet(raw_path, index=False)
    else:
        new_df.to_parquet(raw_path, index=False)


async def run_scrape(year: int, month: int) -> None:
    input_file = get_input_file(year, month)
    if not input_file:
        logger.error(
            f"No input file found for {year}-{month:02d}. Expected in {INPUT_DIR}"
        )
        return

    init_error_file_from_input_path(input_file)
    logger.info(f"Reading input data from {input_file}")
    input_df = pd.read_parquet(input_file)
    input_df.columns = [c.lower() for c in input_df.columns]

    raw_output_path = get_raw_output_path(year, month)

    processed_ids = set()
    if os.path.exists(raw_output_path):
        logger.info(f"Found existing output file {raw_output_path}. Resuming...")
        existing_df = pd.read_parquet(raw_output_path)
        if "companyid" in existing_df.columns:
            processed_ids = set(existing_df["companyid"])
        logger.info(f"Already processed {len(processed_ids)} companies.")

    companies_to_process = input_df[~input_df["companyid"].isin(processed_ids)].copy()
    companies_to_process.sort_values("companyid", inplace=True)
    total_to_process = len(companies_to_process)
    logger.info(f"Remaining companies to process: {total_to_process}")

    if total_to_process == 0:
        logger.info("All companies already scraped for this month.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    company_rows = companies_to_process.to_dict("records")
    tasks = []
    results_buffer: List[Dict[str, Any]] = []

    async def process_wrapper(row: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            task = asyncio.create_task(process_company(row))
            try:
                return await asyncio.wait_for(task, timeout=COMPANY_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Company {row.get('companyid')} timed out after {COMPANY_TIMEOUT_SECONDS}s, skipping."
                )
                task.cancel()
                fail_row = row.copy()
                fail_row["text"] = None
                fail_row["failure"] = 1
                fail_row["failure_reason"] = "timeout"
                fail_row["num_pages_scraped"] = 0
                fail_row["text_length"] = 0
                return fail_row

    for i, row in enumerate(company_rows):
        tasks.append(process_wrapper(row))
        if len(tasks) >= BATCH_SIZE or i == total_to_process - 1:
            logger.info(f"Scraping batch ending at index {i}...")
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res_row in batch_results:
                if isinstance(res_row, BaseException):
                    log_error(res_row, __file__)
                    logger.error(f"Task failed with exception: {res_row}")
                    continue
                res_row["year"] = year
                res_row["month"] = month
                results_buffer.append(res_row)

            if results_buffer:
                _append_raw_rows(results_buffer, raw_output_path)
                results_buffer = []
            tasks = []

    logger.info(f"Scrape complete. Raw output written to {raw_output_path}")


async def main() -> None:
    # Default to current time (in GitHub Actions this is typically UTC).
    now = datetime.datetime.now()

    # Optional overrides for CI/manual runs (e.g. GitHub Actions workflow_dispatch).
    env_year = os.getenv("SCRAPE_YEAR")
    env_month = os.getenv("SCRAPE_MONTH")
    year = int(env_year) if env_year else now.year
    month = int(env_month) if env_month else now.month

    logger.info(f"Starting scrape job for {year}-{month:02d}")
    await run_scrape(year, month)


if __name__ == "__main__":
    asyncio.run(main())
