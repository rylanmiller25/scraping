import asyncio
import datetime

from analyze_month import run_analysis
from scrape_month import run_scrape
from utils import setup_logging

logger = setup_logging()


async def main() -> None:
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    logger.info(f"Starting full pipeline for {year}-{month:02d}")
    await run_scrape(year, month)
    run_analysis(year, month)
    logger.info("Full pipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())
