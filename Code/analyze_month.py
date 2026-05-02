import datetime
import os
from typing import Dict

import pandas as pd

from nlp import NLPEngine
from scrape_month import get_input_file, get_raw_output_path
from utils import init_error_file_from_input_path, log_error, setup_logging

OUTPUT_ANALYSIS_DIR = "../Output Data/Analysis Datasets"
REF_STATE_FILE = "latest_state.parquet"

logger = setup_logging()


def load_reference_state() -> Dict[str, str]:
    if not os.path.exists(REF_STATE_FILE):
        logger.info("No reference state file found. Starting with empty state.")
        return {}
    try:
        df = pd.read_parquet(REF_STATE_FILE)
        if "companyid" not in df.columns or "text" not in df.columns:
            logger.error("Reference state file missing required columns.")
            return {}
        return dict(zip(df["companyid"], df["text"]))
    except Exception as e:
        logger.error(f"Error loading reference state: {e}")
        return {}


def save_reference_state(state: Dict[str, str]) -> None:
    logger.info("Saving updated reference state...")
    try:
        df = pd.DataFrame(list(state.items()), columns=["companyid", "text"])
        df.to_parquet(REF_STATE_FILE, index=False)
        logger.info("Reference state saved.")
    except Exception as e:
        logger.error(f"Failed to save reference state: {e}")


def get_analysis_output_path(year: int, month: int) -> str:
    return os.path.join(OUTPUT_ANALYSIS_DIR, f"analysis_{month:02d}_{year}.dta")


def run_analysis(year: int, month: int) -> None:
    input_file = get_input_file(year, month)
    if input_file:
        init_error_file_from_input_path(input_file)

    raw_path = get_raw_output_path(year, month)
    if not os.path.exists(raw_path):
        logger.error(f"No raw scrape file found for {year}-{month:02d}: {raw_path}")
        return

    logger.info(f"Reading raw data from {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    raw_df.columns = [c.lower() for c in raw_df.columns]

    ref_state = load_reference_state()
    nlp_engine = NLPEngine()

    rows = raw_df.to_dict("records")
    analysis_rows = []
    for row in rows:
        try:
            company_id = row.get("companyid")
            current_text = row.get("text")
            failure = int(row.get("failure", 1))

            similarity_score = 0.0
            has_change = 0

            if failure == 0 and current_text:
                ref_text = ref_state.get(company_id)
                if ref_text == current_text:
                    similarity_score = 1.0
                    has_change = 0
                else:
                    has_change = 1
                    if ref_text:
                        similarity_score = nlp_engine.compute_similarity(
                            current_text, ref_text
                        )
                    else:
                        similarity_score = 0.0
                    ref_state[company_id] = current_text

            row["similarity_score"] = similarity_score
            row["has_change"] = has_change
            row["year"] = year
            row["month"] = month
            analysis_rows.append(row)
        except Exception as e:
            log_error(e, __file__)
            logger.error(f"Failed processing analysis row: {e}")

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_cols = [
        "companyid",
        "companyname",
        "companyformername",
        "website",
        "yearfounded",
        "year",
        "month",
        "failure",
        "failure_reason",
        "similarity_score",
        "has_change",
        "num_pages_scraped",
        "text_length",
    ]
    for col in analysis_cols:
        if col not in analysis_df.columns:
            analysis_df[col] = None
        if analysis_df[col].dtype == "object":
            analysis_df[col] = analysis_df[col].fillna("")

    analysis_path = get_analysis_output_path(year, month)
    analysis_df[analysis_cols].to_stata(analysis_path, write_index=False, version=118)
    logger.info(f"Analysis output written to {analysis_path}")

    save_reference_state(ref_state)


def main() -> None:
    now = datetime.datetime.now()
    logger.info(f"Starting analysis job for {now.year}-{now.month:02d}")
    run_analysis(now.year, now.month)


if __name__ == "__main__":
    main()
