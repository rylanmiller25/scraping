import asyncio
import os
import glob
import logging
import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any

from scraper import process_company
from nlp import NLPEngine
from utils import setup_logging

# --- Configuration ---
INPUT_DIR = "../Input Data"
OUTPUT_RAW_DIR = "../Output Data/Raw Text Datasets"
OUTPUT_ANALYSIS_DIR = "../Output Data/Analysis Datasets"
REF_STATE_FILE = "latest_state.parquet"
BATCH_SIZE = 50  # Write checkpoint every 50 companies

# Setup logging
logger = setup_logging()

async def load_reference_state() -> Dict[str, str]:
    """
    Loads the last seen text for every company into memory.
    Returns a dictionary: companyid -> text
    """
    if not os.path.exists(REF_STATE_FILE):
        logger.info("No reference state file found. Starting with empty state.")
        return {}
    
    logger.info(f"Loading reference state from {REF_STATE_FILE}...")
    try:
        df = pd.read_parquet(REF_STATE_FILE)
        # Ensure correct columns
        if 'companyid' not in df.columns or 'text' not in df.columns:
            logger.error("Reference state file missing required columns.")
            return {}
        
        # Convert to dict
        return dict(zip(df['companyid'], df['text']))
    except Exception as e:
        logger.error(f"Error loading reference state: {e}")
        return {}

def save_reference_state(state: Dict[str, str]):
    """
    Saves the updated reference state to Parquet.
    """
    logger.info("Saving updated reference state...")
    try:
        df = pd.DataFrame(list(state.items()), columns=['companyid', 'text'])
        df.to_parquet(REF_STATE_FILE, index=False)
        logger.info("Reference state saved.")
    except Exception as e:
        logger.error(f"Failed to save reference state: {e}")

def get_input_file(year: int, month: int) -> str:
    """
    Finds the input file for the specified month and year.
    Naming convention: pb_{month}_{year}.parquet (month is 2-digit)
    """
    month_str = f"{month:02d}"
    filename = f"pb_{month_str}_{year}.parquet"
    filepath = os.path.join(INPUT_DIR, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    # Check if any input file exists for testing purposes if exact match missing
    # (Optional fallback logic could go here, but strict adherence to plan preferred)
    return ""

async def main():
    # 1. Determine Date context
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month
    
    logger.info(f"Starting scraping job for {current_year}-{current_month:02d}")
    
    # 2. Load Input Data
    input_file = get_input_file(current_year, current_month)
    if not input_file:
        logger.error(f"No input file found for {current_year}-{current_month:02d}. Expected in {INPUT_DIR}")
        return

    logger.info(f"Reading input data from {input_file}")
    input_df = pd.read_parquet(input_file)
    
    # 3. Load Reference State
    ref_state = await load_reference_state()
    
    # 4. Initialize NLP Engine
    nlp_engine = NLPEngine()
    
    # 5. Checkpointing / Resume Logic
    # We'll write to a temporary monthly file and append to it.
    # Check if output files already exist to see what we've done.
    raw_output_path = os.path.join(OUTPUT_RAW_DIR, f"raw_{current_month:02d}_{current_year}.parquet")
    
    processed_ids = set()
    if os.path.exists(raw_output_path):
        logger.info(f"Found existing output file {raw_output_path}. Resuming...")
        existing_df = pd.read_parquet(raw_output_path)
        processed_ids = set(existing_df['companyid'])
        logger.info(f"Already processed {len(processed_ids)} companies.")
        
    # Filter input_df
    companies_to_process = input_df[~input_df['companyid'].isin(processed_ids)].copy()
    companies_to_process.sort_values('companyid', inplace=True)
    
    total_to_process = len(companies_to_process)
    logger.info(f"Remaining companies to process: {total_to_process}")
    
    if total_to_process == 0:
        logger.info("All companies processed.")
        return

    # 6. Setup Concurrency
    semaphore = asyncio.Semaphore(5)
    
    # buffer for batch writing
    results_buffer = []
    
    # Processing Loop
    # We iterate and create tasks. For massive datasets, we might want to chunk this 
    # so we don't create millions of coroutines at once. 
    # Given "startups founded since 2026", the list might grow but manageable.
    # Let's process in chunks or just iterate with semaphore.
    
    tasks = []
    
    # Helper to process one company with semaphore
    async def process_wrapper(row):
        async with semaphore:
            return await process_company(row)

    # Convert dataframe to list of dicts
    company_rows = companies_to_process.to_dict('records')
    
    for i, row in enumerate(company_rows):
        # Run scraping
        # Note: We run strictly sequentially or in parallel?
        # Plan: "Max 5 concurrent companies". 
        # We can create tasks and use asyncio.gather, but we need to handle batch writing.
        # Better: use an async iterator or manage a pool of workers.
        # Simple approach: asyncio.as_completed or similar.
        
        # We'll use a task list and wait for them, but strictly limiting concurrency is key.
        # Semaphore handles the resource limit, but creating 10k tasks at once consumes memory.
        # Let's use a bounded semaphore and simple gather for small batches or 
        # just iterate and append to tasks, but await them in chunks to write to disk.
        
        # Let's do chunks of BATCH_SIZE
        tasks.append(process_wrapper(row))
        
        if len(tasks) >= BATCH_SIZE or i == total_to_process - 1:
            logger.info(f"Processing batch ending at index {i}...")
            batch_results = await asyncio.gather(*tasks)
            
            # Process results (Deduplication + NLP)
            for res_row in batch_results:
                company_id = res_row['companyid']
                current_text = res_row.get('text')
                
                # Logic from plan:
                # If failure, current_text is None or empty.
                # If success, we have text.
                
                similarity_score = 0.0
                has_change = 0
                final_stored_text = None
                
                if res_row['failure'] == 0 and current_text:
                    ref_text = ref_state.get(company_id)
                    
                    if ref_text == current_text:
                        # No change
                        final_stored_text = '-'
                        similarity_score = 1.0
                        has_change = 0
                        # Do NOT update ref_state
                    else:
                        # Changed or New
                        final_stored_text = current_text
                        has_change = 1
                        
                        # Calculate Similarity
                        if ref_text:
                            similarity_score = nlp_engine.compute_similarity(current_text, ref_text)
                        else:
                            # New company, no history to compare?
                            # Plan doesn't explicitly specify score for new, but usually 0 or None.
                            # "Compare... against the Reference State". If no ref state, similarity is undefined or 0.
                            # Let's assume 0.0 implies "completely different" (which it is, compared to nothing).
                            similarity_score = 0.0
                        
                        # Update Reference State
                        ref_state[company_id] = current_text
                else:
                    # Failure case
                    final_stored_text = None # or empty string? Plan says "null value for text" (Line 81)
                    similarity_score = 0.0 # or NaN
                    has_change = 0
                
                # Update row for output
                res_row['text'] = final_stored_text
                res_row['similarity_score'] = similarity_score
                res_row['has_change'] = has_change
                res_row['year'] = current_year
                res_row['month'] = current_month
                
                results_buffer.append(res_row)
            
            # Write Batch to Disk
            if results_buffer:
                _write_batch(results_buffer, raw_output_path, current_year, current_month)
                results_buffer = [] # clear buffer
            
            # Save Reference State Checkpoint (optional but good for safety)
            save_reference_state(ref_state)
            
            tasks = [] # Reset tasks

    # Final Save of Reference State
    save_reference_state(ref_state)
    logger.info("Job complete.")

def _write_batch(rows: List[Dict], raw_path: str, year: int, month: int):
    """
    Appends rows to the Raw Text dataset and updates the Analysis dataset.
    Note: Parquet doesn't strictly support 'append' easily without FastParquet or reading/writing.
    For this implementation, we will check if file exists, read, append, write. 
    Ideally, we'd write separate part files and merge, but for simplicity we append to the main DF.
    """
    
    # Prepare DataFrames
    new_df = pd.DataFrame(rows)
    
    # 1. Raw Text Output
    # Columns: companyid, year, month, text
    raw_cols = ['companyid', 'year', 'month', 'text']
    raw_df_batch = new_df[raw_cols].copy()
    
    if os.path.exists(raw_path):
        existing_raw = pd.read_parquet(raw_path)
        combined_raw = pd.concat([existing_raw, raw_df_batch], ignore_index=True)
        combined_raw.to_parquet(raw_path)
    else:
        raw_df_batch.to_parquet(raw_path)
        
    # 2. Analysis Output
    # Path: Output Data/Analysis Datasets/analysis_{month}_{year}.dta
    analysis_path = os.path.join(OUTPUT_ANALYSIS_DIR, f"analysis_{month:02d}_{year}.dta")
    
    analysis_cols = [
        'companyid', 'companyname', 'companyformername', 'website', 
        'yearfounded', 'year', 'month', 'failure', 'failure_reason', 
        'similarity_score', 'has_change', 'num_pages_scraped', 'text_length'
    ]
    
    # Ensure all cols exist
    for col in analysis_cols:
        if col not in new_df.columns:
            new_df[col] = None
            
    analysis_df_batch = new_df[analysis_cols].copy()
    
    # Append logic for Stata file (read -> concat -> write)
    # Stata files are not appendable. Must rewrite.
    if os.path.exists(analysis_path):
        existing_analysis = pd.read_stata(analysis_path)
        combined_analysis = pd.concat([existing_analysis, analysis_df_batch], ignore_index=True)
        combined_analysis.to_stata(analysis_path, write_index=False, version=118) # 118 is a common modern format
    else:
        analysis_df_batch.to_stata(analysis_path, write_index=False, version=118)

if __name__ == "__main__":
    asyncio.run(main())

