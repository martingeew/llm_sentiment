"""
Phase 2: Download and Merge Chunk Results

This script downloads results from all completed chunks, merges them
into a single dataset, and runs validation.

For beginners:
- Use this after all chunks are completed
- No additional charges (just downloading results)
- Merges all chunk CSVs into one master file

Usage:
    python phase2_download_results.py
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from batch_processor import BatchProcessor
from output_validator import OutputValidator
from utils import print_section_header, save_json, load_json


def main():
    """
    Download and merge all chunk results.
    """

    print("=" * 70)
    print("PHASE 2: DOWNLOAD & MERGE CHUNK RESULTS".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Find All Chunk Info Files
    # ============================================================
    print_section_header("STEP 1: FIND CHUNK INFO FILES")

    chunk_files = sorted(Config.RESULTS_DIR.glob("phase2_chunk*_info.json"))

    if not chunk_files:
        print("\n   Error: No chunk info files found")
        print(f"   Have you run: python phase2_batch_submit.py --all-chunks")
        return

    print(f"\n   Found {len(chunk_files)} chunk info files")

    # ============================================================
    # STEP 2: Check Which Chunks Are Completed
    # ============================================================
    print_section_header("STEP 2: CHECK CHUNK STATUS")

    processor = BatchProcessor()
    completed_chunks = []
    in_progress_chunks = []
    failed_chunks = []

    for chunk_file in chunk_files:
        chunk_info = load_json(chunk_file)
        chunk_num = chunk_info['chunk_number']
        batch_id = chunk_info['batch_id']

        try:
            status = processor.check_batch_status(batch_id)

            if status['status'] == 'completed':
                completed_chunks.append((chunk_num, chunk_info, status))
                print(f"   Chunk {chunk_num:2d}: completed")
            elif status['status'] in ['in_progress', 'validating']:
                in_progress_chunks.append(chunk_num)
                print(f"   Chunk {chunk_num:2d}: {status['status']}")
            else:
                failed_chunks.append(chunk_num)
                print(f"   Chunk {chunk_num:2d}: {status['status']}")

        except Exception as e:
            print(f"   Chunk {chunk_num:2d}: error - {e}")
            failed_chunks.append(chunk_num)

    # Summary
    print(f"\n   Summary:")
    print(f"      Completed: {len(completed_chunks)}")
    print(f"      In progress: {len(in_progress_chunks)}")
    print(f"      Failed: {len(failed_chunks)}")

    if not completed_chunks:
        print(f"\n   No completed chunks to download. Exiting.")
        return

    if in_progress_chunks:
        print(f"\n   Note: {len(in_progress_chunks)} chunks still processing")
        print(f"   Will download and merge the {len(completed_chunks)} completed chunks")

    # ============================================================
    # STEP 3: Download Results for Completed Chunks
    # ============================================================
    print_section_header("STEP 3: DOWNLOAD RESULTS")

    downloaded_files = []

    for chunk_num, chunk_info, status in completed_chunks:
        batch_id = chunk_info['batch_id']

        # Define output file
        results_file = Config.BATCH_OUTPUT_DIR / f"chunk{chunk_num:02d}_results.jsonl"

        # Check if already downloaded
        if results_file.exists():
            print(f"\n   Chunk {chunk_num:2d}: Already downloaded ({results_file.name})")
            downloaded_files.append((chunk_num, results_file))
        else:
            try:
                print(f"\n   Chunk {chunk_num:2d}: Downloading...")
                processor.download_results(batch_id, results_file)
                downloaded_files.append((chunk_num, results_file))
                print(f"               Saved to {results_file.name}")
            except Exception as e:
                print(f"               Error: {e}")

    # ============================================================
    # STEP 4: Parse Each Chunk to CSV
    # ============================================================
    print_section_header("STEP 4: PARSE CHUNK RESULTS")

    parsed_dfs = []

    for chunk_num, results_file in downloaded_files:
        parsed_csv = Config.RESULTS_DIR / f"chunk{chunk_num:02d}_parsed.csv"

        # Check if already parsed
        if parsed_csv.exists():
            print(f"\n   Chunk {chunk_num:2d}: Loading existing CSV ({parsed_csv.name})")
            df = pd.read_csv(parsed_csv)
            parsed_dfs.append(df)
        else:
            try:
                print(f"\n   Chunk {chunk_num:2d}: Parsing...")
                df = processor.parse_results(results_file, parsed_csv)
                parsed_dfs.append(df)
                print(f"               Parsed {len(df)} speeches")
            except Exception as e:
                print(f"               Error: {e}")

    if not parsed_dfs:
        print(f"\n   Error: No chunk results could be parsed")
        return

    # ============================================================
    # STEP 5: Merge All Chunks
    # ============================================================
    print_section_header("STEP 5: MERGE ALL CHUNKS")

    # Combine all DataFrames
    merged_df = pd.concat(parsed_dfs, ignore_index=True)

    # Sort by speech_id
    merged_df = merged_df.sort_values('speech_id').reset_index(drop=True)

    print(f"\n   Total speeches merged: {len(merged_df)}")
    print(f"   From {len(parsed_dfs)} chunks")

    # Save merged results
    merged_csv = Config.RESULTS_DIR / "sentiment_results_2022_2023.csv"
    merged_df.to_csv(merged_csv, index=False)

    print(f"\n   Saved merged results to: {merged_csv.name}")

    # ============================================================
    # STEP 6: Validate Merged Results
    # ============================================================
    print_section_header("STEP 6: VALIDATE MERGED RESULTS")

    validator = OutputValidator()

    # Run validation
    validation_results = validator.validate_batch_results(merged_df)
    validator.print_validation_report(validation_results)

    # Check distributions
    distributions = validator.check_score_distributions(merged_df)
    validator.print_distribution_report(distributions)

    # Save validation report
    validation_file = Config.RESULTS_DIR / "phase2_validation_report.json"
    validation_report = {
        'validation_results': validation_results,
        'distributions': distributions,
        'validated_at': datetime.now().isoformat(),
        'chunks_merged': len(parsed_dfs),
        'total_chunks': len(chunk_files),
        'in_progress_chunks': len(in_progress_chunks),
        'failed_chunks': len(failed_chunks)
    }
    save_json(validation_report, validation_file)

    print(f"\n   Validation report saved to: {validation_file.name}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print_section_header("RESULTS READY")

    print("\n   Files created:")
    print(f"      1. Merged results:     {merged_csv}")
    print(f"      2. Validation report:  {validation_file}")
    print(f"      3. Individual chunks:  {len(downloaded_files)} files in batch_output/")

    print(f"\n   Summary:")
    print(f"      Total speeches: {len(merged_df)}")
    print(f"      Valid outputs: {validation_results['valid_speeches']}")
    print(f"      Validation rate: {validation_results['validation_rate']:.1f}%")

    if validation_results['validation_rate'] >= 95:
        print(f"\n   Excellent! {validation_results['validation_rate']:.1f}% validation rate")
    elif validation_results['validation_rate'] >= 85:
        print(f"\n   Good, but some issues: {validation_results['validation_rate']:.1f}% validation rate")
    else:
        print(f"\n   Low validation rate: {validation_results['validation_rate']:.1f}%")
        print(f"   Review validation report for details")

    if in_progress_chunks:
        print(f"\n   Note: {len(in_progress_chunks)} chunks still processing")
        print(f"   You can run this script again later to include them")

    print(f"\n   Next Steps:")
    print(f"      1. Review the validation report")
    print(f"      2. Inspect sample results in the CSV file")
    print(f"      3. If quality is good, proceed to Phase 3 (build indices)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n   Interrupted by user")
    except Exception as e:
        print(f"\n\n   Error: {e}")
        import traceback
        traceback.print_exc()
