"""
Phase 3: Data Preparation for Index Building

This script loads sentiment results and speech metadata, merges them,
and prepares a clean dataset for building time series indices.

For beginners:
- Combines LLM sentiment scores with original speech information
- Cleans and validates the data
- Creates a ready-to-analyze dataset

Usage: python phase3_data_prep.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from utils import print_section_header


def main():
    """
    Prepare and merge sentiment data with speech metadata.
    """

    print("=" * 70)
    print("PHASE 3: DATA PREPARATION".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Sentiment Results
    # ============================================================
    print_section_header("STEP 1: LOAD SENTIMENT RESULTS")

    sentiment_file = Config.RESULTS_DIR / "sentiment_results_2022_2023.csv"

    if not sentiment_file.exists():
        print(f"\nERROR Error: Sentiment results not found!")
        print(f"   Expected: {sentiment_file}")
        print(f"\n   Run: python phase2_download_results.py")
        return

    sentiment_df = pd.read_csv(sentiment_file)

    print(f"\nOK Loaded sentiment results")
    print(f"   File: {sentiment_file.name}")
    print(f"   Speeches: {len(sentiment_df)}")
    print(f"   Columns: {len(sentiment_df.columns)}")

    # ============================================================
    # STEP 2: Load Speech Metadata
    # ============================================================
    print_section_header("STEP 2: LOAD SPEECH METADATA")

    # Try parquet first, then CSV
    metadata_file = Config.PROCESSED_DATA_DIR / "sample_2022_2023.parquet"
    if not metadata_file.exists():
        metadata_file = Config.PROCESSED_DATA_DIR / "sample_2022_2023.csv"

    if not metadata_file.exists():
        print(f"\nERROR Error: Speech metadata not found!")
        print(f"   Expected: {metadata_file}")
        print(f"\n   Run: python phase1_data_prep.py")
        return

    if metadata_file.suffix == ".parquet":
        metadata_df = pd.read_parquet(metadata_file)
    else:
        metadata_df = pd.read_csv(metadata_file)

    print(f"\nOK Loaded speech metadata")
    print(f"   File: {metadata_file.name}")
    print(f"   Speeches: {len(metadata_df)}")
    print(f"   Columns: {metadata_df.columns.tolist()}")

    # ============================================================
    # STEP 3: Merge Sentiment with Metadata
    # ============================================================
    print_section_header("STEP 3: MERGE DATASETS")

    # The sentiment results have speech_id like "speech_0", "speech_1", etc.
    # This indicates which row from the original metadata they correspond to
    # We need to extract the row number and use it to properly align the data

    print(f"\n   Sentiment results: {len(sentiment_df)} rows")
    print(f"   Metadata: {len(metadata_df)} rows")

    # Extract row number from speech_id (e.g., "speech_88" -> 88)
    print(f"\n   Extracting row indices from speech_id...")
    sentiment_df['row_index'] = sentiment_df['speech_id'].str.extract(r'speech_(\d+)').astype(int)

    # Add row index to metadata
    metadata_df = metadata_df.reset_index(drop=True)
    metadata_df['row_index'] = metadata_df.index

    # Merge on row_index to ensure correct alignment
    print(f"   Merging on row_index to ensure correct alignment...")
    merged_df = pd.merge(
        metadata_df[["row_index", "date", "author", "country", "title"]],
        sentiment_df,
        on='row_index',
        how='inner'
    )

    # Drop the temporary row_index column
    merged_df = merged_df.drop(columns=['row_index'])

    print(f"\nOK Merged datasets")
    print(f"   Final rows: {len(merged_df)}")
    print(f"   Final columns: {len(merged_df.columns)}")

    # ============================================================
    # STEP 4: Parse and Clean Dates
    # ============================================================
    print_section_header("STEP 4: PARSE DATES")

    # Convert date column to datetime
    merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")

    # Check for invalid dates
    invalid_dates = merged_df["date"].isna().sum()
    if invalid_dates > 0:
        print(f"\nWARNING  Warning: {invalid_dates} speeches have invalid dates")
        print(f"   These will be dropped")
        merged_df = merged_df.dropna(subset=["date"])

    # Sort chronologically
    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    print(f"\nOK Dates parsed and sorted")
    print(
        f"   Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}"
    )
    print(f"   Total days: {(merged_df['date'].max() - merged_df['date'].min()).days}")

    # ============================================================
    # STEP 5: Handle Missing Values
    # ============================================================
    print_section_header("STEP 5: CHECK MISSING VALUES")

    # Check for missing values in key columns
    key_columns = [
        "date",
        "author",
        "country",
        "hawkish_dovish_score",
        "uncertainty",
        "forward_guidance_strength",
        "topic_inflation",
        "topic_growth",
    ]

    print(f"\n   Missing values in key columns:")
    missing_summary = []
    for col in key_columns:
        if col in merged_df.columns:
            missing = merged_df[col].isna().sum()
            pct = (missing / len(merged_df)) * 100
            missing_summary.append({"column": col, "missing": missing, "percent": pct})
            if missing > 0:
                print(f"   {col}: {missing} ({pct:.1f}%)")

    total_missing = sum(s["missing"] for s in missing_summary)
    if total_missing == 0:
        print(f"\nOK No missing values in key columns!")
    else:
        print(f"\n   Total missing: {total_missing}")

    # ============================================================
    # STEP 6: Data Quality Summary
    # ============================================================
    print_section_header("STEP 6: DATA QUALITY SUMMARY")

    # Institution breakdown
    print(f"\n   Speeches by institution:")
    for institution, count in merged_df["country"].value_counts().items():
        print(f"      {institution}: {count}")

    # Time coverage
    speeches_by_month = merged_df.groupby(merged_df["date"].dt.to_period("M")).size()
    print(f"\n   Speeches by month:")
    for period, count in speeches_by_month.items():
        print(f"      {period}: {count}")

    # Score distributions
    print(f"\n   Score statistics:")
    print(
        f"      Hawkish/Dovish: mean={merged_df['hawkish_dovish_score'].mean():.1f}, "
        f"std={merged_df['hawkish_dovish_score'].std():.1f}, "
        f"range=[{merged_df['hawkish_dovish_score'].min():.0f}, {merged_df['hawkish_dovish_score'].max():.0f}]"
    )
    print(
        f"      Uncertainty: mean={merged_df['uncertainty'].mean():.1f}, "
        f"std={merged_df['uncertainty'].std():.1f}"
    )
    print(
        f"      Forward Guidance: mean={merged_df['forward_guidance_strength'].mean():.1f}, "
        f"std={merged_df['forward_guidance_strength'].std():.1f}"
    )

    # ============================================================
    # STEP 7: Save Prepared Dataset
    # ============================================================
    print_section_header("STEP 7: SAVE PREPARED DATASET")

    output_file = Config.RESULTS_DIR / "phase3_prepared_data.csv"
    merged_df.to_csv(output_file, index=False)

    print(f"\nOK Saved prepared dataset")
    print(f"   File: {output_file}")
    print(f"   Rows: {len(merged_df)}")
    print(f"   Columns: {len(merged_df.columns)}")
    print(
        f"\n   NOTE: When loading this CSV, use parse_dates=['date'] to preserve datetime type:"
    )
    print(f"   df = pd.read_csv('{output_file.name}', parse_dates=['date'])")

    # ============================================================
    # STEP 8: Show Sample Data
    # ============================================================
    print_section_header("SAMPLE DATA (First 5 Speeches)")

    # Show key columns
    display_cols = [
        "date",
        "country",
        "author",
        "hawkish_dovish_score",
        "uncertainty",
        "topic_inflation",
        "topic_growth",
    ]

    print(f"\n{merged_df[display_cols].head(5).to_string()}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE".center(70))
    print("=" * 70)

    print(f"\nOutput file: {output_file}")
    print(f"\nDataset summary:")
    print(f"   Total speeches: {len(merged_df)}")
    print(
        f"   Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}"
    )
    print(f"   Institutions: {', '.join(merged_df['country'].unique())}")
    print(f"   Average hawkish/dovish: {merged_df['hawkish_dovish_score'].mean():.1f}")

    print(f"\nNext steps:")
    print(f"   1. Explore the data in Jupyter:")
    print(
        f"      df = pd.read_csv('data/results/phase3_prepared_data.csv', parse_dates=['date'])"
    )
    print(f"   2. Decide on aggregation strategy (daily/weekly/monthly)")
    print(f"   3. Choose which indices to build")
    print(f"   4. Run index building script (to be created)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING  Interrupted by user")
    except Exception as e:
        print(f"\n\nERROR Error: {e}")
        import traceback

        traceback.print_exc()
