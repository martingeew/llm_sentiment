"""
Phase 1 - Part 2: Batch File Preparation

This script creates batch processing files and shows cost estimates:
- Step 1: Validate Configuration
- Step 4: Build Batch Request File
- Step 5: Cost Comparison Summary

For beginners:
- Run this AFTER phase1_data_prep.py
- Requires: sample_2022_2023.csv (created by phase1_data_prep.py)
- Output: batch_sample_2022_2023.jsonl + cost statistics
- No API calls, no costs
- Usage: python phase1_batch_prep.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add src directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from batch_builder import BatchRequestBuilder
from utils import print_section_header, save_json


def main():
    """
    Main function that creates batch files and shows cost comparison.
    """

    print("=" * 70)
    print("PHASE 1 - PART 2: BATCH FILE PREPARATION".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Validate Configuration
    # ============================================================
    print_section_header("STEP 1: VALIDATE CONFIGURATION")

    try:
        Config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Add your Hugging Face token to .env")
        print("4. Run this script again")
        return

    # ============================================================
    # LOAD SAMPLE DATA
    # ============================================================
    print_section_header("LOADING SAMPLE DATA")

    sample_file = Config.PROCESSED_DATA_DIR / "sample_2022_2023.csv"

    if not sample_file.exists():
        print(f"\n❌ Error: Sample file not found at {sample_file}")
        print("\nPlease run phase1_data_prep.py first to create the sample data:")
        print("   python phase1_data_prep.py")
        return

    print(f"📂 Loading sample data from: {sample_file}")
    sample_df = pd.read_csv(sample_file)
    print(f"✓ Loaded {len(sample_df)} speeches")

    # ============================================================
    # STEP 4: Build Batch Request File
    # ============================================================
    print_section_header("STEP 4: BUILD BATCH REQUEST FILE")

    # Determine column names (they might vary in the dataset)
    # Common possibilities for column names
    text_col = None
    id_col = None
    speaker_col = None
    institution_col = None
    date_col = None

    # Find text column
    for possible_name in ['text', 'content', 'speech', 'body']:
        if possible_name in sample_df.columns:
            text_col = possible_name
            break

    # Find ID column (or create one)
    for possible_name in ['id', 'speech_id', 'index']:
        if possible_name in sample_df.columns:
            id_col = possible_name
            break

    if not id_col:
        # Create ID column
        sample_df['id'] = range(len(sample_df))
        id_col = 'id'

    # Find other columns
    for col in sample_df.columns:
        col_lower = col.lower()
        if 'speaker' in col_lower and not speaker_col:
            speaker_col = col
        elif 'institution' in col_lower and not institution_col:
            institution_col = col
        elif 'date' in col_lower and not date_col:
            date_col = col

    # Validate we have required columns
    if not text_col:
        print("\n❌ Error: Could not find text column in dataset")
        print(f"   Available columns: {list(sample_df.columns)}")
        return

    # Fill in missing columns with defaults
    if not speaker_col:
        sample_df['speaker'] = 'Unknown'
        speaker_col = 'speaker'

    if not institution_col:
        sample_df['institution'] = 'Unknown'
        institution_col = 'institution'

    if not date_col:
        sample_df['date'] = 'Unknown'
        date_col = 'date'

    print(f"\n📋 Using columns:")
    print(f"   Text: {text_col}")
    print(f"   ID: {id_col}")
    print(f"   Speaker: {speaker_col}")
    print(f"   Institution: {institution_col}")
    print(f"   Date: {date_col}")

    # Build batch file
    builder = BatchRequestBuilder()
    batch_file = Config.BATCH_INPUT_DIR / f"batch_sample_{Config.SAMPLE_START_YEAR}_{Config.SAMPLE_END_YEAR}.jsonl"

    stats = builder.build_batch_file(
        df=sample_df,
        output_file=batch_file,
        text_column=text_col,
        id_column=id_col,
        speaker_column=speaker_col,
        institution_column=institution_col,
        date_column=date_col
    )

    # Validate the batch file
    print_section_header("VALIDATING BATCH FILE")
    is_valid = builder.validate_batch_file(batch_file)

    # ============================================================
    # STEP 5: Cost Comparison Summary
    # ============================================================
    print_section_header("STEP 5: COST COMPARISON")

    print("\n💰 DETAILED COST BREAKDOWN")
    print("=" * 60)

    print(f"\n📊 Processing {stats['num_requests']} speeches")
    print(f"   Input tokens:  {stats['total_input_tokens']:,}")
    print(f"   Output tokens: {stats['estimated_output_tokens']:,} (estimated)")

    print(f"\n💵 BATCH API (50% discount):")
    print(f"   Input cost:  ${stats['total_input_tokens']/1000 * Config.BATCH_INPUT_PRICE_PER_1K:.2f}")
    print(f"   Output cost: ${stats['estimated_output_tokens']/1000 * Config.BATCH_OUTPUT_PRICE_PER_1K:.2f}")
    print(f"   TOTAL:       ${stats['batch_cost_estimate']:.2f}")

    print(f"\n💵 REAL-TIME API (standard pricing):")
    print(f"   Input cost:  ${stats['total_input_tokens']/1000 * Config.REALTIME_INPUT_PRICE_PER_1K:.2f}")
    print(f"   Output cost: ${stats['estimated_output_tokens']/1000 * Config.REALTIME_OUTPUT_PRICE_PER_1K:.2f}")
    print(f"   TOTAL:       ${stats['realtime_cost_estimate']:.2f}")

    print(f"\n✅ SAVINGS WITH BATCH API:")
    print(f"   ${stats['savings']:.2f} ({stats['savings']/stats['realtime_cost_estimate']*100:.0f}% discount)")

    # Save statistics
    stats_file = Config.RESULTS_DIR / "phase1_statistics.json"
    save_json(stats, stats_file)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print_section_header("BATCH PREPARATION COMPLETE ✅")

    print("\n📁 Files created:")
    print(f"   1. Batch file:   {batch_file}")
    print(f"   2. Statistics:   {stats_file}")

    print("\n🎯 What we've demonstrated:")
    print(f"   ✓ Created batch processing file with {stats['num_requests']} requests")
    print(f"   ✓ Compared costs: ${stats['batch_cost_estimate']:.2f} (batch) vs ${stats['realtime_cost_estimate']:.2f} (real-time)")
    print(f"   ✓ Validated batch file format")

    print("\n📊 Cost savings: ${:.2f} ({:.0f}% discount)".format(
        stats['savings'],
        stats['savings']/stats['realtime_cost_estimate']*100
    ))

    print("\n📝 Next Steps:")
    print("   1. Review the batch file to ensure it looks correct")
    print("   2. Check the cost estimates are within your budget")
    print("   3. If ready, proceed to actually submit the batch job")
    print("      (This will be in Phase 2 - actual batch submission)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
