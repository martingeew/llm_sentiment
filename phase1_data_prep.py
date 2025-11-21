"""
Phase 1 - Part 1: Data Preparation

This script handles data loading and sampling:
- Step 1: Validate Configuration
- Step 2: Load ECB-FED speeches dataset from Hugging Face
- Step 3: Sample 2 years of data (2022-2023) and save

For beginners:
- Run this first to download and prepare the data
- Output: sample_2022_2023.csv in data/processed/
- No API calls, no costs
- Usage: python phase1_data_prep.py
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from data_loader import DataLoader
from utils import print_section_header


def main():
    """
    Main function that loads and samples the data.
    """

    print("=" * 70)
    print("PHASE 1 - PART 1: DATA PREPARATION".center(70))
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
    # STEP 2: Load Dataset
    # ============================================================
    print_section_header("STEP 2: LOAD DATASET FROM HUGGING FACE")

    loader = DataLoader()

    # Load the complete dataset
    df = loader.load_from_huggingface()

    # Show summary
    loader.print_dataset_summary(df)

    # ============================================================
    # STEP 3: Sample 2 Years of Data
    # ============================================================
    print_section_header("STEP 3: SAMPLE TEST DATA")

    # Sample 2022-2023 for testing
    sample_df = loader.sample_data(
        df,
        start_year=Config.SAMPLE_START_YEAR,
        end_year=Config.SAMPLE_END_YEAR
    )

    print(f"\n📋 Sample Data Summary:")
    print(f"   Total speeches: {len(sample_df)}")

    if 'institution' in sample_df.columns:
        print(f"\n   By institution:")
        for institution, count in sample_df['institution'].value_counts().items():
            print(f"      {institution}: {count}")

    # Save sample for reference
    sample_file = Config.PROCESSED_DATA_DIR / "sample_2022_2023.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"\n✓ Saved sample to: {sample_file}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print_section_header("DATA PREPARATION COMPLETE ✅")

    print("\n📁 File created:")
    print(f"   Sample data: {sample_file}")

    print("\n🎯 What we've done:")
    print("   ✓ Loaded ECB-FED speeches dataset from Hugging Face")
    print(f"   ✓ Sampled {len(sample_df)} speeches from {Config.SAMPLE_START_YEAR}-{Config.SAMPLE_END_YEAR}")
    print("   ✓ Saved sample to CSV file")

    print("\n📝 Next Step:")
    print("   Run: python phase1_batch_prep.py")
    print("   This will create the batch file and show cost estimates")

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
