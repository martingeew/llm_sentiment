"""
Phase 2: Batch Job Submission and Processing

This script submits the batch file to OpenAI, monitors progress,
downloads results, and validates LLM outputs.

For beginners:
- Run this AFTER Phase 1 (batch file must exist)
- This will make REAL API calls and incur costs
- Estimated cost: ~$2.32 for 2022-2023 sample
- Process takes up to 24 hours (usually 30min - 4 hours)
- Usage: python phase2_batch_submit.py

IMPORTANT: This script makes actual API calls and charges will apply!
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
from utils import print_section_header, save_json


def main():
    """
    Main function for Phase 2: Submit batch and process results.
    """

    print("=" * 70)
    print("PHASE 2: BATCH JOB SUBMISSION & PROCESSING".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # SAFETY CHECK
    # ============================================================
    print("\n⚠️  WARNING: This script will make REAL API calls")
    print("   Estimated cost: ~$2.32 for the 2022-2023 sample")
    print("   Processing time: Up to 24 hours (typically 30min-4hrs)")

    response = input("\n   Type 'YES' to proceed: ")
    if response.upper() != 'YES':
        print("\n❌ Aborted by user. No charges incurred.")
        return

    # ============================================================
    # STEP 1: Validate Configuration
    # ============================================================
    print_section_header("STEP 1: VALIDATE CONFIGURATION")

    try:
        Config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        return

    # ============================================================
    # STEP 2: Check Batch File Exists
    # ============================================================
    print_section_header("STEP 2: CHECK BATCH FILE")

    batch_file = Config.BATCH_INPUT_DIR / "batch_sample_2022_2023.jsonl"

    if not batch_file.exists():
        print(f"\n❌ Error: Batch file not found!")
        print(f"   Expected: {batch_file}")
        print(f"\n   Please run phase1_batch_prep.py first to create the batch file.")
        return

    print(f"✓ Batch file found: {batch_file}")

    # Show file size
    file_size_mb = batch_file.stat().st_size / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.2f} MB")

    # ============================================================
    # STEP 3: Upload and Submit Batch
    # ============================================================
    print_section_header("STEP 3: UPLOAD & SUBMIT BATCH JOB")

    processor = BatchProcessor()

    # Upload file
    file_id = processor.upload_batch_file(batch_file)

    # Create batch job
    batch_id = processor.create_batch_job(
        file_id,
        description="Phase 2: ECB-FED Speeches 2022-2023 Sentiment Analysis"
    )

    # Save batch ID for later reference
    batch_info = {
        'batch_id': batch_id,
        'file_id': file_id,
        'submitted_at': datetime.now().isoformat(),
        'batch_file': str(batch_file),
        'status': 'submitted'
    }

    batch_info_file = Config.RESULTS_DIR / "phase2_batch_info.json"
    save_json(batch_info, batch_info_file)
    print(f"\n💾 Batch info saved to: {batch_info_file}")
    print(f"   Batch ID: {batch_id}")
    print(f"   (You can check status later using this ID)")

    # ============================================================
    # STEP 4: Monitor Batch Progress
    # ============================================================
    print_section_header("STEP 4: MONITOR BATCH PROGRESS")

    print("\n📝 Note: You can safely stop this script and check back later.")
    print("   The batch job will continue processing on OpenAI's servers.")
    print("   To resume monitoring, run: python phase2_check_status.py")

    response = input("\n   Monitor now? (Y/n): ")
    if response.lower() == 'n':
        print("\n✅ Batch submitted successfully!")
        print(f"   Batch ID: {batch_id}")
        print("\n   Run phase2_check_status.py later to check progress and download results.")
        return

    # Wait for completion
    status = processor.wait_for_completion(batch_id)

    # Update batch info with completion status
    batch_info['status'] = status['status']
    batch_info['completed_at'] = datetime.now().isoformat()
    save_json(batch_info, batch_info_file)

    # ============================================================
    # STEP 5: Download Results
    # ============================================================
    if status['status'] == 'completed':
        print_section_header("STEP 5: DOWNLOAD RESULTS")

        results_file = Config.BATCH_OUTPUT_DIR / "batch_results_2022_2023.jsonl"
        processor.download_results(batch_id, results_file)

        # ============================================================
        # STEP 6: Parse Results
        # ============================================================
        print_section_header("STEP 6: PARSE RESULTS")

        parsed_csv = Config.RESULTS_DIR / "sentiment_results_2022_2023.csv"
        results_df = processor.parse_results(results_file, parsed_csv)

        # ============================================================
        # STEP 7: Validate Outputs
        # ============================================================
        print_section_header("STEP 7: VALIDATE LLM OUTPUTS")

        validator = OutputValidator()

        # Run validation
        validation_results = validator.validate_batch_results(results_df)
        validator.print_validation_report(validation_results)

        # Check distributions
        distributions = validator.check_score_distributions(results_df)
        validator.print_distribution_report(distributions)

        # Save validation report
        validation_file = Config.RESULTS_DIR / "phase2_validation_report.json"
        validation_report = {
            'validation_results': validation_results,
            'distributions': distributions,
            'validated_at': datetime.now().isoformat()
        }
        save_json(validation_report, validation_file)

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print_section_header("PHASE 2 COMPLETE ✅")

        print("\n📁 Files created:")
        print(f"   1. Raw results:      {results_file}")
        print(f"   2. Parsed results:   {parsed_csv}")
        print(f"   3. Validation report: {validation_file}")
        print(f"   4. Batch info:       {batch_info_file}")

        print(f"\n📊 Processing Summary:")
        print(f"   Total speeches processed: {len(results_df)}")
        print(f"   Valid outputs: {validation_results['valid_speeches']}")
        print(f"   Validation rate: {validation_results['validation_rate']:.1f}%")

        if validation_results['validation_rate'] >= 95:
            print(f"\n✅ Excellent! {validation_results['validation_rate']:.1f}% validation rate")
        elif validation_results['validation_rate'] >= 85:
            print(f"\n⚠️  Good, but some issues: {validation_results['validation_rate']:.1f}% validation rate")
        else:
            print(f"\n❌ Low validation rate: {validation_results['validation_rate']:.1f}%")
            print("   Review validation report for details")

        print(f"\n📝 Next Steps:")
        print("   1. Review the validation report")
        print("   2. Inspect sample results in the CSV file")
        print("   3. If quality is good, proceed to Phase 3 (build indices)")

    else:
        print(f"\n⚠️  Batch job did not complete successfully")
        print(f"   Status: {status['status']}")
        print(f"   Please check the batch status and try again")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("   Batch job will continue processing on OpenAI's servers")
        print("   Run phase2_check_status.py to resume monitoring")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
