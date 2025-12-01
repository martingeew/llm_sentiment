"""
Phase 2: Check Batch Status and Download Results

This script checks the status of a previously submitted batch job
and downloads results if complete.

For beginners:
- Use this if you stopped monitoring or want to check back later
- Reads the batch ID from phase2_batch_info.json or chunk info files
- No additional charges (just checking status)

Usage:
    python phase2_check_status.py              # Auto-detect chunks or single batch
    python phase2_check_status.py --all-chunks # Check all chunks
    python phase2_check_status.py --chunk 1    # Check specific chunk
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from batch_processor import BatchProcessor
from output_validator import OutputValidator
from utils import print_section_header, save_json, load_json


def detect_chunk_files():
    """Detect if chunk info files exist."""
    chunk_files = sorted(Config.RESULTS_DIR.glob("phase2_chunk*_info.json"))
    return chunk_files if chunk_files else None


def check_all_chunks_status():
    """Check status of all chunks and display summary."""
    chunk_files = detect_chunk_files()

    if not chunk_files:
        print("\n   Error: No chunk info files found")
        print(f"   Have you run: python phase2_batch_submit.py --all-chunks")
        return

    print("=" * 70)
    print("CHUNK STATUS SUMMARY".center(70))
    print("=" * 70)

    processor = BatchProcessor()

    # Collect status for all chunks
    chunk_statuses = []
    for chunk_file in chunk_files:
        chunk_info = load_json(chunk_file)
        chunk_num = chunk_info['chunk_number']
        batch_id = chunk_info['batch_id']

        try:
            status = processor.check_batch_status(batch_id)
            chunk_statuses.append({
                'chunk': chunk_num,
                'batch_id': batch_id,
                'status': status['status'],
                'total': status['request_counts'].get('total', 0) if status['request_counts'] else 0,
                'completed': status['request_counts'].get('completed', 0) if status['request_counts'] else 0,
                'failed': status['request_counts'].get('failed', 0) if status['request_counts'] else 0
            })
        except Exception as e:
            print(f"\n   Error checking chunk {chunk_num}: {e}")
            chunk_statuses.append({
                'chunk': chunk_num,
                'batch_id': batch_id,
                'status': 'error',
                'total': 0,
                'completed': 0,
                'failed': 0
            })

    # Display table
    print(f"\n{'Chunk':<8} {'Status':<15} {'Progress':<15} {'Batch ID':<40}")
    print("-" * 70)

    status_counts = {'completed': 0, 'in_progress': 0, 'failed': 0, 'other': 0}

    for cs in chunk_statuses:
        progress = f"{cs['completed']}/{cs['total']}" if cs['total'] > 0 else "N/A"
        print(f"{cs['chunk']:<8} {cs['status']:<15} {progress:<15} {cs['batch_id']:<40}")

        # Count statuses
        if cs['status'] == 'completed':
            status_counts['completed'] += 1
        elif cs['status'] == 'in_progress' or cs['status'] == 'validating':
            status_counts['in_progress'] += 1
        elif cs['status'] == 'failed':
            status_counts['failed'] += 1
        else:
            status_counts['other'] += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY".center(70))
    print("=" * 70)

    total_chunks = len(chunk_statuses)
    print(f"\n   Total chunks: {total_chunks}")
    print(f"   Completed: {status_counts['completed']}")
    print(f"   In progress: {status_counts['in_progress']}")
    print(f"   Failed: {status_counts['failed']}")

    if status_counts['completed'] == total_chunks:
        print(f"\n   ALL CHUNKS COMPLETED!")
        print(f"\n   Next step:")
        print(f"   python phase2_download_results.py")
    elif status_counts['in_progress'] > 0:
        print(f"\n   Some chunks still processing. Check back later.")
    elif status_counts['failed'] > 0:
        print(f"\n   Some chunks failed. Check error details.")

    print("\n" + "=" * 70)


def check_single_chunk_status(chunk_num):
    """Check status of a specific chunk."""
    chunk_file = Config.RESULTS_DIR / f"phase2_chunk{chunk_num:02d}_info.json"

    if not chunk_file.exists():
        print(f"\n   Error: Chunk {chunk_num} info file not found")
        print(f"   Expected: {chunk_file}")
        return

    print("=" * 70)
    print(f"CHUNK {chunk_num} STATUS".center(70))
    print("=" * 70)

    chunk_info = load_json(chunk_file)
    batch_id = chunk_info['batch_id']

    print(f"\n   Batch ID: {batch_id}")
    print(f"   Submitted: {chunk_info['submitted_at']}")
    print(f"   Speeches: {chunk_info['num_requests']}")

    processor = BatchProcessor()
    status = processor.check_batch_status(batch_id)

    print(f"\n   Status: {status['status']}")

    if status['request_counts']:
        counts = status['request_counts']
        total = counts.get('total', 0)
        completed = counts.get('completed', 0)
        failed = counts.get('failed', 0)

        if total > 0:
            progress = (completed / total) * 100
            print(f"   Progress: {completed}/{total} ({progress:.1f}%)")
            if failed > 0:
                print(f"   Failed: {failed}")

    print("\n" + "=" * 70)


def main():
    """
    Check batch status and download results if ready.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Check batch job status"
    )
    parser.add_argument(
        '--all-chunks',
        action='store_true',
        help='Check status of all chunks'
    )
    parser.add_argument(
        '--chunk',
        type=int,
        help='Check status of specific chunk'
    )

    args = parser.parse_args()

    # Detect if chunks exist
    chunk_files = detect_chunk_files()

    # Determine mode
    if args.all_chunks:
        check_all_chunks_status()
        return

    elif args.chunk:
        check_single_chunk_status(args.chunk)
        return

    # Auto-detect mode
    if chunk_files:
        # Chunks exist, use chunk mode
        print("=" * 70)
        print("CHUNK FILES DETECTED".center(70))
        print("=" * 70)

        print(f"\n   Found {len(chunk_files)} chunk info files")
        print(f"\n   Checking all chunks...")
        print()
        check_all_chunks_status()
        return

    # No chunks, use original workflow
    print("=" * 70)
    print("PHASE 2: CHECK BATCH STATUS".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Batch Info
    # ============================================================
    print_section_header("STEP 1: LOAD BATCH INFORMATION")

    batch_info_file = Config.RESULTS_DIR / "phase2_batch_info.json"

    if not batch_info_file.exists():
        print(f"\n❌ Error: Batch info file not found!")
        print(f"   Expected: {batch_info_file}")
        print(f"\n   Have you run phase2_batch_submit.py yet?")
        return

    batch_info = load_json(batch_info_file)
    batch_id = batch_info['batch_id']

    print(f"✓ Batch ID: {batch_id}")
    print(f"✓ Submitted at: {batch_info['submitted_at']}")
    print(f"✓ Previous status: {batch_info.get('status', 'unknown')}")

    # ============================================================
    # STEP 2: Check Current Status
    # ============================================================
    print_section_header("STEP 2: CHECK CURRENT STATUS")

    processor = BatchProcessor()
    status = processor.check_batch_status(batch_id)

    print(f"\n📊 Batch Status: {status['status']}")

    if status['request_counts']:
        counts = status['request_counts']
        total = counts.get('total', 0)
        completed = counts.get('completed', 0)
        failed = counts.get('failed', 0)

        if total > 0:
            progress = (completed / total) * 100
            print(f"📈 Progress: {completed}/{total} ({progress:.1f}%)")
            if failed > 0:
                print(f"⚠️  Failed: {failed}")

    # ============================================================
    # STEP 3: Download Results if Complete
    # ============================================================
    if status['status'] == 'completed':
        print_section_header("STEP 3: DOWNLOAD RESULTS")

        results_file = Config.BATCH_OUTPUT_DIR / "batch_results_2022_2023.jsonl"

        # Check if already downloaded
        if results_file.exists():
            print(f"ℹ️  Results file already exists: {results_file}")
            response = input("   Download again? (y/N): ")
            if response.lower() != 'y':
                print("   Skipping download, using existing file")
            else:
                processor.download_results(batch_id, results_file)
        else:
            processor.download_results(batch_id, results_file)

        # ============================================================
        # STEP 4: Parse Results
        # ============================================================
        print_section_header("STEP 4: PARSE RESULTS")

        parsed_csv = Config.RESULTS_DIR / "sentiment_results_2022_2023.csv"

        # Check if already parsed
        if parsed_csv.exists():
            print(f"ℹ️  Parsed results already exist: {parsed_csv}")
            response = input("   Parse again? (y/N): ")
            if response.lower() != 'y':
                print("   Skipping parse, loading existing file")
                results_df = pd.read_csv(parsed_csv)
            else:
                results_df = processor.parse_results(results_file, parsed_csv)
        else:
            results_df = processor.parse_results(results_file, parsed_csv)

        # ============================================================
        # STEP 5: Validate Outputs
        # ============================================================
        print_section_header("STEP 5: VALIDATE LLM OUTPUTS")

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

        # Update batch info
        batch_info['status'] = 'completed_and_validated'
        batch_info['completed_at'] = datetime.now().isoformat()
        save_json(batch_info, batch_info_file)

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print_section_header("RESULTS READY ✅")

        print("\n📁 Files available:")
        print(f"   1. Raw results:      {results_file}")
        print(f"   2. Parsed results:   {parsed_csv}")
        print(f"   3. Validation report: {validation_file}")

        print(f"\n📊 Summary:")
        print(f"   Total speeches: {len(results_df)}")
        print(f"   Valid outputs: {validation_results['valid_speeches']}")
        print(f"   Validation rate: {validation_results['validation_rate']:.1f}%")

    elif status['status'] == 'in_progress' or status['status'] == 'validating':
        print("\n⏳ Batch is still processing...")
        print("   Check back later by running this script again")
        print(f"   Expected completion within 24 hours of submission")

    elif status['status'] == 'failed':
        print("\n❌ Batch job failed!")
        print("   Please check the error details on OpenAI dashboard")
        print(f"   Batch ID: {batch_id}")

    else:
        print(f"\n⚠️  Unknown status: {status['status']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
