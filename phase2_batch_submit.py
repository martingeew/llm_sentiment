"""
Phase 2: Batch Job Submission and Processing

This script submits the batch file to OpenAI, monitors progress,
downloads results, and validates LLM outputs.

For beginners:
- Run this AFTER Phase 1 (batch file must exist)
- This will make REAL API calls and incur costs
- Estimated cost: ~$2.32 for 2022-2023 sample
- Process takes up to 24 hours (usually 30min - 4 hours)

Usage:
    python phase2_batch_submit.py              # Auto-detect chunks or single batch
    python phase2_batch_submit.py --all-chunks # Submit all chunks sequentially
    python phase2_batch_submit.py --chunk 1    # Submit single chunk

IMPORTANT: This script makes actual API calls and charges will apply!
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from batch_processor import BatchProcessor
from output_validator import OutputValidator
from utils import print_section_header, save_json, load_json


def detect_chunks():
    """Detect if batch has been split into chunks."""
    chunks_dir = Config.BATCH_INPUT_DIR / "chunks"
    if not chunks_dir.exists():
        return None

    # Find all chunk files
    chunk_files = sorted(chunks_dir.glob("batch_sample_2022_2023_chunk*.jsonl"))
    return chunk_files if chunk_files else None


def submit_single_chunk(chunk_num, chunk_file, processor):
    """
    Submit a single chunk and wait for it to enter 'in_progress' state.

    Returns:
        Dictionary with chunk submission info
    """
    print(f"\n{'='*70}")
    print(f"CHUNK {chunk_num}".center(70))
    print(f"{'='*70}")

    # Count requests in chunk
    with open(chunk_file, "r", encoding="utf-8") as f:
        num_requests = sum(1 for _ in f)

    print(f"\nChunk file: {chunk_file.name}")
    print(f"Speeches: {num_requests}")

    # Upload file
    file_id = processor.upload_batch_file(chunk_file)

    # Create batch job
    batch_id = processor.create_batch_job(
        file_id, description=f"Phase 2: ECB-FED Speeches 2022-2023 - Chunk {chunk_num}"
    )

    # Save chunk info
    chunk_info = {
        "chunk_number": chunk_num,
        "batch_id": batch_id,
        "file_id": file_id,
        "submitted_at": datetime.now().isoformat(),
        "chunk_file": str(chunk_file),
        "num_requests": num_requests,
        "status": "submitted",
    }

    chunk_info_file = Config.RESULTS_DIR / f"phase2_chunk{chunk_num:02d}_info.json"
    save_json(chunk_info, chunk_info_file)

    print(f"\n   Batch ID: {batch_id}")
    print(f"   Info saved: {chunk_info_file.name}")

    # Wait for it to enter 'in_progress' state (exits validation queue)
    print(f"\n   Waiting for chunk to start processing...")
    status_info = processor.wait_for_in_progress(batch_id)

    # Update chunk info with current status
    chunk_info["status"] = status_info["status"]
    chunk_info["status_checked_at"] = datetime.now().isoformat()
    save_json(chunk_info, chunk_info_file)

    return chunk_info


def submit_all_chunks(chunk_files):
    """
    Submit all chunks using smart sequential approach.
    """
    print("=" * 70)
    print("PHASE 2: CHUNKED BATCH SUBMISSION".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    print(f"\n   Found {len(chunk_files)} chunks to submit")
    print(
        f"   Strategy: Smart sequential (submit, wait until processing starts, repeat)"
    )
    print(
        f"   Estimated time: ~{len(chunk_files) * 3:.0f}-{len(chunk_files) * 5:.0f} minutes (~{len(chunk_files) * 4 / 60:.1f} hours)"
    )

    # Calculate total cost estimate
    total_requests = 0
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            total_requests += sum(1 for _ in f)

    estimated_cost = total_requests * 0.0075  # Rough estimate
    print(f"\n   Total speeches: {total_requests}")
    print(f"   Estimated cost: ${estimated_cost:.2f}")

    # Safety check
    print("\n" + "=" * 70)
    print("   WARNING: This will make REAL API calls".upper())
    print("=" * 70)

    response = input("\n   Type 'YES' to proceed with all chunks: ")
    if response.upper() != "YES":
        print("\n   Aborted by user. No charges incurred.")
        return None

    # Validate configuration
    print_section_header("VALIDATE CONFIGURATION")
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n   Configuration Error: {e}")
        return None

    # Submit all chunks
    print_section_header("SUBMITTING ALL CHUNKS")
    processor = BatchProcessor()
    submitted_chunks = []
    skipped_chunks = []

    for i, chunk_file in enumerate(chunk_files, 1):
        # Check if chunk was already submitted
        chunk_info_file = Config.RESULTS_DIR / f"phase2_chunk{i:02d}_info.json"

        if chunk_info_file.exists():
            existing_info = load_json(chunk_info_file)
            batch_id = existing_info["batch_id"]

            # Check current status
            try:
                status = processor.check_batch_status(batch_id)
                current_status = status["status"]

                if current_status in ["in_progress", "completed"]:
                    print(f"\n   Chunk {i}: Already {current_status}, skipping")
                    skipped_chunks.append(i)
                    continue
                elif current_status == "failed":
                    print(f"\n   Chunk {i}: Previously failed, will retry")
                else:
                    print(f"\n   Chunk {i}: Status is {current_status}, will resubmit")
            except Exception as e:
                print(f"\n   Chunk {i}: Error checking existing batch, will resubmit")

        # Submit the chunk
        try:
            chunk_info = submit_single_chunk(i, chunk_file, processor)
            submitted_chunks.append(chunk_info)

            print(f"\n   Chunk {i}/{len(chunk_files)} submitted successfully!")

            if chunk_info["status"] == "in_progress":
                print(f"   Status: in_progress (out of queue, can submit next)")
            else:
                print(f"   Status: {chunk_info['status']}")

        except Exception as e:
            print(f"\n   Error submitting chunk {i}: {e}")
            print(
                f"   Successfully submitted {len(submitted_chunks)}/{len(chunk_files)} chunks"
            )
            print(
                f"   Skipped {len(skipped_chunks)} chunks (already processing/completed)"
            )
            return submitted_chunks

    # All chunks processed
    print("\n" + "=" * 70)
    print("SUBMISSION COMPLETE!".center(70))
    print("=" * 70)

    print(f"\n   Newly submitted: {len(submitted_chunks)}")
    print(f"   Skipped (already processing/completed): {len(skipped_chunks)}")
    print(f"   Total chunks: {len(chunk_files)}")

    if submitted_chunks:
        print(f"\n   YOU CAN NOW CLOSE YOUR LAPTOP")
        print(f"   All chunks are processing on OpenAI's servers")
    else:
        print(f"\n   All chunks already submitted!")

    print(f"\n   Next steps:")
    print(f"   1. Check progress: python phase2_check_status.py --all-chunks")
    print(f"   2. Download results: python phase2_download_results.py")

    print("\n" + "=" * 70)

    return submitted_chunks


def main():
    """
    Main function for Phase 2: Submit batch and process results.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Submit batch jobs to OpenAI for sentiment analysis"
    )
    parser.add_argument(
        "--all-chunks", action="store_true", help="Submit all chunks sequentially"
    )
    parser.add_argument("--chunk", type=int, help="Submit a specific chunk number")

    args = parser.parse_args()

    # Detect if chunks exist
    chunk_files = detect_chunks()

    # Determine mode
    if args.all_chunks:
        # Explicitly requested all chunks
        if not chunk_files:
            print("\n   Error: --all-chunks specified but no chunks found")
            print(f"   Run: python split_batch.py")
            return
        submit_all_chunks(chunk_files)
        return

    elif args.chunk:
        # Submit specific chunk
        if not chunk_files:
            print(f"\n   Error: --chunk specified but no chunks found")
            print(f"   Run: python split_batch.py")
            return

        chunk_num = args.chunk
        if chunk_num < 1 or chunk_num > len(chunk_files):
            print(f"\n   Error: Chunk {chunk_num} not found")
            print(f"   Valid chunks: 1-{len(chunk_files)}")
            return

        processor = BatchProcessor()
        chunk_file = chunk_files[chunk_num - 1]
        submit_single_chunk(chunk_num, chunk_file, processor)
        return

    # Auto-detect mode (no explicit flags)
    if chunk_files:
        # Chunks exist, ask user
        print("=" * 70)
        print("CHUNK FILES DETECTED".center(70))
        print("=" * 70)

        print(f"\n   Found {len(chunk_files)} chunk files")
        print(f"\n   Options:")
        print(f"   1. Submit all chunks: python phase2_batch_submit.py --all-chunks")
        print(f"   2. Submit single chunk: python phase2_batch_submit.py --chunk N")
        print(f"\n   Run with --all-chunks to proceed")
        return

    # No chunks, use original workflow
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
    if response.upper() != "YES":
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
        file_id, description="Phase 2: ECB-FED Speeches 2022-2023 Sentiment Analysis"
    )

    # Save batch ID for later reference
    batch_info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "submitted_at": datetime.now().isoformat(),
        "batch_file": str(batch_file),
        "status": "submitted",
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
    if response.lower() == "n":
        print("\n✅ Batch submitted successfully!")
        print(f"   Batch ID: {batch_id}")
        print(
            "\n   Run phase2_check_status.py later to check progress and download results."
        )
        return

    # Wait for completion
    status = processor.wait_for_completion(batch_id)

    # Update batch info with completion status
    batch_info["status"] = status["status"]
    batch_info["completed_at"] = datetime.now().isoformat()
    save_json(batch_info, batch_info_file)

    # ============================================================
    # STEP 5: Download Results
    # ============================================================
    if status["status"] == "completed":
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
            "validation_results": validation_results,
            "distributions": distributions,
            "validated_at": datetime.now().isoformat(),
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

        if validation_results["validation_rate"] >= 95:
            print(
                f"\n✅ Excellent! {validation_results['validation_rate']:.1f}% validation rate"
            )
        elif validation_results["validation_rate"] >= 85:
            print(
                f"\n⚠️  Good, but some issues: {validation_results['validation_rate']:.1f}% validation rate"
            )
        else:
            print(
                f"\n❌ Low validation rate: {validation_results['validation_rate']:.1f}%"
            )
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
