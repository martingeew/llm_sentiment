"""
Split a large batch file into smaller chunks that fit within token limits.

Usage: python split_batch.py

This will split the batch_sample_2022_2023.jsonl file into multiple smaller files
that each stay under the 90,000 token queue limit.
"""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import Config
from utils import estimate_tokens


def split_batch_file(
    input_file: Path,
    output_dir: Path,
    max_tokens_per_batch: int = 80000  # Leave buffer below 90K limit
):
    """
    Split a batch file into smaller chunks.

    Args:
        input_file: Path to the large batch JSONL file
        output_dir: Directory to save the split batch files
        max_tokens_per_batch: Maximum tokens per batch (default 80K for safety)
    """

    print("=" * 70)
    print("BATCH FILE SPLITTER".center(70))
    print("=" * 70)

    # Read all requests
    print(f"\nReading batch file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_requests = [json.loads(line) for line in f]

    print(f"Total requests: {len(all_requests)}")

    # Split into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0

    for request in all_requests:
        # Estimate tokens for this request
        request_str = json.dumps(request)
        request_tokens = estimate_tokens(request_str)

        # If adding this request would exceed limit, start new chunk
        if current_tokens + request_tokens > max_tokens_per_batch and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [request]
            current_tokens = request_tokens
        else:
            current_chunk.append(request)
            current_tokens += request_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    print(f"\nSplit into {len(chunks)} chunks:")
    print(f"Target max tokens per chunk: {max_tokens_per_batch:,}")

    # Save chunks
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    for i, chunk in enumerate(chunks, 1):
        # Calculate chunk stats
        chunk_str = '\n'.join(json.dumps(req) for req in chunk)
        chunk_tokens = estimate_tokens(chunk_str)

        output_file = output_dir / f"batch_sample_2022_2023_chunk{i:02d}.jsonl"

        # Write chunk
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in chunk:
                f.write(json.dumps(request) + '\n')

        chunk_files.append(output_file)

        print(f"\n  Chunk {i}: {output_file.name}")
        print(f"    Requests: {len(chunk)}")
        print(f"    Est. tokens: {chunk_tokens:,}")
        print(f"    File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY".center(70))
    print("=" * 70)

    print(f"\nOriginal file: {len(all_requests)} requests")
    print(f"Split into: {len(chunks)} chunks")
    print(f"\nYou can now submit each chunk separately:")
    print(f"\n  python phase2_batch_submit.py --chunk 1")
    print(f"  python phase2_batch_submit.py --chunk 2")
    print(f"  etc.")

    print(f"\nChunk files saved to: {output_dir}")

    return chunk_files


if __name__ == "__main__":
    input_file = Config.BATCH_INPUT_DIR / "batch_sample_2022_2023.jsonl"
    output_dir = Config.BATCH_INPUT_DIR / "chunks"

    if not input_file.exists():
        print(f"\nError: Batch file not found: {input_file}")
        print("Please run phase1_batch_prep.py first")
        sys.exit(1)

    split_batch_file(input_file, output_dir)
