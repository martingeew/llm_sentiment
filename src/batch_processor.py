"""
Batch job processor for OpenAI Batch API.

This module handles:
- Uploading batch files to OpenAI
- Submitting batch jobs
- Monitoring job progress
- Downloading results

For beginners:
- The Batch API is asynchronous (you submit and wait)
- OpenAI processes your requests in the background (up to 24 hours)
- This module manages that entire workflow for you
"""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI
from config import Config
from utils import print_section_header, save_json, load_json


class BatchProcessor:
    """
    Manages batch processing jobs with OpenAI's Batch API.

    Workflow:
    1. Upload JSONL file → Get file ID
    2. Create batch job → Get batch ID
    3. Poll status → Wait for completion
    4. Download results → Parse and save
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the batch processor.

        Args:
            api_key: OpenAI API key (if not provided, uses from config)

        For beginners:
        - Creates connection to OpenAI's servers
        - API key is like a password that proves you're allowed to use the service
        """

        self.api_key = api_key or Config.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. " "Set OPENAI_API_KEY in your .env file"
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def upload_batch_file(self, file_path: Path) -> str:
        """
        Upload a batch file to OpenAI.

        Args:
            file_path: Path to the JSONL batch file

        Returns:
            File ID from OpenAI (needed to create the batch job)

        For beginners:
        - This uploads your file to OpenAI's servers
        - OpenAI gives you a file ID to reference it later
        - Think of it like uploading a file to Google Drive and getting a shareable link
        """

        print(f"\n📤 Uploading batch file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                # Upload file for batch processing
                file_response = self.client.files.create(
                    file=f, purpose="batch"  # Tells OpenAI this is for batch processing
                )

            file_id = file_response.id
            print(f"✓ File uploaded successfully!")
            print(f"   File ID: {file_id}")

            return file_id

        except Exception as e:
            print(f"❌ Error uploading file: {e}")
            raise

    def create_batch_job(self, file_id: str, description: Optional[str] = None) -> str:
        """
        Create a batch processing job.

        Args:
            file_id: The file ID from upload_batch_file()
            description: Optional description for the batch job

        Returns:
            Batch ID (needed to check status and get results)

        For beginners:
        - After uploading the file, we tell OpenAI to process it
        - This creates a "batch job" that runs in the background
        - OpenAI returns a batch ID to track this job
        """

        print(f"\n🚀 Creating batch job...")

        try:
            # Create the batch job
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",  # What API endpoint to use
                completion_window="24h",  # OpenAI has 24 hours to complete
                metadata={
                    "description": description
                    or "Central bank speech sentiment analysis"
                },
            )

            batch_id = batch_response.id
            print(f"✓ Batch job created successfully!")
            print(f"   Batch ID: {batch_id}")
            print(f"   Status: {batch_response.status}")

            return batch_id

        except Exception as e:
            print(f"❌ Error creating batch job: {e}")
            raise

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check the status of a batch job.

        Args:
            batch_id: The batch ID from create_batch_job()

        Returns:
            Dictionary with batch status information

        For beginners:
        - Batch jobs take time to complete (up to 24 hours)
        - This function checks if it's done yet
        - Returns information about progress
        """

        try:
            batch = self.client.batches.retrieve(batch_id)

            status_info = {
                "id": batch.id,
                "status": batch.status,  # validating, in_progress, completed, failed, etc.
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "failed_at": batch.failed_at,
                "request_counts": (
                    {
                        "total": batch.request_counts.total,
                        "completed": batch.request_counts.completed,
                        "failed": batch.request_counts.failed,
                    }
                    if batch.request_counts
                    else {}
                ),
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            }

            return status_info

        except Exception as e:
            print(f"❌ Error checking batch status: {e}")
            raise

    def wait_for_completion(
        self, batch_id: str, check_interval: int = None, timeout: int = None
    ) -> Dict[str, Any]:
        """
        Wait for a batch job to complete, with progress updates.

        Args:
            batch_id: The batch ID to monitor
            check_interval: How often to check status (seconds)
            timeout: Maximum time to wait (seconds)

        Returns:
            Final batch status information

        For beginners:
        - This function waits for your batch to finish
        - It checks every minute (by default) if it's done
        - Shows progress in the console
        - Stops when job completes or times out
        """

        check_interval = check_interval or Config.BATCH_CHECK_INTERVAL
        timeout = timeout or Config.BATCH_TIMEOUT

        print_section_header("MONITORING BATCH JOB")
        print(f"\n⏳ Waiting for batch job to complete...")
        print(f"   Batch ID: {batch_id}")
        print(f"   Check interval: {check_interval} seconds")
        print(f"   Timeout: {timeout / 3600:.1f} hours")
        print(f"\n   (This can take up to 24 hours. The API runs in the background.)")

        start_time = time.time()
        last_status = None

        while True:
            # Check how long we've been waiting
            elapsed = time.time() - start_time

            if elapsed > timeout:
                print(f"\n⏱️  Timeout reached ({timeout / 3600:.1f} hours)")
                print("   The batch is still processing. Check back later with:")
                print(f"   batch_id: {batch_id}")
                break

            # Get current status
            status_info = self.check_batch_status(batch_id)
            current_status = status_info["status"]

            # Print update if status changed
            if current_status != last_status:
                print(f"\n📊 Status: {current_status}")

                if status_info["request_counts"]:
                    counts = status_info["request_counts"]
                    total = counts.get("total", 0)
                    completed = counts.get("completed", 0)
                    failed = counts.get("failed", 0)

                    if total > 0:
                        progress = (completed / total) * 100
                        print(f"   Progress: {completed}/{total} ({progress:.1f}%)")
                        if failed > 0:
                            print(f"   ⚠️  Failed: {failed}")

                last_status = current_status

            # Check if job is finished
            if current_status == "completed":
                print(f"\n✅ Batch job completed successfully!")
                elapsed_hours = elapsed / 3600
                print(f"   Total time: {elapsed_hours:.2f} hours")
                return status_info

            elif current_status == "failed":
                print(f"\n❌ Batch job failed!")
                return status_info

            elif current_status in ["expired", "cancelled"]:
                print(f"\n⚠️  Batch job {current_status}!")
                return status_info

            # Wait before checking again
            print(
                f"   Waiting {check_interval} seconds... (elapsed: {elapsed / 60:.1f} min)",
                end="\r",
            )
            time.sleep(check_interval)

        return status_info

    def wait_for_completion_or_processing(
        self,
        batch_id: str,
        wait_for_completion: bool = False,
        timeout: int = 86400,
        poll_interval: int = 60,
    ) -> Dict[str, Any]:
        """
        Wait for batch to complete or start processing.

        Args:
            batch_id: The batch ID to monitor
            wait_for_completion: If True, wait until fully completed (slow but safe)
                                If False, wait until processing starts (faster but risky)
            timeout: Maximum time to wait in seconds (default 24 hours)
            poll_interval: How often to check in seconds (default 60 seconds)

        Returns:
            Current batch status information
        """

        start_time = time.time()
        last_status = None

        if wait_for_completion:
            print(f"   Waiting for batch to COMPLETE (safe mode, may take hours)...")
        else:
            print(f"   Waiting for batch to start processing...")

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                print(f"\n   Timeout (waited {elapsed/3600:.1f} hours)")
                break

            # Get current status
            status_info = self.check_batch_status(batch_id)
            current_status = status_info["status"]
            request_counts = status_info.get("request_counts", {})
            total_requests = request_counts.get("total", 0)
            completed_requests = request_counts.get("completed", 0)

            # Print update if status changed
            if current_status != last_status:
                print(f"   Status: {current_status}")
                last_status = current_status

            # If waiting for completion
            if wait_for_completion:
                if current_status == "completed":
                    elapsed_min = elapsed / 60
                    print(f"   Batch completed! (took {elapsed_min:.1f} min)")
                    return status_info
                elif current_status in ["failed", "expired", "cancelled"]:
                    print(f"   Batch entered terminal state: {current_status}")
                    return status_info

                # Show progress
                if total_requests > 0 and completed_requests > 0:
                    progress = (completed_requests / total_requests) * 100
                    print(
                        f"   Progress: {completed_requests}/{total_requests} ({progress:.0f}%)",
                        end="\r",
                    )

            else:
                # If just waiting for processing to start
                if current_status != "validating" and total_requests > 0:
                    elapsed_min = elapsed / 60
                    print(f"   Batch started processing (after {elapsed_min:.1f} min)")
                    return status_info

                if current_status in ["failed", "completed", "expired", "cancelled"]:
                    print(f"   Batch entered terminal state: {current_status}")
                    return status_info

            # Wait before checking again
            time.sleep(poll_interval)

        # Return current status even if timeout
        return self.check_batch_status(batch_id)

    def wait_for_in_progress(
        self, batch_id: str, timeout: int = 86400, poll_interval: int = 60
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use wait_for_completion_or_processing instead.
        Kept for backward compatibility.
        """
        return self.wait_for_completion_or_processing(
            batch_id,
            wait_for_completion=True,  # Use safe mode by default
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def download_results(self, batch_id: str, output_file: Path) -> Path:
        """
        Download results from a completed batch job.

        Args:
            batch_id: The completed batch ID
            output_file: Where to save the results

        Returns:
            Path to the saved results file

        For beginners:
        - After a batch completes, results are stored on OpenAI's servers
        - This downloads those results to your computer
        - Results are saved as JSONL (one result per line)
        """

        print(f"\n⬇️  Downloading results...")

        try:
            # Get batch status to find output file
            status_info = self.check_batch_status(batch_id)

            if status_info["status"] != "completed":
                raise ValueError(
                    f"Batch is not completed (status: {status_info['status']})"
                )

            output_file_id = status_info["output_file_id"]

            if not output_file_id:
                raise ValueError("No output file available")

            # Download the file
            file_content = self.client.files.content(output_file_id)

            # Save to disk
            with open(output_file, "wb") as f:
                f.write(file_content.content)

            print(f"✓ Results saved to: {output_file}")

            return output_file

        except Exception as e:
            print(f"❌ Error downloading results: {e}")
            raise

    def parse_results(self, results_file: Path, output_csv: Path) -> pd.DataFrame:
        """
        Parse batch results into a structured DataFrame.

        Args:
            results_file: Path to the JSONL results file
            output_csv: Where to save the parsed CSV

        Returns:
            DataFrame with parsed sentiment analysis results

        For beginners:
        - Raw results from OpenAI are in JSONL format
        - This parses them into a nice table (CSV) you can analyze
        - Each row is one speech with its sentiment scores
        """

        print(f"\n📊 Parsing results...")

        results = []

        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)

                # Extract the data we need
                custom_id = result["custom_id"]
                response = result["response"]

                if response["status_code"] == 200:
                    # Successful response
                    message_content = response["body"]["choices"][0]["message"][
                        "content"
                    ]

                    # Parse the JSON response from GPT
                    try:
                        sentiment_data = json.loads(message_content)

                        # Flatten the nested structure
                        flat_result = {
                            "speech_id": custom_id,
                            "hawkish_dovish_score": sentiment_data.get(
                                "hawkish_dovish_score"
                            ),
                            "topic_inflation": sentiment_data.get("topics", {}).get(
                                "inflation"
                            ),
                            "topic_growth": sentiment_data.get("topics", {}).get(
                                "growth"
                            ),
                            "topic_financial_stability": sentiment_data.get(
                                "topics", {}
                            ).get("financial_stability"),
                            "topic_labor_market": sentiment_data.get("topics", {}).get(
                                "labor_market"
                            ),
                            "topic_international": sentiment_data.get("topics", {}).get(
                                "international"
                            ),
                            "uncertainty": sentiment_data.get("uncertainty"),
                            "forward_guidance_strength": sentiment_data.get(
                                "forward_guidance_strength"
                            ),
                            "market_impact_stocks": sentiment_data.get(
                                "market_impact", {}
                            ).get("stocks"),
                            "market_impact_bonds": sentiment_data.get(
                                "market_impact", {}
                            ).get("bonds"),
                            "market_impact_currency": sentiment_data.get(
                                "market_impact", {}
                            ).get("currency"),
                            "market_impact_reasoning": sentiment_data.get(
                                "market_impact", {}
                            ).get("reasoning"),
                            "summary": sentiment_data.get("summary"),
                            "key_sentences": "|".join(
                                sentiment_data.get("key_sentences", [])
                            ),
                        }

                        results.append(flat_result)

                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  Failed to parse response for {custom_id}: {e}")

                else:
                    print(
                        f"   ⚠️  Request failed for {custom_id}: {response['status_code']}"
                    )

        # Create DataFrame
        df = pd.DataFrame(results)

        print(f"✓ Parsed {len(df)} results")

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}")

        return df


# Example usage
if __name__ == "__main__":
    """
    This demonstrates how to use the BatchProcessor.

    WARNING: This will use real API calls if you have an API key set!
    """

    print_section_header("BATCH PROCESSOR DEMO")
    print("\n⚠️  This demo will make real API calls if you run it!")
    print("    Uncomment the code below to test with a real batch file.\n")

    # Uncomment to test (make sure you have a valid batch file first!)
    """
    processor = BatchProcessor()

    # Upload file
    file_id = processor.upload_batch_file(
        Config.BATCH_INPUT_DIR / "test_batch.jsonl"
    )

    # Create batch job
    batch_id = processor.create_batch_job(
        file_id,
        description="Test batch for Phase 1"
    )

    # Monitor until complete
    status = processor.wait_for_completion(batch_id)

    # Download results
    if status['status'] == 'completed':
        results_file = processor.download_results(
            batch_id,
            Config.BATCH_OUTPUT_DIR / "test_results.jsonl"
        )

        # Parse results
        df = processor.parse_results(
            results_file,
            Config.RESULTS_DIR / "test_sentiment_results.csv"
        )

        print(f"\n✅ Complete! Processed {len(df)} speeches")
    """

    print("✅ Batch processor code is ready to use!")
