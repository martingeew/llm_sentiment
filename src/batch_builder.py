"""
Batch request builder for OpenAI Batch API.

This module creates the JSONL (JSON Lines) files needed for batch processing.

For beginners:
- The Batch API requires a special file format called JSONL
- JSONL = JSON Lines = one JSON object per line
- Each line is a separate API request
- OpenAI processes all requests together (batch processing)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from config import Config
from utils import estimate_tokens, print_section_header, print_progress_bar


class BatchRequestBuilder:
    """
    Builds batch request files for OpenAI's Batch API.

    This class:
    1. Takes a DataFrame of speeches
    2. Creates a prompt for each speech
    3. Formats them into the JSONL format OpenAI expects
    4. Saves to a file ready for upload
    """

    def __init__(self):
        """Initialize the batch request builder."""
        self.config = Config()

    def create_single_request(self, speech_id: str, speech_text: str,
                             speaker: str, institution: str, date: str) -> Dict[str, Any]:
        """
        Create a single batch API request for one speech.

        Args:
            speech_id: Unique identifier for the speech
            speech_text: The actual speech content
            speaker: Who gave the speech
            institution: Fed or ECB
            date: When it was given

        Returns:
            Dictionary formatted for OpenAI Batch API

        For beginners:
        - This creates ONE request in the format OpenAI expects
        - The "custom_id" helps us match results back to speeches later
        - The "body" contains the actual GPT-4 request
        """

        # Get the sentiment analysis prompt
        prompt = Config.get_sentiment_prompt(
            speech_text=speech_text,
            speaker=speaker,
            institution=institution,
            date=date
        )

        # Create the request in OpenAI's batch format
        request = {
            "custom_id": speech_id,  # Our unique ID to identify this speech
            "method": "POST",  # HTTP method (always POST for API calls)
            "url": "/v1/chat/completions",  # The API endpoint
            "body": {
                "model": Config.MODEL_NAME,  # Which GPT model to use
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in central bank communication analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "response_format": {"type": "json_object"},  # We want JSON output
                "temperature": Config.TEMPERATURE  # Controls randomness
            }
        }

        return request

    def build_batch_file(self, df: pd.DataFrame, output_file: Path,
                        text_column: str = 'text',
                        id_column: str = 'id',
                        speaker_column: str = 'speaker',
                        institution_column: str = 'institution',
                        date_column: str = 'date') -> Dict[str, Any]:
        """
        Build a complete batch request file from a DataFrame.

        Args:
            df: DataFrame containing speeches
            output_file: Where to save the JSONL file
            text_column: Name of column with speech text
            id_column: Name of column with speech IDs
            speaker_column: Name of column with speaker names
            institution_column: Name of column with institution
            date_column: Name of column with dates

        Returns:
            Dictionary with statistics about the batch file

        For beginners:
        - This processes ALL speeches in the DataFrame
        - Creates one request per speech
        - Saves them all to a JSONL file
        - Returns statistics about what was created
        """

        print_section_header("BUILDING BATCH REQUEST FILE")

        print(f"\n📝 Creating batch requests for {len(df)} speeches...")

        # Check that all required columns exist
        required_cols = [text_column, id_column, speaker_column,
                        institution_column, date_column]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Statistics tracking
        total_input_tokens = 0
        total_output_tokens_estimate = 0  # We estimate ~500 tokens output per speech
        OUTPUT_TOKENS_ESTIMATE = 500

        # Create requests
        requests = []

        for idx, row in df.iterrows():
            # Create unique ID for this speech
            speech_id = f"speech_{row[id_column]}"

            # Get speech details
            speech_text = str(row[text_column])
            speaker = str(row[speaker_column]) if pd.notna(row[speaker_column]) else "Unknown"
            institution = str(row[institution_column])
            date = str(row[date_column])

            # Truncate very long speeches to avoid token limits
            # GPT-4o has a 128K token context window, but we'll be conservative
            MAX_SPEECH_TOKENS = 8000
            estimated_tokens = estimate_tokens(speech_text)

            if estimated_tokens > MAX_SPEECH_TOKENS:
                # Truncate to fit within limits
                char_limit = MAX_SPEECH_TOKENS * 4  # Roughly 4 chars per token
                speech_text = speech_text[:char_limit] + "\n\n[Speech truncated due to length]"
                print(f"   ⚠️  Truncated speech {speech_id} (was {estimated_tokens} tokens)")

            # Create the request
            request = self.create_single_request(
                speech_id=speech_id,
                speech_text=speech_text,
                speaker=speaker,
                institution=institution,
                date=date
            )

            requests.append(request)

            # Estimate tokens for this request
            prompt_tokens = estimate_tokens(
                Config.get_sentiment_prompt(speech_text, speaker, institution, date)
            )
            total_input_tokens += prompt_tokens
            total_output_tokens_estimate += OUTPUT_TOKENS_ESTIMATE

            # Progress bar
            print_progress_bar(
                idx + 1, len(df),
                prefix="Creating requests:",
                suffix=f"{idx + 1}/{len(df)} speeches"
            )

        # Write to JSONL file
        print(f"\n💾 Writing {len(requests)} requests to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests:
                # Write each request as a JSON line
                f.write(json.dumps(request) + '\n')

        print(f"✓ Batch file created successfully!")

        # Calculate cost estimates
        batch_cost = (
            (total_input_tokens / 1000) * Config.BATCH_INPUT_PRICE_PER_1K +
            (total_output_tokens_estimate / 1000) * Config.BATCH_OUTPUT_PRICE_PER_1K
        )

        realtime_cost = (
            (total_input_tokens / 1000) * Config.REALTIME_INPUT_PRICE_PER_1K +
            (total_output_tokens_estimate / 1000) * Config.REALTIME_OUTPUT_PRICE_PER_1K
        )

        savings = realtime_cost - batch_cost

        # Summary statistics
        stats = {
            'num_requests': len(requests),
            'total_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens_estimate,
            'batch_cost_estimate': batch_cost,
            'realtime_cost_estimate': realtime_cost,
            'savings': savings,
            'output_file': str(output_file)
        }

        # Print summary
        print("\n" + "=" * 60)
        print("BATCH FILE SUMMARY")
        print("=" * 60)
        print(f"\n📊 Requests: {stats['num_requests']}")
        print(f"📏 Total input tokens: {stats['total_input_tokens']:,}")
        print(f"📏 Estimated output tokens: {stats['estimated_output_tokens']:,}")
        print(f"\n💰 Cost Estimates:")
        print(f"   Batch API:    ${stats['batch_cost_estimate']:.2f}")
        print(f"   Real-time API: ${stats['realtime_cost_estimate']:.2f}")
        print(f"   💵 Savings:    ${stats['savings']:.2f} ({savings/realtime_cost*100:.0f}% discount)")
        print(f"\n📁 File: {output_file}")
        print("=" * 60)

        return stats

    def validate_batch_file(self, file_path: Path) -> bool:
        """
        Validate that a batch file is correctly formatted.

        Args:
            file_path: Path to the JSONL file to validate

        Returns:
            True if valid

        For beginners:
        - This checks that the file we created is correct
        - Catches errors before we upload to OpenAI
        - Saves time and prevents wasted API calls
        """

        print(f"\n🔍 Validating batch file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Check each line is valid JSON
            for i, line in enumerate(lines, 1):
                try:
                    request = json.loads(line)

                    # Check required fields exist
                    required_fields = ['custom_id', 'method', 'url', 'body']
                    for field in required_fields:
                        if field not in request:
                            raise ValueError(f"Line {i}: Missing required field '{field}'")

                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {i}: Invalid JSON - {e}")

            print(f"✓ Batch file is valid! ({len(lines)} requests)")
            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    """
    Test the batch request builder.

    For beginners:
    - This creates a small test batch file
    - Run: python src/batch_builder.py
    """

    print_section_header("TESTING BATCH REQUEST BUILDER")

    # Create a small test DataFrame
    test_data = pd.DataFrame({
        'id': ['test_1', 'test_2'],
        'text': [
            'Inflation remains elevated and we are committed to bringing it back to our 2% target.',
            'The economy shows signs of moderating growth, which is what we want to see.'
        ],
        'speaker': ['Jerome Powell', 'Christine Lagarde'],
        'institution': ['Federal Reserve', 'ECB'],
        'date': ['2023-05-15', '2023-05-20']
    })

    # Build batch file
    builder = BatchRequestBuilder()
    test_file = Config.BATCH_INPUT_DIR / "test_batch.jsonl"

    stats = builder.build_batch_file(test_data, test_file)

    # Validate it
    is_valid = builder.validate_batch_file(test_file)

    if is_valid:
        print("\n✅ Batch request builder is working correctly!")
    else:
        print("\n❌ Something went wrong!")
