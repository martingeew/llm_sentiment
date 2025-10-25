"""
Utility functions for the LLM Sentiment Analysis project.

This module contains helper functions used throughout the project.

For beginners:
- Utilities are like tools in a toolbox
- These are common operations we need in multiple places
- Instead of repeating code, we write it once here and reuse it
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    For beginners:
    - GPT models charge based on "tokens" (roughly word pieces)
    - 1 token ≈ 4 characters in English
    - 100 tokens ≈ 75 words
    - This is a rough estimate (OpenAI's actual tokenizer is more complex)

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated number of tokens
    """

    # Rough estimation: 1 token ≈ 4 characters
    # This is a simplification but good enough for cost estimation
    return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int,
                  input_price_per_1k: float, output_price_per_1k: float) -> float:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        input_tokens: Number of input tokens (the prompt)
        output_tokens: Number of output tokens (the response)
        input_price_per_1k: Price per 1,000 input tokens
        output_price_per_1k: Price per 1,000 output tokens

    Returns:
        Total cost in dollars

    For beginners:
    - OpenAI charges separately for input (your prompt) and output (GPT's response)
    - Input is usually cheaper than output
    - Prices are per 1,000 tokens, so we divide by 1000
    """

    input_cost = (input_tokens / 1000) * input_price_per_1k
    output_cost = (output_tokens / 1000) * output_price_per_1k
    total_cost = input_cost + output_cost

    return total_cost


def format_cost(cost: float) -> str:
    """
    Format a cost value for display.

    Args:
        cost: Cost in dollars

    Returns:
        Nicely formatted cost string

    For beginners:
    - Makes costs easier to read
    - Shows cents for small amounts, dollars for large amounts
    """

    if cost < 0.01:
        return f"${cost * 100:.4f}¢"  # Show in cents if less than 1 cent
    elif cost < 1.0:
        return f"${cost:.4f}"  # Show 4 decimal places
    else:
        return f"${cost:.2f}"  # Show 2 decimal places for dollars


def save_json(data: Dict[Any, Any], file_path: Path):
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Where to save the file

    For beginners:
    - JSON is a common format for structured data
    - It's human-readable (you can open it in a text editor)
    - It's also easy for programs to read and write
    """

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved JSON to: {file_path}")


def load_json(file_path: Path) -> Dict[Any, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with the loaded data
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_dataframe(df: pd.DataFrame, file_path: Path, format: str = 'csv'):
    """
    Save a pandas DataFrame to file.

    Args:
        df: The DataFrame to save
        file_path: Where to save it
        format: File format ('csv', 'parquet', or 'excel')

    For beginners:
    - CSV is human-readable but larger files
    - Parquet is compressed and faster but not human-readable
    - Excel is good for sharing with non-technical people
    """

    if format == 'csv':
        df.to_csv(file_path, index=False)
    elif format == 'parquet':
        df.to_parquet(file_path, index=False)
    elif format == 'excel':
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Saved DataFrame to: {file_path}")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: What to add at the end if truncated

    Returns:
        Truncated text

    For beginners:
    - Useful for displaying long text in a readable way
    - Shows first part of text + "..." to indicate there's more
    """

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def create_timestamp() -> str:
    """
    Create a timestamp string for file naming.

    Returns:
        Timestamp string like '20240125_143022'

    For beginners:
    - Useful for creating unique file names
    - Format: YYYYMMDD_HHMMSS
    - Makes it easy to sort files by date
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section_header(title: str, width: int = 60):
    """
    Print a nicely formatted section header.

    Args:
        title: The header title
        width: Width of the header line

    For beginners:
    - Makes console output easier to read
    - Visual separation between different sections
    """

    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_progress_bar(current: int, total: int, prefix: str = "",
                      suffix: str = "", length: int = 40):
    """
    Print a progress bar to the console.

    Args:
        current: Current progress
        total: Total items
        prefix: Text before the bar
        suffix: Text after the bar
        length: Length of the progress bar

    For beginners:
    - Shows visual progress in the console
    - Helpful when processing many items
    """

    if total == 0:
        percent = 100
    else:
        percent = int(100 * current / total)

    filled = int(length * current / total) if total > 0 else length
    bar = "█" * filled + "-" * (length - filled)

    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="")

    if current == total:
        print()  # New line when complete


def validate_dataframe_columns(df: pd.DataFrame, required_columns: list,
                               df_name: str = "DataFrame") -> bool:
    """
    Check if a DataFrame has all required columns.

    Args:
        df: DataFrame to check
        required_columns: List of column names that must exist
        df_name: Name of the DataFrame (for error messages)

    Returns:
        True if all columns exist

    Raises:
        ValueError: If any required column is missing

    For beginners:
    - Data validation prevents errors later
    - Better to catch problems early than crash during processing
    """

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return True


# Example usage
if __name__ == "__main__":
    """
    Test the utility functions.
    """

    print_section_header("TESTING UTILITY FUNCTIONS")

    # Test token estimation
    text = "This is a sample text for testing token estimation."
    tokens = estimate_tokens(text)
    print(f"\nText: '{text}'")
    print(f"Estimated tokens: {tokens}")

    # Test cost calculation
    cost = calculate_cost(1000, 500, 0.0025, 0.010)
    print(f"\nCost for 1000 input tokens and 500 output tokens:")
    print(f"Raw: ${cost:.6f}")
    print(f"Formatted: {format_cost(cost)}")

    # Test timestamp
    timestamp = create_timestamp()
    print(f"\nCurrent timestamp: {timestamp}")

    # Test progress bar
    print("\nProgress bar demo:")
    import time
    for i in range(101):
        print_progress_bar(i, 100, prefix="Processing:", suffix="Complete")
        time.sleep(0.01)

    print("\n✅ All utility functions working!")
