"""
Data loader module for ECB-FED speeches dataset.

This module handles downloading and loading the central bank speeches
dataset from Hugging Face.

For beginners:
- Hugging Face is like GitHub for datasets and AI models
- The 'datasets' library makes it easy to download and work with data
- We load the data once and save it locally to avoid re-downloading
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Optional
from config import Config


class DataLoader:
    """
    Loads and manages the ECB-FED speeches dataset.

    This class handles:
    1. Downloading the dataset from Hugging Face
    2. Converting it to pandas DataFrame (easier to work with)
    3. Saving it locally
    4. Loading from local storage (faster for subsequent runs)
    """

    def __init__(self):
        """Initialize the DataLoader with configuration settings."""
        self.dataset_name = Config.DATASET_NAME
        self.raw_data_dir = Config.RAW_DATA_DIR

    def load_from_huggingface(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load the dataset from Hugging Face.

        Args:
            force_download: If True, download even if local copy exists

        Returns:
            DataFrame containing all speeches with metadata

        For beginners:
        - This function first checks if we already have the data saved locally
        - If yes, it loads from local file (much faster!)
        - If no, it downloads from Hugging Face and saves for next time
        """

        # File path where we save the data locally
        local_file = self.raw_data_dir / "ecb_fed_speeches.parquet"

        # Check if we already have the data saved
        if local_file.exists() and not force_download:
            print(f"📂 Loading dataset from local file: {local_file}")
            df = pd.read_parquet(local_file)
            print(f"✓ Loaded {len(df)} speeches from local storage")
            return df

        # Download from Hugging Face
        print(f"🌐 Downloading dataset from Hugging Face: {self.dataset_name}")
        print("   (This may take a few minutes on first run...)")

        try:
            # Load dataset using Hugging Face datasets library
            dataset = load_dataset(self.dataset_name)

            # Convert to pandas DataFrame
            # Note: Hugging Face datasets usually have a 'train' split
            # We check what splits are available and use the first one
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no 'train' split, use the first available split
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()

            print(f"✓ Downloaded {len(df)} speeches")

            # Save to local file for faster loading next time
            # We use parquet format (compressed, efficient)
            print(f"💾 Saving to local file: {local_file}")
            df.to_parquet(local_file, index=False)
            print("✓ Saved successfully")

            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about the dataset.

        Args:
            df: The speeches DataFrame

        Returns:
            Dictionary with dataset statistics

        For beginners:
        - This function gives us a quick overview of what's in the dataset
        - Useful for understanding the data before analysis
        """

        info = {
            'total_speeches': len(df),
            'columns': list(df.columns),
            'institutions': df['institution'].value_counts().to_dict() if 'institution' in df.columns else {},
            'date_range': {
                'earliest': df['date'].min() if 'date' in df.columns else None,
                'latest': df['date'].max() if 'date' in df.columns else None
            },
            'year_distribution': df['year'].value_counts().sort_index().to_dict() if 'year' in df.columns else {}
        }

        return info

    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Print a nicely formatted summary of the dataset.

        Args:
            df: The speeches DataFrame

        For beginners:
        - This makes it easy to see what's in your data
        - Always good to check your data before processing!
        """

        info = self.get_dataset_info(df)

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"\n📊 Total speeches: {info['total_speeches']}")

        print(f"\n📋 Available columns:")
        for col in info['columns']:
            print(f"   - {col}")

        if info['institutions']:
            print(f"\n🏛️  Speeches by institution:")
            for institution, count in info['institutions'].items():
                print(f"   - {institution}: {count}")

        if info['date_range']['earliest']:
            print(f"\n📅 Date range:")
            print(f"   - Earliest: {info['date_range']['earliest']}")
            print(f"   - Latest: {info['date_range']['latest']}")

        if info['year_distribution']:
            print(f"\n📈 Speeches by year (showing last 10 years):")
            years = sorted(info['year_distribution'].keys(), reverse=True)[:10]
            for year in years:
                count = info['year_distribution'][year]
                bar = "█" * min(50, count // 2)  # Visual bar chart
                print(f"   {year}: {count:3d} {bar}")

        print("\n" + "=" * 60)

    def sample_data(self, df: pd.DataFrame, start_year: int, end_year: int,
                   institution: Optional[str] = None) -> pd.DataFrame:
        """
        Extract a sample of speeches from specific years.

        Args:
            df: The full speeches DataFrame
            start_year: First year to include (e.g., 2022)
            end_year: Last year to include (e.g., 2023)
            institution: Optional filter by institution ('Fed' or 'ECB')

        Returns:
            DataFrame with filtered speeches

        For beginners:
        - This lets us work with a smaller subset of data for testing
        - Like taking a representative sample before processing everything
        - Much faster and cheaper for initial testing!
        """

        print(f"\n🔍 Sampling data:")
        print(f"   Years: {start_year} to {end_year}")
        if institution:
            print(f"   Institution: {institution}")

        # Ensure we have a 'year' column
        if 'year' not in df.columns and 'date' in df.columns:
            # Extract year from date if needed
            df['year'] = pd.to_datetime(df['date']).dt.year

        # Filter by year
        sample = df[
            (df['year'] >= start_year) &
            (df['year'] <= end_year)
        ].copy()

        # Filter by institution if specified
        if institution and 'institution' in df.columns:
            sample = sample[sample['institution'] == institution]

        print(f"✓ Sampled {len(sample)} speeches")

        return sample


# Example usage and testing
if __name__ == "__main__":
    """
    This code runs when you execute this file directly.
    It demonstrates how to use the DataLoader class.

    For beginners:
    - This is a good way to test if the module works
    - Run: python src/data_loader.py
    """

    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)

    # Create a data loader
    loader = DataLoader()

    # Load the dataset
    df = loader.load_from_huggingface()

    # Print summary
    loader.print_dataset_summary(df)

    # Test sampling (2022-2023)
    sample = loader.sample_data(
        df,
        start_year=Config.SAMPLE_START_YEAR,
        end_year=Config.SAMPLE_END_YEAR
    )

    print(f"\n✅ Sample contains {len(sample)} speeches")
    print("\nFirst few speeches:")
    print(sample[['date', 'institution', 'speaker']].head() if 'speaker' in sample.columns else sample[['date', 'institution']].head())
