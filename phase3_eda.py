"""
Phase 3: Exploratory Data Analysis

This script performs comprehensive exploratory analysis on the prepared
sentiment dataset to understand patterns before building indices.

For beginners:
- EDA = Exploratory Data Analysis
- Helps understand data distributions, patterns, and quality
- Guides decisions for index construction

Usage: python phase3_eda.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from utils import print_section_header


def main():
    """
    Perform exploratory data analysis on prepared sentiment data.
    """

    print("=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Prepared Dataset
    # ============================================================
    print_section_header("STEP 1: LOAD PREPARED DATASET")

    data_file = Config.RESULTS_DIR / "phase3_prepared_data.csv"

    if not data_file.exists():
        print(f"\nERROR Error: Prepared data not found!")
        print(f"   Expected: {data_file}")
        print(f"\n   Run: python phase3_data_prep.py")
        return

    df = pd.read_csv(data_file, parse_dates=['date'])

    print(f"\nOK Loaded prepared dataset")
    print(f"   File: {data_file.name}")
    print(f"   Speeches: {len(df)}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ============================================================
    # STEP 2: Speeches by Month and Institution
    # ============================================================
    print_section_header("STEP 2: SPEECHES BY MONTH & INSTITUTION")

    # Add year-month column for grouping
    df['year_month'] = df['date'].dt.to_period('M')

    # Cross-tabulation: month x institution
    month_inst_counts = pd.crosstab(
        df['year_month'],
        df['country'],
        margins=True,
        margins_name='Total'
    )

    print(f"\n   Speeches by Month and Institution:")
    print(f"\n{month_inst_counts.to_string()}")

    # Summary statistics
    print(f"\n   Summary by Institution:")
    for institution in df['country'].unique():
        inst_df = df[df['country'] == institution]
        print(f"\n   {institution}:")
        print(f"      Total speeches: {len(inst_df)}")
        print(f"      Avg per month: {len(inst_df) / df['year_month'].nunique():.1f}")
        print(f"      Most active month: {inst_df.groupby('year_month').size().idxmax()} "
              f"({inst_df.groupby('year_month').size().max()} speeches)")

    # ============================================================
    # STEP 3: Speeches by Date and Institution
    # ============================================================
    print_section_header("STEP 3: SPEECHES BY DATE & INSTITUTION")

    # Count speeches per day by institution
    date_inst_counts = df.groupby(['date', 'country']).size().reset_index(name='count')

    # Show days with multiple speeches
    multi_speech_days = date_inst_counts[date_inst_counts['count'] > 1].sort_values('count', ascending=False)

    print(f"\n   Total unique speech dates: {df['date'].nunique()}")
    print(f"\n   Days with multiple speeches (top 10):")
    if len(multi_speech_days) > 0:
        print(f"\n{multi_speech_days.head(10).to_string(index=False)}")
    else:
        print(f"\n   No days with multiple speeches")

    # Distribution of speeches per day
    speeches_per_day = df.groupby('date').size()
    print(f"\n   Distribution of speeches per day:")
    print(f"      Mean: {speeches_per_day.mean():.2f}")
    print(f"      Median: {speeches_per_day.median():.0f}")
    print(f"      Max: {speeches_per_day.max():.0f}")
    print(f"      Days with 1 speech: {(speeches_per_day == 1).sum()}")
    print(f"      Days with 2+ speeches: {(speeches_per_day >= 2).sum()}")

    # ============================================================
    # STEP 4: Score Distributions by Institution
    # ============================================================
    print_section_header("STEP 4: SCORE DISTRIBUTIONS BY INSTITUTION")

    score_columns = [
        'hawkish_dovish_score',
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international',
        'uncertainty',
        'forward_guidance_strength'
    ]

    for col in score_columns:
        if col not in df.columns:
            continue

        print(f"\n   {col.upper()}:")
        print(f"   {'Institution':<15} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'25%':<8} {'50%':<8} {'75%':<8} {'Max':<8}")
        print(f"   {'-'*95}")

        # Overall statistics
        overall_stats = df[col].describe()
        print(f"   {'OVERALL':<15} {len(df):<8} {overall_stats['mean']:<8.1f} {overall_stats['std']:<8.1f} "
              f"{overall_stats['min']:<8.1f} {overall_stats['25%']:<8.1f} {overall_stats['50%']:<8.1f} "
              f"{overall_stats['75%']:<8.1f} {overall_stats['max']:<8.1f}")

        # By institution
        for institution in sorted(df['country'].unique()):
            inst_data = df[df['country'] == institution][col]
            stats = inst_data.describe()
            print(f"   {institution:<15} {len(inst_data):<8} {stats['mean']:<8.1f} {stats['std']:<8.1f} "
                  f"{stats['min']:<8.1f} {stats['25%']:<8.1f} {stats['50%']:<8.1f} "
                  f"{stats['75%']:<8.1f} {stats['max']:<8.1f}")

        # Check for differences between institutions
        fed_mean = df[df['country'] == 'United States'][col].mean()
        ecb_mean = df[df['country'] == 'Euro area'][col].mean()
        diff = fed_mean - ecb_mean
        print(f"\n   Difference (Fed - ECB): {diff:+.1f}")
        if abs(diff) > 10:
            print(f"   NOTE: Substantial difference between institutions")

    # ============================================================
    # STEP 5: Market Impact Value Counts by Institution
    # ============================================================
    print_section_header("STEP 5: MARKET IMPACT BY INSTITUTION")

    market_columns = [
        'market_impact_stocks',
        'market_impact_bonds',
        'market_impact_currency'
    ]

    for col in market_columns:
        if col not in df.columns:
            continue

        print(f"\n   {col.upper()}:")

        # Overall counts
        print(f"\n   Overall:")
        value_counts = df[col].value_counts().sort_index()
        total = len(df)
        for value, count in value_counts.items():
            pct = (count / total) * 100
            print(f"      {value:<10}: {count:>4} ({pct:>5.1f}%)")

        # By institution
        for institution in sorted(df['country'].unique()):
            print(f"\n   {institution}:")
            inst_data = df[df['country'] == institution][col]
            value_counts = inst_data.value_counts().sort_index()
            inst_total = len(inst_data)
            for value, count in value_counts.items():
                pct = (count / inst_total) * 100
                print(f"      {value:<10}: {count:>4} ({pct:>5.1f}%)")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS COMPLETE".center(70))
    print("=" * 70)

    print(f"\nKey Findings:")
    print(f"   1. Total speeches: {len(df)} ({df['country'].value_counts().to_dict()})")
    print(f"   2. Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   3. Days with multiple speeches: {(speeches_per_day >= 2).sum()}")
    print(f"   4. Institution breakdown shown in detail above")

    print(f"\nNext steps:")
    print(f"   1. Review the distributions and value counts above")
    print(f"   2. Decide on aggregation strategy based on speech frequency")
    print(f"   3. Consider whether to weight by institution or treat equally")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING  Interrupted by user")
    except Exception as e:
        print(f"\n\nERROR Error: {e}")
        import traceback
        traceback.print_exc()
