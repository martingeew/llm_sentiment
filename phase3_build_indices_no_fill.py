"""
Phase 3: Build Daily Time Series Indices (No Forward Fill)
Central Bank Communication Sentiment Analysis

Creates daily frequency indices for Fed and ECB separately.
- Aggregation: Mean when multiple speeches on same date
- Gap filling: NONE - only dates with actual speeches
- Market impact: Diffusion index (0-100, where 50=neutral)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import Config


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


def calculate_diffusion_index(values):
    """
    Calculate diffusion index for market impact variables.

    Formula: (% rise) + (0.5 * % neutral)

    Scale:
    - 100 = all 'rise'
    - 50 = neutral/mixed
    - 0 = all 'fall'

    Args:
        values: Series with 'rise', 'fall', or 'neutral' values

    Returns:
        Diffusion index (0-100)
    """
    if len(values) == 0:
        return np.nan

    # Count occurrences
    counts = values.value_counts()
    total = len(values)

    # Get percentages (0-1 scale)
    pct_rise = counts.get('rise', 0) / total
    pct_neutral = counts.get('neutral', 0) / total

    # Calculate diffusion index (0-100 scale)
    diffusion = (pct_rise + 0.5 * pct_neutral) * 100

    return diffusion


def aggregate_daily_scores(df, institution):
    """
    Aggregate speeches to daily frequency for one institution.

    Args:
        df: DataFrame with all speeches
        institution: 'United States' or 'Euro area'

    Returns:
        DataFrame with daily indices (only dates with speeches)
    """
    # Filter to institution
    inst_df = df[df['country'] == institution].copy()

    print(f"\n   Processing {institution}:")
    print(f"   - Total speeches: {len(inst_df)}")
    print(f"   - Date range: {inst_df['date'].min().date()} to {inst_df['date'].max().date()}")

    # Define continuous metrics (calculate mean)
    continuous_metrics = [
        'hawkish_dovish_score',
        'uncertainty',
        'forward_guidance_strength',
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international'
    ]

    # Define market impact variables (calculate diffusion index)
    market_metrics = [
        'market_impact_stocks',
        'market_impact_bonds',
        'market_impact_currency'
    ]

    # Group by date and aggregate
    daily_continuous = inst_df.groupby('date')[continuous_metrics].mean()

    # Calculate diffusion indices for market impact
    daily_market = inst_df.groupby('date')[market_metrics].agg(calculate_diffusion_index)

    # Rename market columns to indicate they're diffusion indices
    daily_market.columns = [col.replace('market_impact_', '') + '_diffusion_index'
                            for col in daily_market.columns]

    # Count speeches per day
    daily_counts = inst_df.groupby('date').size().rename('speech_count')

    # Combine all metrics
    daily_indices = pd.concat([daily_continuous, daily_market, daily_counts], axis=1)

    # Reset index to make date a column
    daily_indices = daily_indices.reset_index()

    print(f"   - Unique dates with speeches: {len(daily_indices)}")
    print(f"   - Days with multiple speeches: {(daily_counts > 1).sum()}")

    return daily_indices


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("PHASE 3: BUILD DAILY INDICES (NO FORWARD FILL)".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Prepared Data
    # ============================================================
    print_section_header("STEP 1: LOAD PREPARED DATA")

    data_file = Config.RESULTS_DIR / "phase3_prepared_data.csv"

    if not data_file.exists():
        print(f"\nERROR  File not found: {data_file}")
        print(f"   Please run phase3_data_prep.py first")
        return

    df = pd.read_csv(data_file, parse_dates=['date'])

    print(f"\nOK Loaded prepared data")
    print(f"   File: {data_file.name}")
    print(f"   Speeches: {len(df)}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Institutions: {df['country'].unique().tolist()}")

    # ============================================================
    # STEP 2: Build Indices for Each Institution (No Forward Fill)
    # ============================================================
    print_section_header("STEP 2: BUILD DAILY INDICES BY INSTITUTION")
    print("\n   NOTE: Only creating indices for dates with actual speeches")
    print("   No forward filling of gaps")

    institutions = {
        'United States': 'fed',
        'Euro area': 'ecb'
    }

    for institution_name, file_prefix in institutions.items():
        print(f"\n--- {institution_name} ---")

        # Aggregate to daily frequency (no forward fill)
        daily_indices = aggregate_daily_scores(df, institution_name)

        # Save to CSV
        output_file = Config.RESULTS_DIR / f"{file_prefix}_daily_indices_no_fill.csv"
        daily_indices.to_csv(output_file, index=False)

        print(f"\nOK Saved indices")
        print(f"   File: {output_file.name}")
        print(f"   Rows: {len(daily_indices)}")
        print(f"   Columns: {len(daily_indices.columns)}")

    # ============================================================
    # STEP 3: Summary Statistics
    # ============================================================
    print_section_header("STEP 3: INDEX SUMMARY STATISTICS")

    for institution_name, file_prefix in institutions.items():
        output_file = Config.RESULTS_DIR / f"{file_prefix}_daily_indices_no_fill.csv"
        indices_df = pd.read_csv(output_file, parse_dates=['date'])

        print(f"\n--- {institution_name} ({file_prefix.upper()}) ---")
        print(f"\n   Date range: {indices_df['date'].min().date()} to {indices_df['date'].max().date()}")
        print(f"   Total dates with speeches: {len(indices_df)}")
        print(f"   Total speeches: {indices_df['speech_count'].sum():.0f}")
        print(f"   Avg speeches per date: {indices_df['speech_count'].mean():.2f}")

        print(f"\n   Index Statistics (mean ± std):")

        # Continuous metrics
        metrics = {
            'Hawkish/Dovish': 'hawkish_dovish_score',
            'Uncertainty': 'uncertainty',
            'Forward Guidance': 'forward_guidance_strength',
            'Inflation Topic': 'topic_inflation',
            'Growth Topic': 'topic_growth'
        }

        for label, col in metrics.items():
            mean_val = indices_df[col].mean()
            std_val = indices_df[col].std()
            print(f"      {label:<20}: {mean_val:>6.1f} ± {std_val:>5.1f}")

        # Diffusion indices
        print(f"\n   Diffusion Indices (mean, range 0-100, 50=neutral):")
        diffusion_cols = ['stocks_diffusion_index', 'bonds_diffusion_index', 'currency_diffusion_index']

        for col in diffusion_cols:
            if col in indices_df.columns:
                mean_val = indices_df[col].mean()
                label = col.replace('_diffusion_index', '').capitalize()
                print(f"      {label:<20}: {mean_val:>6.1f}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("DAILY INDICES BUILD COMPLETE (NO FORWARD FILL)".center(70))
    print("=" * 70)

    print(f"\nOutput files created:")
    for institution_name, file_prefix in institutions.items():
        output_file = Config.RESULTS_DIR / f"{file_prefix}_daily_indices_no_fill.csv"
        print(f"   - {output_file}")

    print(f"\nIndex features:")
    print(f"   - Frequency: Daily (only dates with speeches)")
    print(f"   - Multiple speeches: Averaged within same day")
    print(f"   - Gap filling: NONE - sparse time series")
    print(f"   - Market impact: Diffusion index (0-100, 50=neutral)")

    print(f"\nNext steps:")
    print(f"   1. Load indices for visualization:")
    print(f"      fed = pd.read_csv('data/results/fed_daily_indices_no_fill.csv', parse_dates=['date'])")
    print(f"      ecb = pd.read_csv('data/results/ecb_daily_indices_no_fill.csv', parse_dates=['date'])")
    print(f"   2. Create bar chart visualizations")
    print(f"   3. Analyze speech-level patterns")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
