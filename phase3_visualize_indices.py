"""
Phase 3: Visualize Daily Time Series Indices
Central Bank Communication Sentiment Analysis

Creates clean line charts comparing Fed and ECB indices over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import Config


def create_timeseries_chart(fed_df, ecb_df, variables, output_filename, suptitle, reports_dir):
    """
    Create a set of time series charts comparing Fed and ECB.

    Args:
        fed_df: DataFrame with Fed daily indices
        ecb_df: DataFrame with ECB daily indices
        variables: List of variable names to plot
        output_filename: Name of output PNG file
        suptitle: Super title for the figure
        reports_dir: Path to reports directory
    """
    # Calculate subplot grid (2 columns, variable rows)
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // 2  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Define colors
    fed_color = '#1f77b4'  # Blue
    ecb_color = '#ff7f0e'  # Orange

    # Plot each variable
    for idx, var in enumerate(variables):
        ax = axes[idx]

        # Add horizontal reference line for neutral values
        neutral_value = get_neutral_value(var)
        if neutral_value is not None:
            ax.axhline(y=neutral_value, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

        # Plot Fed
        ax.plot(fed_df['date'], fed_df[var],
                color=fed_color, linewidth=1.5, label='Fed', alpha=0.8, zorder=2)

        # Plot ECB
        ax.plot(ecb_df['date'], ecb_df[var],
                color=ecb_color, linewidth=1.5, label='ECB', alpha=0.8, zorder=2)

        # Format subplot with scale information in title
        title_with_scale = format_title_with_scale(var)
        ax.set_title(title_with_scale, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='upper left', frameon=True, fontsize=10)

    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = reports_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_filename}")

    plt.close()


def format_title(var_name):
    """
    Format variable name into a nice title.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Formatted title string
    """
    # Replace underscores with spaces and title case
    title = var_name.replace('_', ' ').title()

    # Special formatting for specific terms
    replacements = {
        'Hawkish Dovish': 'Hawkish/Dovish',
        'Diffusion Index': 'Diffusion',
        'Topic ': ''
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    return title


def format_title_with_scale(var_name):
    """
    Format variable name into a title with scale information.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Formatted title string with scale info
    """
    base_title = format_title(var_name)

    # Add scale information for specific metrics
    if var_name == 'hawkish_dovish_score':
        return f"{base_title} (0=neutral, +100=hawkish, -100=dovish)"
    elif var_name in ['stocks_diffusion_index', 'bonds_diffusion_index', 'currency_diffusion_index']:
        return f"{base_title} (50=neutral, 100=rise, 0=fall)"
    elif var_name == 'uncertainty':
        return f"{base_title} (0=confident, 100=uncertain)"
    elif var_name == 'forward_guidance_strength':
        return f"{base_title} (0=none, 100=explicit commitment)"
    elif var_name.startswith('topic_'):
        return f"{base_title} (0-100, higher=more emphasis)"
    elif var_name == 'speech_count':
        return f"{base_title}"
    else:
        return base_title


def get_neutral_value(var_name):
    """
    Get the neutral reference value for a variable (if applicable).

    Args:
        var_name: Variable name from dataframe

    Returns:
        Neutral value (float) or None if not applicable
    """
    if var_name == 'hawkish_dovish_score':
        return 0.0
    elif var_name in ['stocks_diffusion_index', 'bonds_diffusion_index', 'currency_diffusion_index']:
        return 50.0
    else:
        return None


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("PHASE 3: VISUALIZE DAILY TIME SERIES INDICES".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create reports directory if it doesn't exist
    reports_dir = Config.PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ============================================================
    # Load Data
    # ============================================================
    print("\nLoading indices...")

    fed_file = Config.RESULTS_DIR / "fed_daily_indices.csv"
    ecb_file = Config.RESULTS_DIR / "ecb_daily_indices.csv"

    fed_df = pd.read_csv(fed_file, parse_dates=['date'])
    ecb_df = pd.read_csv(ecb_file, parse_dates=['date'])

    print(f"   Fed: {len(fed_df)} days ({fed_df['date'].min().date()} to {fed_df['date'].max().date()})")
    print(f"   ECB: {len(ecb_df)} days ({ecb_df['date'].min().date()} to {ecb_df['date'].max().date()})")

    # Determine longest date range
    min_date = min(fed_df['date'].min(), ecb_df['date'].min())
    max_date = max(fed_df['date'].max(), ecb_df['date'].max())

    print(f"\n   Using date range: {min_date.date()} to {max_date.date()}")

    # Extend Fed data to match ECB range if needed
    if fed_df['date'].max() < max_date:
        # Create missing dates
        missing_dates = pd.date_range(
            start=fed_df['date'].max() + pd.Timedelta(days=1),
            end=max_date,
            freq='D'
        )
        # Get last row values and forward fill
        last_values = fed_df.iloc[-1].copy()
        missing_rows = pd.DataFrame([last_values] * len(missing_dates))
        missing_rows['date'] = missing_dates
        missing_rows['speech_count'] = 0

        fed_df = pd.concat([fed_df, missing_rows], ignore_index=True)
        print(f"   Extended Fed data by {len(missing_dates)} days (forward fill)")

    # ============================================================
    # Create Visualizations
    # ============================================================
    print("\nCreating visualizations...")

    # Set 1: Policy Metrics
    policy_vars = [
        'hawkish_dovish_score',
        'uncertainty',
        'forward_guidance_strength',
        'speech_count'
    ]

    create_timeseries_chart(
        fed_df, ecb_df,
        variables=policy_vars,
        output_filename='policy_metrics_timeseries.png',
        suptitle='Policy Metrics - Fed vs ECB',
        reports_dir=reports_dir
    )

    # Set 2: Topic Indices
    topic_vars = [
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international'
    ]

    create_timeseries_chart(
        fed_df, ecb_df,
        variables=topic_vars,
        output_filename='topic_indices_timeseries.png',
        suptitle='Topic Indices - Fed vs ECB',
        reports_dir=reports_dir
    )

    # Set 3: Market Impact
    market_vars = [
        'stocks_diffusion_index',
        'bonds_diffusion_index',
        'currency_diffusion_index'
    ]

    create_timeseries_chart(
        fed_df, ecb_df,
        variables=market_vars,
        output_filename='market_impact_timeseries.png',
        suptitle='Market Impact Diffusion Indices - Fed vs ECB',
        reports_dir=reports_dir
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE".center(70))
    print("=" * 70)

    print(f"\nOutput files (saved to {reports_dir}):")
    print(f"   1. policy_metrics_timeseries.png")
    print(f"   2. topic_indices_timeseries.png")
    print(f"   3. market_impact_timeseries.png")

    print(f"\nChart features:")
    print(f"   - Style: Seaborn whitegrid")
    print(f"   - Layout: 2 columns per set")
    print(f"   - Lines: Fed (blue) vs ECB (orange)")
    print(f"   - Resolution: 300 DPI")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
