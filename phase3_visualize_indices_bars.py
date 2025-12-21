"""
Phase 3: Visualize Daily Time Series Indices (Bar Charts)
Central Bank Communication Sentiment Analysis

Creates bar chart visualizations for Fed and ECB separately.
- Diverging bars for metrics with neutral values
- Standard bars for other metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from src.config import Config


def create_bar_charts(df, institution_name, variables, output_filename, suptitle, reports_dir):
    """
    Create a set of bar charts for one institution.

    Args:
        df: DataFrame with daily indices (no forward fill)
        institution_name: 'Fed' or 'ECB'
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

    # Plot each variable
    for idx, var in enumerate(variables):
        ax = axes[idx]

        # Get data
        dates = df['date']
        values = df[var]

        # Determine if this is a diverging chart
        neutral_value = get_neutral_value(var)

        if neutral_value is not None:
            # Diverging bar chart
            plot_diverging_bars(ax, dates, values, var, neutral_value)
        else:
            # Standard bar chart
            plot_standard_bars(ax, dates, values, var, institution_name)

        # Format subplot with scale information in title
        title_with_scale = format_title_with_scale(var)
        ax.set_title(title_with_scale, fontsize=12, pad=10)

        # Special formatting for market impact indices
        if var in ['stocks_diffusion_index', 'bonds_diffusion_index', 'currency_diffusion_index']:
            # Set y-axis limits
            ax.set_ylim(0, 100)
            # Remove grid lines
            ax.grid(False)
            # Remove all spines
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(False)
        else:
            # Keep grid for other metrics
            ax.grid(True, alpha=0.3, axis='y')

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Rotate x-axis labels if many dates
        if len(dates) > 50:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        elif len(dates) > 30:
            ax.tick_params(axis='x', rotation=30, labelsize=9)

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


def plot_diverging_bars(ax, dates, values, var, neutral_value):
    """
    Plot diverging bars centered on neutral value.

    Args:
        ax: Matplotlib axis
        dates: Series of dates
        values: Series of values
        var: Variable name
        neutral_value: Neutral reference value (0 or 50)
    """
    # Determine colors based on variable type
    if var == 'hawkish_dovish_score':
        # Hawkish (positive) = blue, Dovish (negative) = red
        positive_color = '#2E86AB'
        negative_color = '#E63946'
    else:
        # Diffusion indices: Bullish (>50) = darker green, Bearish (<50) = darker red
        positive_color = '#048060'
        negative_color = '#C41E28'

    # Create color array based on values relative to neutral
    colors = [positive_color if v > neutral_value else negative_color for v in values]

    # For diverging bars, plot values relative to neutral
    # Bars extend FROM neutral_value
    bar_heights = values - neutral_value

    # Plot bars
    ax.bar(dates, bar_heights, bottom=neutral_value, color=colors, width=0.8, edgecolor='none')

    # Add horizontal reference line at neutral value
    ax.axhline(y=neutral_value, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=3)


def plot_standard_bars(ax, dates, values, var, institution_name):
    """
    Plot standard bars starting from 0.

    Args:
        ax: Matplotlib axis
        dates: Series of dates
        values: Series of values
        var: Variable name
        institution_name: 'Fed' or 'ECB'
    """
    # Use institution color
    if institution_name == 'Fed':
        color = '#1f77b4'  # Blue
    else:
        color = '#ff7f0e'  # Orange

    # Plot bars from 0
    ax.bar(dates, values, color=color, width=0.8, edgecolor='none', alpha=0.8)


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
    print("PHASE 3: VISUALIZE DAILY INDICES (BAR CHARTS)".center(70))
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
    print("\nLoading indices (no forward fill)...")

    fed_file = Config.RESULTS_DIR / "fed_daily_indices_no_fill.csv"
    ecb_file = Config.RESULTS_DIR / "ecb_daily_indices_no_fill.csv"

    fed_df = pd.read_csv(fed_file, parse_dates=['date'])
    ecb_df = pd.read_csv(ecb_file, parse_dates=['date'])

    print(f"   Fed: {len(fed_df)} dates with speeches ({fed_df['date'].min().date()} to {fed_df['date'].max().date()})")
    print(f"   ECB: {len(ecb_df)} dates with speeches ({ecb_df['date'].min().date()} to {ecb_df['date'].max().date()})")

    # ============================================================
    # Define Variable Sets
    # ============================================================

    # Set 1: Policy Metrics
    policy_vars = [
        'hawkish_dovish_score',
        'uncertainty',
        'forward_guidance_strength',
        'speech_count'
    ]

    # Set 2: Topic Indices
    topic_vars = [
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international'
    ]

    # Set 3: Market Impact
    market_vars = [
        'stocks_diffusion_index',
        'bonds_diffusion_index',
        'currency_diffusion_index'
    ]

    # ============================================================
    # Create Fed Visualizations
    # ============================================================
    print("\nCreating Fed visualizations...")

    create_bar_charts(
        fed_df, 'Fed',
        variables=policy_vars,
        output_filename='fed_policy_metrics_bars.png',
        suptitle='Fed Policy Metrics',
        reports_dir=reports_dir
    )

    create_bar_charts(
        fed_df, 'Fed',
        variables=topic_vars,
        output_filename='fed_topic_indices_bars.png',
        suptitle='Fed Topic Indices',
        reports_dir=reports_dir
    )

    create_bar_charts(
        fed_df, 'Fed',
        variables=market_vars,
        output_filename='fed_market_impact_bars.png',
        suptitle='Fed Market Impact Diffusion Indices',
        reports_dir=reports_dir
    )

    # ============================================================
    # Create ECB Visualizations
    # ============================================================
    print("\nCreating ECB visualizations...")

    create_bar_charts(
        ecb_df, 'ECB',
        variables=policy_vars,
        output_filename='ecb_policy_metrics_bars.png',
        suptitle='ECB Policy Metrics',
        reports_dir=reports_dir
    )

    create_bar_charts(
        ecb_df, 'ECB',
        variables=topic_vars,
        output_filename='ecb_topic_indices_bars.png',
        suptitle='ECB Topic Indices',
        reports_dir=reports_dir
    )

    create_bar_charts(
        ecb_df, 'ECB',
        variables=market_vars,
        output_filename='ecb_market_impact_bars.png',
        suptitle='ECB Market Impact Diffusion Indices',
        reports_dir=reports_dir
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("BAR CHART VISUALIZATION COMPLETE".center(70))
    print("=" * 70)

    print(f"\nOutput files (saved to {reports_dir}):")
    print(f"\n   Fed:")
    print(f"      1. fed_policy_metrics_bars.png")
    print(f"      2. fed_topic_indices_bars.png")
    print(f"      3. fed_market_impact_bars.png")
    print(f"\n   ECB:")
    print(f"      4. ecb_policy_metrics_bars.png")
    print(f"      5. ecb_topic_indices_bars.png")
    print(f"      6. ecb_market_impact_bars.png")

    print(f"\nChart features:")
    print(f"   - Style: Seaborn whitegrid")
    print(f"   - Layout: 2 columns per set")
    print(f"   - Diverging bars: Hawkish/Dovish (centered on 0), Market Impact (centered on 50)")
    print(f"   - Standard bars: Topics, Uncertainty, Forward Guidance, Speech Count")
    print(f"   - Colors:")
    print(f"      - Hawkish/Dovish: Blue (hawkish) / Red (dovish)")
    print(f"      - Market Impact: Green (bullish) / Red (bearish)")
    print(f"      - Standard: Blue (Fed) / Orange (ECB)")
    print(f"   - Resolution: 300 DPI")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
