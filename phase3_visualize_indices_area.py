"""
Phase 3: Visualize Daily Time Series Indices (Area Plots)
Central Bank Communication Sentiment Analysis

Creates area-filled time series charts for Fed and ECB separately.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from pathlib import Path
from src.config import Config


def format_title_with_scale(var_name):
    """
    Format variable name into a title with scale information.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Formatted title string with scale info
    """
    # Replace underscores with spaces and title case
    title = var_name.replace('_', ' ').title()

    # Special formatting
    replacements = {
        'Hawkish Dovish': 'Hawkish/Dovish',
        'Topic ': ''
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    # Add scale information
    if var_name == 'hawkish_dovish_score':
        return f"{title} (0=neutral, +100=hawkish, -100=dovish)"
    elif var_name == 'uncertainty':
        return f"{title} (0=confident, 100=uncertain)"
    elif var_name == 'forward_guidance_strength':
        return f"{title} (0=none, 100=explicit commitment)"
    elif var_name.startswith('topic_'):
        return f"{title} (0-100, higher=more emphasis)"
    else:
        return title


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
    else:
        return None


def create_area_timeseries(df, institution_name, variables, output_filename, suptitle, reports_dir):
    """
    Create area-filled time series charts for one institution.

    Args:
        df: DataFrame with daily indices (forward-filled)
        institution_name: 'Fed' or 'ECB'
        variables: List of variable names to plot
        output_filename: Name of output PNG file
        suptitle: Super title for the figure
        reports_dir: Path to reports directory
    """
    # Calculate subplot grid (2 columns)
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

    # Define color based on institution
    if institution_name == 'Fed':
        fill_color = '#1f77b4'  # Blue
    else:  # ECB
        fill_color = '#ff7f0e'  # Orange

    # Plot each variable
    for idx, var in enumerate(variables):
        ax = axes[idx]

        # Special handling for hawkish_dovish_score (diverging colors)
        if var == 'hawkish_dovish_score':
            # Use matplotlib for conditional coloring
            dates = df['date']
            values = df[var]

            # Get default seaborn colors
            palette = sns.color_palette()
            blue = palette[0]  # Default seaborn blue
            red = palette[3]   # Default seaborn red

            # Fill positive values (hawkish) in red
            ax.fill_between(dates, 0, values, where=(values >= 0),
                           color=red, alpha=0.3, interpolate=True, edgecolor='none')
            # Fill negative values (dovish) in blue
            ax.fill_between(dates, 0, values, where=(values < 0),
                           color=blue, alpha=0.3, interpolate=True, edgecolor='none')

            # Plot neutral-colored line on top
            ax.plot(dates, values, color='#888888', linewidth=2)
        else:
            # Regular single-color area plot
            plot = (
                so.Plot(df, x="date", y=var)
                .add(so.Area(color=fill_color, alpha=0.3, edgewidth=0))
                .add(so.Line(color=fill_color, linewidth=2))
                .on(ax)
                .plot()
            )

        # Add horizontal reference line for neutral values
        neutral_value = get_neutral_value(var)
        if neutral_value is not None:
            ax.axhline(y=neutral_value, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=3)

        # Format subplot
        title_with_scale = format_title_with_scale(var)
        ax.set_title(title_with_scale, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Set y-axis limits
        if var == 'hawkish_dovish_score':
            ax.set_ylim(-100, 100)
        else:
            ax.set_ylim(0, 100)

        # Remove all spines
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)

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


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("PHASE 3: AREA-FILLED TIME SERIES VISUALIZATIONS".center(70))
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
    print("\nLoading indices (forward-filled)...")

    fed_file = Config.RESULTS_DIR / "fed_daily_indices.csv"
    ecb_file = Config.RESULTS_DIR / "ecb_daily_indices.csv"

    fed_df = pd.read_csv(fed_file, parse_dates=['date'])
    ecb_df = pd.read_csv(ecb_file, parse_dates=['date'])

    print(f"   Fed: {len(fed_df)} days ({fed_df['date'].min().date()} to {fed_df['date'].max().date()})")
    print(f"   ECB: {len(ecb_df)} days ({ecb_df['date'].min().date()} to {ecb_df['date'].max().date()})")

    # ============================================================
    # Define Variable Sets
    # ============================================================

    # Policy Metrics (excluding speech_count)
    policy_vars = [
        'hawkish_dovish_score',
        'uncertainty',
        'forward_guidance_strength'
    ]

    # Topic Indices
    topic_vars = [
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international'
    ]

    # ============================================================
    # Create Fed Visualizations
    # ============================================================
    print("\nCreating Fed area plots...")

    create_area_timeseries(
        fed_df,
        'Fed',
        variables=policy_vars,
        output_filename='fed_policy_metrics_area.png',
        suptitle='Fed Policy Metrics - Time Series',
        reports_dir=reports_dir
    )

    create_area_timeseries(
        fed_df,
        'Fed',
        variables=topic_vars,
        output_filename='fed_topic_indices_area.png',
        suptitle='Fed Topic Indices - Time Series',
        reports_dir=reports_dir
    )

    # ============================================================
    # Create ECB Visualizations
    # ============================================================
    print("\nCreating ECB area plots...")

    create_area_timeseries(
        ecb_df,
        'ECB',
        variables=policy_vars,
        output_filename='ecb_policy_metrics_area.png',
        suptitle='ECB Policy Metrics - Time Series',
        reports_dir=reports_dir
    )

    create_area_timeseries(
        ecb_df,
        'ECB',
        variables=topic_vars,
        output_filename='ecb_topic_indices_area.png',
        suptitle='ECB Topic Indices - Time Series',
        reports_dir=reports_dir
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("AREA PLOT VISUALIZATION COMPLETE".center(70))
    print("=" * 70)

    print(f"\nOutput files (saved to {reports_dir}):")
    print(f"\n   Fed:")
    print(f"      1. fed_policy_metrics_area.png (3 metrics)")
    print(f"      2. fed_topic_indices_area.png (5 topics)")
    print(f"\n   ECB:")
    print(f"      3. ecb_policy_metrics_area.png (3 metrics)")
    print(f"      4. ecb_topic_indices_area.png (5 topics)")

    print(f"\nChart features:")
    print(f"   - Style: Area-filled time series")
    print(f"   - Layout: 2 columns per set")
    print(f"   - Data: Forward-filled (continuous)")
    print(f"   - Spines: Removed")
    print(f"   - Resolution: 300 DPI")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
