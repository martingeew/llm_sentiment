"""
Phase 3: Visualize Daily Time Series Indices (Calendar Heatmaps)
Central Bank Communication Sentiment Analysis

Creates calendar heatmap visualizations using dayplot for Fed and ECB separately.
- Uses data without forward filling (sparse - only dates with speeches)
- Shows full calendar years (2022, 2023)
- Diverging colormaps for metrics with neutral values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import dayplot as dp
from pathlib import Path
from src.config import Config


def get_colormap_settings(var_name):
    """
    Get colormap and vcenter settings for a variable.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Tuple of (cmap, vcenter, vmin, vmax)
    """
    # Diverging colormaps for metrics with neutral values
    if var_name == 'hawkish_dovish_score':
        return ('coolwarm', 0, -100, 100)
    elif var_name in ['stocks_diffusion_index', 'bonds_diffusion_index', 'currency_diffusion_index']:
        return ('coolwarm', 50, 0, 100)
    elif var_name.startswith('topic_'):
        # YlOrRd colormap for topics (yellow=0, orange=mid, red=high)
        # Set vmin=0, vmax=100 to ensure full colormap range in legend
        return ('YlOrRd', None, 0, 100)
    elif var_name == 'speech_count':
        # Speech count: 1-6 scale (0 = grey for no speeches)
        # Use darker subset of Greens colormap (0.2 to 1.0 instead of 0.0 to 1.0)
        # This makes value=1 appear darker without affecting legend labels
        base_cmap = plt.cm.get_cmap('Greens')
        colors = base_cmap(np.linspace(0.3, 1.0, 256))  # Use 0.3-1.0 range for darker colors
        dark_greens = mcolors.LinearSegmentedColormap.from_list('DarkGreens', colors)
        return (dark_greens, None, 1, 6)
    else:
        # Greens colormap for other metrics (uncertainty, forward_guidance)
        return ('Greens', None, 0, 100)


def format_metric_title(var_name):
    """
    Format variable name into a readable title.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Formatted title string
    """
    # Replace underscores with spaces and title case
    title = var_name.replace('_', ' ').title()

    # Special formatting
    replacements = {
        'Hawkish Dovish': 'Hawkish/Dovish',
        'Diffusion Index': 'Diffusion',
        'Topic ': '',
        'Score': ''
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    return title.strip()


def create_calendar_heatmaps(df, institution_name, variables, output_filename, suptitle, reports_dir):
    """
    Create calendar heatmap visualizations for one institution.

    Each metric gets 2 rows (one for 2022, one for 2023).

    Args:
        df: DataFrame with daily indices (no forward fill)
        institution_name: 'Fed' or 'ECB'
        variables: List of variable names to plot
        output_filename: Name of output PNG file
        suptitle: Super title for the figure
        reports_dir: Path to reports directory
    """
    n_metrics = len(variables)
    years = [2022, 2023]
    n_rows = n_metrics * len(years)  # Each metric gets 2 rows (one per year)

    # Create figure with vertical stacking
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(16, 2.5 * n_rows),
        gridspec_kw={'hspace': 0.4}
    )

    # Ensure axes is always an array
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.998)

    # Plot each metric for each year
    ax_idx = 0
    for var in variables:
        metric_title = format_metric_title(var)
        cmap, vcenter, vmin, vmax = get_colormap_settings(var)

        for year in years:
            ax = axes[ax_idx]

            # Filter data for this year
            year_df = df[df['date'].dt.year == year].copy()

            # Create calendar heatmap
            if len(year_df) > 0:
                # Prepare data
                dates = year_df['date'].tolist()
                values = year_df[var].tolist()

                # For topic variables, replace 0 with small epsilon to avoid color_for_none
                # (dayplot treats 0 as "no contribution" and colors it grey)
                if var.startswith('topic_'):
                    values = [0.01 if v == 0 else v for v in values]

                # Use vmin/vmax as-is
                vmin_adjusted = vmin
                vmax_adjusted = vmax

                # Determine legend bins based on variable
                if var == 'speech_count':
                    legend_bins_count = 6  # Show 1, 2, 3, 4, 5, 6
                else:
                    legend_bins_count = 11  # Standard bins for other metrics

                # Set date range for full calendar year
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"

                # Create calendar with appropriate settings
                if vcenter is not None:
                    # Diverging colormap
                    # Note: color_for_none doesn't work with diverging colormaps in dayplot
                    # So we extend the full date range and fill missing dates with vcenter value
                    # to make them appear neutral (white/gray in coolwarm)

                    # Create full date range for the year
                    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

                    # Create a series with all dates, filling missing with vcenter
                    date_series = pd.Series(values, index=dates)
                    full_series = date_series.reindex(full_dates, fill_value=vcenter)

                    dp.calendar(
                        dates=full_series.index.tolist(),
                        values=full_series.tolist(),
                        start_date=start_date,
                        end_date=end_date,
                        cmap=cmap,
                        vcenter=vcenter,
                        vmin=vmin_adjusted,
                        vmax=vmax_adjusted,
                        edgecolor='white',
                        edgewidth=0.5,
                        legend=True,
                        legend_bins=legend_bins_count,
                        legend_labels='auto',
                        ax=ax
                    )
                else:
                    # Regular colormap - color_for_none works fine here
                    dp.calendar(
                        dates=dates,
                        values=values,
                        start_date=start_date,
                        end_date=end_date,
                        cmap=cmap,
                        vmin=vmin_adjusted,
                        vmax=vmax_adjusted,
                        color_for_none='#e8e8e8',  # Light gray for days with no speeches
                        edgecolor='white',
                        edgewidth=0.5,
                        legend=True,
                        legend_bins=legend_bins_count,
                        legend_labels='auto',
                        ax=ax
                    )

            # Add year label on the left
            ax.text(
                -4, 3.5,
                str(year),
                size=16,
                rotation=90,
                color="#666",
                va="center",
                ha="center",
                weight='bold'
            )

            # Add metric title on the right (only for first year of each metric)
            if year == years[0]:
                ax.text(
                    1.02, 0.5,
                    metric_title,
                    transform=ax.transAxes,
                    size=14,
                    va='center',
                    ha='left',
                    weight='bold',
                    color='#333'
                )

            ax_idx += 1

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    # Save figure
    output_path = reports_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_filename}")

    plt.close()


def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("PHASE 3: CALENDAR HEATMAP VISUALIZATIONS".center(70))
    print("Central Bank Communication Sentiment Analysis".center(70))
    print("=" * 70)

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

    print(f"   Fed: {len(fed_df)} dates with speeches")
    print(f"   ECB: {len(ecb_df)} dates with speeches")

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
    print("\nCreating Fed calendar heatmaps...")

    create_calendar_heatmaps(
        fed_df, 'Fed',
        variables=policy_vars,
        output_filename='fed_policy_metrics_calendar.png',
        suptitle='Fed Policy Metrics - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    create_calendar_heatmaps(
        fed_df, 'Fed',
        variables=topic_vars,
        output_filename='fed_topic_indices_calendar.png',
        suptitle='Fed Topic Indices - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    create_calendar_heatmaps(
        fed_df, 'Fed',
        variables=market_vars,
        output_filename='fed_market_impact_calendar.png',
        suptitle='Fed Market Impact - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    # ============================================================
    # Create ECB Visualizations
    # ============================================================
    print("\nCreating ECB calendar heatmaps...")

    create_calendar_heatmaps(
        ecb_df, 'ECB',
        variables=policy_vars,
        output_filename='ecb_policy_metrics_calendar.png',
        suptitle='ECB Policy Metrics - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    create_calendar_heatmaps(
        ecb_df, 'ECB',
        variables=topic_vars,
        output_filename='ecb_topic_indices_calendar.png',
        suptitle='ECB Topic Indices - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    create_calendar_heatmaps(
        ecb_df, 'ECB',
        variables=market_vars,
        output_filename='ecb_market_impact_calendar.png',
        suptitle='ECB Market Impact - Calendar Heatmaps (2022-2023)',
        reports_dir=reports_dir
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("CALENDAR HEATMAP VISUALIZATION COMPLETE".center(70))
    print("=" * 70)

    print(f"\nOutput files (saved to {reports_dir}):")
    print(f"\n   Fed:")
    print(f"      1. fed_policy_metrics_calendar.png (8 calendars: 4 metrics × 2 years)")
    print(f"      2. fed_topic_indices_calendar.png (10 calendars: 5 metrics × 2 years)")
    print(f"      3. fed_market_impact_calendar.png (6 calendars: 3 metrics × 2 years)")
    print(f"\n   ECB:")
    print(f"      4. ecb_policy_metrics_calendar.png (8 calendars: 4 metrics × 2 years)")
    print(f"      5. ecb_topic_indices_calendar.png (10 calendars: 5 metrics × 2 years)")
    print(f"      6. ecb_market_impact_calendar.png (6 calendars: 3 metrics × 2 years)")

    print(f"\nChart features:")
    print(f"   - Layout: Vertical stack (each metric shows 2022 and 2023)")
    print(f"   - Calendar range: Full calendar years (Jan 1 - Dec 31)")
    print(f"   - Sparse data: Only dates with actual speeches are colored")
    print(f"   - Diverging colormap (coolwarm):")
    print(f"      - Hawkish/Dovish (centered on 0)")
    print(f"      - Market Diffusion Indices (centered on 50)")
    print(f"   - Regular colormap (Greens): All other metrics")
    print(f"   - Resolution: 300 DPI")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
