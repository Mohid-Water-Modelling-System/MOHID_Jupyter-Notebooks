#!/usr/bin/env python3

import importlib
import input_plot_timeseries
importlib.reload(input_plot_timeseries)
from input_plot_timeseries import *

import os
import pandas as pd

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_dataframe(path):
    """Load a CSV file with the first column as DateTime index."""
    if not path:
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df

def plot_time_series(df1, var1, df2, var2):
    """Plot up to two time series with optional secondary y-axis."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = None

    # Plot first series
    style1 = '-' if mode_var1 == 'lines' else '--'
    marker1 = 'o' if 'markers' in mode_var1 else ''
    ax1.plot(df1.index, df1[var1],
             color=color_var1, linestyle=style1, marker=marker1,
             label=legend_1)
    ax1.set_ylabel(axis_label_1, color=color_var1)
    ax1.tick_params(axis='y', labelcolor=color_var1)

    # Decide whether to share axis or create twin y-axis
    shared_axis = (var2 == var1)

    # Plot second series
    if df2 is not None and var2:
        style2 = '-' if mode_var2 == 'lines' else '--'
        marker2 = 'o' if 'markers' in mode_var2 else ''
        if shared_axis:
            # Draw on the same axis
            ln2 = ax1.plot(df2.index, df2[var2],
                           color=color_var2, linestyle=style2, marker=marker2,
                           label=legend_2)
        else:
            # Create a second y-axis
            ax2 = ax1.twinx()
            ln2 = ax2.plot(df2.index, df2[var2],
                           color=color_var2, linestyle=style2, marker=marker2,
                           label=legend_2)
            ax2.set_ylabel(axis_label_2, color=color_var2)
            ax2.tick_params(axis='y', labelcolor=color_var2)

    # Format x-axis ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval= days_between))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    if ax2:
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lab2
    ax1.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.tight_layout()
    return fig

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df1 = load_dataframe(csv_file_1)
    df2 = load_dataframe(csv_file_2)

    # Build and save figure
    fig = plot_time_series(df1, var1, df2, var2)
    out_path = os.path.join(output_dir, output_file)
    fig.savefig(out_path, dpi= dpi)
    plt.show()
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()