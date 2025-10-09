import importlib
import input_merge_timeseries
importlib.reload(input_merge_timeseries)
from input_merge_timeseries import *

import os
import glob
import pandas as pd
from pathlib import Path
import re
from io import StringIO

def load_timeseries(
    master_dir: str,
    target_filename: str,
    start_tag: str = '<BeginTimeSerie>',
    end_tag:   str = '<EndTimeSerie>',
    encoding:  str = 'utf-8',
) -> pd.DataFrame:
    dfs = []
    for fp in Path(master_dir).rglob('*'):
        if not fp.is_file() or fp.name != target_filename:
            continue

        # Read with UTF-8, fallback to Latin-1
        try:
            text = fp.read_text(encoding=encoding)
        except UnicodeDecodeError:
            text = fp.read_text(encoding='latin-1', errors='ignore')

        lines = text.splitlines(keepends=True)
        starts = [i for i, L in enumerate(lines) if start_tag in L]
        ends   = [i for i, L in enumerate(lines) if end_tag   in L]
        if len(starts) != len(ends):
            raise ValueError(f"Tag mismatch in {fp}")

        for s, e in zip(starts, ends):
            # Find header line just above the block
            hdr = s - 1
            while hdr >= 0 and not lines[hdr].strip():
                hdr -= 1

            cols       = re.findall(r'\w+', lines[hdr])
            block_text = ''.join(lines[s+1 : e])
            df_block   = pd.read_csv(
                StringIO(block_text),
                sep=r'\s+',
                header=None,
                names=cols,
            )
            dfs.append(df_block)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def merge_and_save(
    master_dir: str,
    target_filename: str,
    output_csv: str,
    **load_kwargs
) -> pd.DataFrame:
    df = load_timeseries(master_dir, target_filename, **load_kwargs)
    if df.empty:
        print("No data loaded—check your filename and directory.")
        return df

    # Drop stray 'Seconds', rename time parts
    df = (
        df
        .drop(columns=['Seconds'], errors='ignore')
        .rename(columns={
            'YY':'year','MM':'month','DD':'day',
            'hh':'hour','mm':'minute','ss':'second'
        })
    )

    # Build DateTimeIndex
    df['timestamp'] = pd.to_datetime(df[
        ['year','month','day','hour','minute','second']
    ])
    df = df.set_index('timestamp').sort_index()

    # Drop exact-duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Clean up leftover columns
    df = df.drop(columns=['year','month','day','hour','minute','second'])
    

    df.to_csv(output_csv)
    print(f"Saved {len(df)} rows to {output_csv}")

merge_and_save(master_dir, filename, output_csv=output_csv)