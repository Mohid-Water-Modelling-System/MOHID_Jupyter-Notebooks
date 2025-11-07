#!/usr/bin/env python3
"""
Generate a true vertical cut (depth vs distance) animation along a path
defined by CSV waypoints. Every grid cell intersected between waypoints
is included via Bresenham’s algorithm—no fixed interpolation needed.
"""

import importlib
import Input_Plot_HDF5_Cut
importlib.reload(Input_Plot_HDF5_Cut)
from Input_Plot_HDF5_Cut import *

import os
import glob
import datetime
import numpy as np
import h5py
import csv
import geopy.distance
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def collect_hdf5_paths(root, h5file, sd, ed):
    """Gather HDF5 file paths in date folders between sd and ed."""
    paths = []
    for ent in os.scandir(root):
        if not ent.is_dir():
            continue
        try:
            day = datetime.datetime.strptime(ent.name.split('_')[0], "%Y%m%d").date()
        except ValueError:
            continue
        if sd <= day <= ed:
            for f in glob.glob(os.path.join(ent.path, h5file)):
                if os.path.isfile(f):
                    paths.append(f)
    return sorted(paths)


def index_by_time(hdf5_paths):
    """
    Build dict mapping each datetime → list of (file_path, time_key).
    """
    idx = {}
    for path in hdf5_paths:
        with h5py.File(path, "r") as h5f:
            for tkey in sorted(h5f["Time"].keys())[::skip_time]:
                y, m, d, H, M = h5f["Time"][tkey][:5]
                dt = datetime.datetime(int(y), int(m), int(d), int(H), int(M))
                idx.setdefault(dt, []).append((path, tkey))
    return idx


def mask_invalid(data3d, open3d, sentinel=None):
    """
    Mask out invalid points in a 3D array.

    Parameters
    ----------
    data3d : np.ndarray
        The 3D data to mask (shape nz×ny×nx).
    open3d : np.ndarray
        Numeric mask (same shape) where 0 means dry/invalid.
    sentinel : float or None
        Optional fill‐value to treat as NaN.

    Returns
    -------
    np.ndarray
        Copy of data3d with dry cells and sentinel values set to np.nan.
    """
    # mask dry cells
    masked = np.where(open3d == 0, np.nan, data3d)

    # mask exact sentinel if provided
    if sentinel is not None:
        masked = np.where(masked == sentinel, np.nan, masked)

    return masked


def bresenham(i0, j0, i1, j1):
    """
    Bresenham’s line algorithm between two grid indices.
    Returns list of (i,j) inclusive of endpoints.
    """
    dx, dy = abs(i1 - i0), abs(j1 - j0)
    sx, sy = (1 if i1 > i0 else -1), (1 if j1 > j0 else -1)
    err = dx - dy
    x, y = i0, j0
    cells = []
    while True:
        cells.append((x, y))
        if x == i1 and y == j1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def grid_location(lon, lat, lonc, latc):
    """
    Find nearest grid cell to (lon,lat).
    Brute‐force search; for large grids, consider KDTree.
    """
    dist = np.hypot(lonc - lon, latc - lat)
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    return int(i), int(j)


def get_path_indices(csv_file, lonc, latc):
    """
    Read lon/lat vertices from CSV, map to grid cells, and
    trace all cells along the path via Bresenham.
    Returns arrays: distance[m], grid_i, grid_j.
    """
    with open(csv_file, newline="") as f:
            
        df = pd.read_csv(
        csv_file,
        sep=',',
        header=None,
        names=['lon', 'lat'],
        engine='python'
    )
        
    lons = df["lon"].tolist()
    lats = df["lat"].tolist()
        
    verts = [grid_location(lo, la, lonc, latc) for lo, la in zip(lons, lats)]

    cells = []
    for k in range(len(verts) - 1):
        seg = bresenham(*verts[k], *verts[k + 1])
        if k > 0:
            seg = seg[1:]
        cells.extend(seg)

    grid_i = np.array([i for i, _ in cells], int)
    grid_j = np.array([j for _, j in cells], int)

    dist = [0.0]
    for n in range(1, len(grid_i)):
        lat0, lon0 = latc[grid_j[n - 1], grid_i[n - 1]], lonc[grid_j[n - 1], grid_i[n - 1]]
        lat1, lon1 = latc[grid_j[n],   grid_i[n]],   lonc[grid_j[n],   grid_i[n]]
        d = geopy.distance.distance((lat0, lon0), (lat1, lon1)).km
        dist.append(dist[-1] + d)

    return np.array(dist), grid_i, grid_j


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":

    # parse dates
    sd = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    ed = datetime.datetime.strptime(end_date_str,   "%Y-%m-%d").date()

    # collect HDF5 files
    scalar_files = collect_hdf5_paths(backup_root, hdf5_file, sd, ed)
    if not scalar_files:
        raise RuntimeError("No HDF5 files in date range.")

    # index times
    scalar_index = index_by_time(scalar_files)
    all_times     = sorted(scalar_index.keys())

    # read lon/lat grid
    with h5py.File(scalar_files[0], "r") as hf0:
        lon = hf0["Grid"]["Longitude"][:]
        lat = hf0["Grid"]["Latitude"][:]

    # cell centers 
    lonc = (lon[:-1,:-1] + lon[:-1,1:] + lon[1:,:-1] + lon[1:,1:]) / 4.0
    latc = (lat[:-1,:-1] + lat[:-1,1:] + lat[1:,:-1] + lat[1:,1:]) / 4.0

    # build path indices & distances
    distance, grid_i, grid_j = get_path_indices(csv_file, lonc, latc)

    # extract vertical slices over time
    vertical_slices = []
    depths = []
    time_titles     = []
    SENTINEL_Z = -9.9e15

    for dt in all_times:
        sfile, stkey = scalar_index[dt][0]
        with h5py.File(sfile, "r") as sh:
            opname = f"OpenPoints_{stkey.split('_')[1]}"
            op3d   = sh["Grid"]["OpenPoints"][opname][:]
            raw3d  = sh["Results"][variable][f"{variable}_{stkey.split('_')[1]}"][:]
            data3d = mask_invalid(raw3d, op3d, sentinel=SENTINEL_Z)

            vzname = f"Vertical_{stkey.split('_')[1]}"
            raw_z3d = sh["Grid"]["VerticalZ"][vzname][:]
            # treat every point as 'open' by passing ones_like()
            z3d     = mask_invalid(raw_z3d, np.ones_like(raw_z3d), sentinel=SENTINEL_Z)
        
        slice2d = np.stack(
            [data3d[:, j, i] for i, j in zip(grid_i, grid_j)],
            axis=1
        )  # shape (nz, nx)

        
        z2d = np.stack(
            [z3d[:, j, i] for i, j in zip(grid_i, grid_j)],
            axis=1
        )  # shape (nz, nx)
        
        
        #find columns that *aren’t* all NaN
        valid_cols = ~np.isnan(z2d).all(axis=0)

        #filter your transect indices & initial arrays
        grid_i    = grid_i[valid_cols]
        grid_j    = grid_j[valid_cols]
        distance  = distance[valid_cols]
        z2d       = z2d[:, valid_cols]
        slice2d   = slice2d[:, valid_cols]

        
        # fill the NaNs with each column’s deepest level
        max_depth_per_col = np.nanmax(z2d, axis=0)

        
        z2d = np.where(np.isnan(z2d), max_depth_per_col, z2d)

        vertical_slices.append(slice2d)
        depths.append(z2d)
   
        time_titles.append(dt.strftime("%d/%m/%Y %H:%M UTC"))
            

    # determine global color limits
    vmin = np.nanmin([np.nanmin(s) for s in vertical_slices])
    vmax = np.nanmax([np.nanmax(s) for s in vertical_slices])

    # --- STATIC PLOT (first timestep) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    #data2d = vertical_slices[0].T  # shape (nz, nx)
    data2d = vertical_slices[0]
    depth = depths[0]
    pcm = ax.pcolormesh(
        distance,     # (nx,)
        depth,      # (nz,)
        data2d[:, :-1],
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Distance along path (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(time_titles[0])

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(f"{label}")

    plt.tight_layout()
    plt.show()

    # --- ANIMATION ---
    def update(idx):
        data2d = vertical_slices[idx]
        pcm.set_array(data2d[:, :-1].ravel())
        ax.set_title(time_titles[idx])
        return [pcm]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(vertical_slices),
        interval=500,
        blit=True
    )
    
    date_span = f"{sd.strftime('%Y%m%d')}_{ed.strftime('%Y%m%d')}"
    out_dir   = os.path.join(figures_folder, date_span, variable)
    os.makedirs(out_dir, exist_ok=True)
    output_mp4  = os.path.join(out_dir, f"{variable}.mp4")

    ani.save(output_mp4, writer="ffmpeg", dpi=dpi)
    print(f"Animation saved to {output_mp4}")
    
    if save_frames:
        for i in range(len(vertical_slices)):
            update(i)
            fig.canvas.draw()
            fn = os.path.join(out_dir, f"{variable}_{i:03d}.png")
            fig.savefig(fn, bbox_inches="tight", dpi=dpi)
        print("Image frames saved in", out_dir)