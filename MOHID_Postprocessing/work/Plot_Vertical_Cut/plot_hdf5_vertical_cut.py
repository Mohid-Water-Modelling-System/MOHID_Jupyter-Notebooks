#!/usr/bin/env python3
"""
Generate a true vertical cut (depth vs distance) animation along a path
defined by CSV waypoints. Optionally plot velocity vectors projected
onto the cut plane using velocity components U, V, W.
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
    Build dict mapping each datetime -> list of (file_path, time_key).
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
    """
    masked = np.where(open3d == 0, np.nan, data3d)
    if sentinel is not None:
        masked = np.where(np.isclose(masked, sentinel), np.nan, masked)
    return masked


def bresenham(i0, j0, i1, j1):
    """
    Bresenham's line algorithm between two grid indices.
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
    Brute-force search; for large grids consider KDTree replacement.
    """
    dist = np.hypot(lonc - lon, latc - lat)
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    return int(i), int(j)


def get_path_indices(csv_file, lonc, latc):
    """
    Read lon/lat vertices from CSV, map to grid cells, and
    trace all cells along the path via Bresenham.
    Returns arrays: distance[km], grid_i, grid_j.
    """
    df = pd.read_csv(csv_file, sep=',', header=None, names=['lon', 'lat'], engine='python')
    lons = df["lon"].tolist()
    lats = df["lat"].tolist()

    verts = [grid_location(lo, la, lonc, latc) for lo, la in zip(lons, lats)]

    cells = []
    for k in range(len(verts) - 1):
        seg = bresenham(*verts[k], *verts[k + 1])
        if k > 0:
            seg = seg[1:]
        cells.extend(seg)

    if len(cells) == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    grid_i = np.array([i for i, _ in cells], int)
    grid_j = np.array([j for _, j in cells], int)

    dist = [0.0]
    for n in range(1, len(grid_i)):
        lat0, lon0 = latc[grid_j[n - 1], grid_i[n - 1]], lonc[grid_j[n - 1], grid_i[n - 1]]
        lat1, lon1 = latc[grid_j[n],   grid_i[n]],   lonc[grid_j[n],   grid_i[n]]
        d = geopy.distance.distance((lat0, lon0), (lat1, lon1)).km
        dist.append(dist[-1] + d)

    return np.array(dist), grid_i, grid_j


def centers_to_bounds(centers):
    """
    Convert 1D array of centers (N,) to boundaries (N+1,).
    Robust to identical adjacent centers by using a tiny epsilon.
    """
    c = np.asarray(centers, dtype=float)
    n = c.size
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    d = np.diff(c)
    eps = 1e-9
    d_safe = np.where(np.isclose(d, 0.0), eps, d)
    left = c[0] - d_safe[0] / 2.0
    right = c[-1] + d_safe[-1] / 2.0
    bounds = np.empty(n + 1, dtype=float)
    bounds[0] = left
    bounds[-1] = right
    bounds[1:-1] = c[:-1] + d_safe / 2.0
    return bounds


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":

    # parse dates (expected in Input_Plot_HDF5_Cut)
    sd = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    ed = datetime.datetime.strptime(end_date_str,   "%Y-%m-%d").date()

    # collect HDF5 files
    scalar_files = collect_hdf5_paths(backup_root, hdf5_file, sd, ed)
    if not scalar_files:
        raise RuntimeError("No HDF5 files in date range.")
        
    vector_files = collect_hdf5_paths(backup_root, hdf5_file_vectors, sd, ed)
    if not vector_files:
        raise RuntimeError("No HDF5 files for vectors in date range.")

    # index times
    scalar_index = index_by_time(scalar_files)
    all_times     = sorted(scalar_index.keys())
    
    vector_index = index_by_time(vector_files)
    vector_all_times     = sorted(vector_index.keys())

    # read lon/lat grid
    with h5py.File(scalar_files[0], "r") as hf0:
        lon = hf0["Grid"]["Longitude"][:]
        lat = hf0["Grid"]["Latitude"][:]

    # cell centers (grid cell centers computed from node corners)
    lonc = (lon[:-1,:-1] + lon[:-1,1:] + lon[1:,:-1] + lon[1:,1:]) / 4.0
    latc = (lat[:-1,:-1] + lat[:-1,1:] + lat[1:,:-1] + lat[1:,1:]) / 4.0

    # build path indices & distances (kept immutable)
    distance, grid_i, grid_j = get_path_indices(csv_file, lonc, latc)

    # prepare storage
    vertical_slices = []      # list of 2D arrays (nz, ncols_filtered)
    depths = []               # list of 2D z arrays (nz, ncols_filtered)
    time_titles = []
    distances_per_timestep = []  # store distance arrays matching each filtered slice
    x_bounds_list = []
    y_bounds_list = []
    quiver_grids = []         # list of (Xq, Yq, Uq, Wq) per timestep
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

        # --- READ VELOCITIES IF REQUESTED (names may vary) ---
        if show_vectors:
            vfile, vtkey = vector_index[dt][0]
            with h5py.File(vfile, "r") as sh:

                rawU = sh["Results"]["velocity U"][f"velocity U_{vtkey.split('_')[1]}"][:]
                rawV = sh["Results"]["velocity V"][f"velocity V_{vtkey.split('_')[1]}"][:]
                rawW = sh["Results"]["velocity W"][f"velocity W_{vtkey.split('_')[1]}"][:]

                U3d = mask_invalid(rawU, op3d, sentinel=SENTINEL_Z)
                V3d = mask_invalid(rawV, op3d, sentinel=SENTINEL_Z)
                W3d = mask_invalid(rawW, op3d, sentinel=SENTINEL_Z)
        else:
            U3d = V3d = W3d = None

        # extract vertical slice along the (immutable) grid_i/grid_j path
        if grid_i.size == 0:
            continue

        slice2d = np.stack(
            [data3d[:, j, i] for i, j in zip(grid_i, grid_j)],
            axis=1
        )  # shape (nz, ncols_original)

        z2d = np.stack(
            [z3d[:, j, i] for i, j in zip(grid_i, grid_j)],
            axis=1
        )  # shape (nz, ncols_original)

        # find columns that *aren't* all NaN (vertical grid not present or fully masked)
        valid_cols = ~np.isnan(z2d).all(axis=0)
        if not np.any(valid_cols):
            # nothing valid for this timestep; skip
            continue

        # operate on local copies for this timestep so the original path remains intact
        grid_i_t = grid_i[valid_cols].copy()
        grid_j_t = grid_j[valid_cols].copy()
        distance_t = distance[valid_cols].copy()
        z2d = z2d[:, valid_cols]
        slice2d = slice2d[:, valid_cols]

        # fill NaNs using per-column deepest valid level
        max_depth_per_col = np.nanmax(z2d, axis=0)   # shape (ncols,)
        z2d_filled = z2d.copy()
        nan_mask = np.isnan(z2d_filled)
        if nan_mask.any():
            rep = np.repeat(max_depth_per_col[np.newaxis, :], z2d_filled.shape[0], axis=0)
            z2d_filled[nan_mask] = rep[nan_mask]
        z2d = z2d_filled

        # compute per-timestep centers and bounds and store them
        nz_data_t = slice2d.shape[0]
        z_centers_t = np.nanmean(z2d[:nz_data_t, :], axis=1)
        x_centers_t = distance_t

        x_bounds_t = centers_to_bounds(x_centers_t)
        y_bounds_t = centers_to_bounds(z_centers_t)

        vertical_slices.append(slice2d)
        depths.append(z2d)
        distances_per_timestep.append(distance_t)
        x_bounds_list.append(x_bounds_t)
        y_bounds_list.append(y_bounds_t)
        time_titles.append(dt.strftime("%d/%m/%Y %H:%M UTC"))

        # --- VELOCITY: extract slices and compute along-path projection and downsample ---
        if show_vectors:
            Uslice = np.stack([U3d[:, j, i] for i, j in zip(grid_i, grid_j)], axis=1)[:, valid_cols]
            Vslice = np.stack([V3d[:, j, i] for i, j in zip(grid_i, grid_j)], axis=1)[:, valid_cols]
            Wslice = np.stack([W3d[:, j, i] for i, j in zip(grid_i, grid_j)], axis=1)[:, valid_cols]

            # get column lon/lat centers for this filtered path
            lon_cols = np.array([lonc[grid_j_t[n], grid_i_t[n]] for n in range(grid_i_t.size)])
            lat_cols = np.array([latc[grid_j_t[n], grid_i_t[n]] for n in range(grid_i_t.size)])

            # approximate meters per degree at mean latitude
            mean_lat = np.nanmean(lat_cols)
            m_per_deg_lat = geopy.distance.distance((mean_lat - 1.0, lon_cols.mean()), (mean_lat, lon_cols.mean())).m
            m_per_deg_lon = geopy.distance.distance((mean_lat, lon_cols.mean() - 1.0), (mean_lat, lon_cols.mean())).m

            Xm = (lon_cols - lon_cols.mean()) * m_per_deg_lon
            Ym = (lat_cols - lat_cols.mean()) * m_per_deg_lat

            dXm = np.gradient(Xm)
            dYm = np.gradient(Ym)
            norms = np.hypot(dXm, dYm)
            norms[norms == 0] = 1.0
            tx = dXm / norms
            ty = dYm / norms

            # project horizontal velocities onto path tangent
            u_along = Uslice * tx[np.newaxis, :] + Vslice * ty[np.newaxis, :]

            # downsample indices to avoid clutter
            ncols_ds = max(1, int(np.ceil(u_along.shape[1] / max_arrows_across)))
            nz_ds = max(1, int(np.ceil(u_along.shape[0] / max_arrows_vertical)))
            cols_ds = np.arange(0, u_along.shape[1], ncols_ds, dtype=int)
            zs_ds = np.arange(0, u_along.shape[0], nz_ds, dtype=int)

            # x (distance) and z (depth) centers for arrows
            x_centers_ds = distance_t[cols_ds]
            z_centers_levels = np.nanmean(z2d[:nz_data_t, :], axis=1)
            y_centers_ds = z_centers_levels[zs_ds]

            Xq, Yq = np.meshgrid(x_centers_ds, y_centers_ds)
            Uq = u_along[zs_ds[:, None], cols_ds[None, :]]
            Wq = Wslice[zs_ds[:, None], cols_ds[None, :]]

            # mask tiny vectors
            mag = np.hypot(Uq, Wq)
            Uq = np.where(mag < min_vector_mag, np.nan, Uq)
            Wq = np.where(mag < min_vector_mag, np.nan, Wq)

            # push to list (note: use -Wq if you want positive downward when depth axis increases downward)
            quiver_grids.append((Xq, Yq, Uq, -Wq))
        else:
            quiver_grids.append((None, None, None, None))

    # check we have something to plot
    if len(vertical_slices) == 0:
        raise RuntimeError("No valid vertical slices found for any timestep.")

    # determine global color limits if not set
    if vmin is None:
        vmin = np.nanmin([np.nanmin(s) for s in vertical_slices])
    if vmax is None:
        vmax = np.nanmax([np.nanmax(s) for s in vertical_slices])

    # --- build representative axes and check shapes before plotting ---
    first_idx = 0
    data2d = vertical_slices[first_idx]   # shape (nz_data, ncols)
    distance0 = distances_per_timestep[first_idx]  # shape (ncols,)

    nz_data, ncols = data2d.shape

    if y_bounds_list[first_idx].size != nz_data + 1:
        raise RuntimeError(f"After alignment y_bounds length {y_bounds_list[first_idx].size} != nz+1 ({nz_data+1}).")
    if x_bounds_list[first_idx].size != ncols + 1:
        raise RuntimeError(f"x_bounds length {x_bounds_list[first_idx].size} != ncols+1 ({ncols+1}).")

    def make_mesh(ax, x_bounds, y_bounds, data2d, **pcm_kwargs):
        """
        Create a pcolormesh on ax using bounds arrays and 2D data.
        x_bounds: (ncols+1,), y_bounds: (nz+1,), data2d: (nz, ncols)
        """
        pcm = ax.pcolormesh(x_bounds, y_bounds, data2d, **pcm_kwargs)
        return pcm

    # --- prepare figure and initial frame (frame 0) ---
    fig, ax = plt.subplots(figsize=(12, 5))

    pcm = make_mesh(
        ax,
        x_bounds_list[0],
        y_bounds_list[0],
        vertical_slices[0],
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    # draw initial contours
    Xb0 = x_bounds_list[0]
    Yb0 = y_bounds_list[0]
    x_centers0 = 0.5 * (Xb0[:-1] + Xb0[1:])
    y_centers0 = 0.5 * (Yb0[:-1] + Yb0[1:])
    Xg0, Yg0 = np.meshgrid(x_centers0, y_centers0)

    contour_set = ax.contour(
        Xg0, Yg0, vertical_slices[0],
        levels=contour_levels,
        colors='w',
        linewidths=0.6,
        linestyles="solid"
    )

    # label only initial contours to avoid clutter during animation
    ax.clabel(contour_set, inline=True, fmt="%.2f", fontsize=8)

    Yb = y_bounds_list[:]
    y_min, y_max = float(np.min(Yb)), float(np.max(Yb))

    ax.set_ylim(y_max, y_min)
    ax.invert_xaxis()
    ax.set_xlabel("Distance along path (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(time_titles[first_idx])

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(f"{label}")

    # initial quiver (if enabled)
    quiv = None
    if show_vectors:
        Xq0, Yq0, Uq0, Wq0 = quiver_grids[0]
        if Xq0 is not None:
            quiv = ax.quiver(Xq0, Yq0, Uq0, Wq0, angles='xy', scale_units='xy',
                             scale=scale_quiver, color=quiver_color, width=quiver_width)

    plt.tight_layout()

    # --- ANIMATION: recreate mesh/contour/quiver each frame (blit=False for robustness) ---
    def update(idx):
        global pcm, contour_set, cbar, quiv

        # remove previous pcm safely
        try:
            if pcm is not None:
                pcm.remove()
        except Exception:
            pass

        for coll in list(ax.collections):
            try:
                coll.remove()
            except Exception:
                pass

        # remove previous quiver
        try:
            if quiv is not None:
                quiv.remove()
        except Exception:
            pass

        # create new mesh for this frame
        x_b = x_bounds_list[idx]
        y_b = y_bounds_list[idx]
        data2d = vertical_slices[idx]

        pcm = make_mesh(
            ax,
            x_b,
            y_b,
            data2d,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        # contours
        x_centers = 0.5 * (x_b[:-1] + x_b[1:])
        y_centers = 0.5 * (y_b[:-1] + y_b[1:])
        if x_centers.size == 0 or y_centers.size == 0:
            contour_set = None
        else:
            Xg, Yg = np.meshgrid(x_centers, y_centers)
            contour_set = ax.contour(
                Xg, Yg, data2d,
                levels=contour_levels,
                colors='w',
                linewidths=0.6,
                linestyles="solid"
            )
            
        ax.clabel(contour_set, inline=True, fmt="%.2f", fontsize=8)

        # quiver for this frame
        if show_vectors:
            Xq, Yq, Uq, Wq = quiver_grids[idx]
            if Xq is not None:
                quiv = ax.quiver(Xq, Yq, Uq, Wq, angles='xy', scale_units='xy',
                                 scale=scale_quiver, color=quiver_color, width=quiver_width)
            else:
                quiv = None

        ax.set_ylim(y_max, y_min)
        ax.set_title(time_titles[idx])

        # keep colorbar stable; update mappable
        try:
            cbar.update_normal(pcm)
        except Exception:
            try:
                cbar.remove()
            except Exception:
                pass
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(f"{label}")

        fig.canvas.draw_idle()
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(vertical_slices),
        interval=500,
        blit=False
    )

    date_span = f"{sd.strftime('%Y%m%d')}_{ed.strftime('%Y%m%d')}"
    out_dir   = os.path.join(figures_folder, date_span, variable)
    os.makedirs(out_dir, exist_ok=True)
    output_mp4  = os.path.join(out_dir, f"{variable}.mp4")

    # save with ffmpeg if available; handle errors gracefully
    try:
        fig.canvas.draw()
        ani.save(output_mp4, writer="ffmpeg", dpi=dpi)
        print(f"Animation saved to {output_mp4}")
    except Exception as exc:
        print("Failed to save animation with ffmpeg:", exc)

    if save_frames:
        for i in range(len(vertical_slices)):
            update(i)
            fig.canvas.draw()
            fn = os.path.join(out_dir, f"{variable}_{i:03d}.png")
            fig.savefig(fn, bbox_inches="tight", dpi=dpi)
        print("Image frames saved in", out_dir)