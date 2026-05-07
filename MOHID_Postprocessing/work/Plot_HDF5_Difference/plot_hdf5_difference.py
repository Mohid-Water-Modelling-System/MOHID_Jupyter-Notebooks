#!/usr/bin/env python3
"""
Generate a pcolormesh animation of a scalar field  
"""
import importlib
import Input_Plot_HDF5_Difference
importlib.reload(Input_Plot_HDF5_Difference)
from Input_Plot_HDF5_Difference import *

import os
import glob
import io
import h5py
import numpy as np
import datetime
from urllib.request import Request, urlopen
from PIL import Image

from matplotlib import pyplot as plt, animation
import imageio_ffmpeg
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Point Matplotlib to the ffmpeg executable provided by imageio_ffmpeg
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
         
import geopandas as gpd
from cartopy.feature import ShapelyFeature
from pathlib import Path
         

# ----------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------
def collect_hdf5_paths(root, h5file, sd, ed):
    paths = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        try:
            day = datetime.datetime.strptime(entry.name.split('_')[0], "%Y%m%d").date()
        except Exception:
            continue
        if sd <= day <= ed:
            # Look directly inside the date-folder
            pattern = os.path.join(entry.path, h5file)
            for f in glob.glob(pattern):
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
                dt = datetime.datetime(int(y),int(m),int(d),int(H),int(M))
                idx.setdefault(dt, []).append((path, tkey))
    return idx

def mask_water(data, openpoints):
    """
    Apply water mask: where openpoints == 0 → NaN.
    Handles optional 3D data by dropping extra dims.
    """
    
    openpoints = np.squeeze(openpoints)
    # If truly 3D, pick surface (or any other) layer
    if openpoints.ndim == 3:
        openpoints = openpoints[-1, :, :]
        
    arr = np.squeeze(data)
    # If truly 3D, pick surface (or any other) layer
    if arr.ndim == 3:
        if map == "layer":
            arr = arr[nlayer, :, :]
        else: #if map == "surface":
            arr = arr[-1, :, :]
    return np.where(openpoints == 0, np.nan, arr)


# ----------------------------------------
# PARSE DATES & COLLECT FILE LISTS
# ----------------------------------------
sd = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
ed = datetime.datetime.strptime(end_date_str,   "%Y-%m-%d").date()

# ----------------------------------------
# INPUT: two scenarios
# ----------------------------------------
scalar_files_A = collect_hdf5_paths(root_A, hdf5_file, sd, ed)
scalar_files_B = collect_hdf5_paths(root_B, hdf5_file, sd, ed)

index_A = index_by_time(scalar_files_A)
index_B = index_by_time(scalar_files_B)

# ----------------------------------------
# FIND COMMON TIMES
# ----------------------------------------
common_times = sorted(set(index_A.keys()) & set(index_B.keys()))

if not common_times:
    raise RuntimeError("No overlapping timesteps between scenarios")

# ----------------------------------------
# INITIAL GRID
# ----------------------------------------
with h5py.File(scalar_files_A[0], "r") as h5f:
    X = h5f["Grid"]["Longitude"][:]
    Y = h5f["Grid"]["Latitude"][:]

# ----------------------------------------
# SYNC TIMES & PREP FRAME CONTAINERS
# ----------------------------------------

frames_data = []
time_titles = []
    
# ----------------------------------------
# LOOP OVER MATCHED TIMES
# ----------------------------------------
for dt in common_times:

    file_A, tkey_A = index_A[dt][0]
    file_B, tkey_B = index_B[dt][0]

    # ---- Read scenario A ----
    with h5py.File(file_A, "r") as hA:
        opname = f"OpenPoints_{tkey_A.split('_')[1]}"
        open_A = hA["Grid"]["OpenPoints"][opname][:]

        dsname_A = f"{variable}_{tkey_A.split('_')[1]}"
        data_A = hA[group][variable][dsname_A][:]

        frame_A = mask_water(data_A, open_A)

    # ---- Read scenario B ----
    with h5py.File(file_B, "r") as hB:
        opname = f"OpenPoints_{tkey_B.split('_')[1]}"
        open_B = hB["Grid"]["OpenPoints"][opname][:]

        dsname_B = f"{variable}_{tkey_B.split('_')[1]}"
        data_B = hB[group][variable][dsname_B][:]

        frame_B = mask_water(data_B, open_B)

    # ---- Compute difference (B - A) ----
    diff = np.where(
        np.isnan(frame_A) | np.isnan(frame_B),
        np.nan,
        frame_B - frame_A
    )

    frames_data.append(diff)
    time_titles.append(dt.strftime("%d/%m/%Y %H:%M"))
# ----------------------------------------
# COMPUTE MAP EXTENT & ZOOM
# ----------------------------------------
if extent == None:
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    dx = (x_max - x_min) / X.shape[0]
    dy = (y_max - y_min) / Y.shape[0]
    extent = [
        x_min - extent_cells*dx, x_max + extent_cells*dx,
        y_min - extent_cells*dy, y_max + extent_cells*dy
    ]
def calculate_zoom_level(increase):
    lat_rng = extent[3] - extent[2]
    lon_rng = extent[1] - extent[0]
    avg = max(lat_rng, lon_rng)
    z = int(np.log2(360/avg))
    return max(1, min(z + increase, 19))
zoom_level = calculate_zoom_level(increase_zoom_level)

# patch Cartopy GoogleTiles
def _spoof(self, tile):
    req = Request(self._image_url(tile))
    req.add_header("User-agent", "Anaconda 3")
    fh = urlopen(req)
    img = Image.open(io.BytesIO(fh.read())).convert(self.desired_tile_form)
    fh.close()
    return img, self.tileextent(tile), "lower"
cimgt.GoogleTiles.get_image = _spoof

# ----------------------------------------
# INITIALIZE FIGURE
# ----------------------------------------
    # determine global color limits if not set
if vmin is None:
    vmin = min(np.nanmin(f) for f in frames_data)
if vmax is None:
    vmax = max(np.nanmax(f) for f in frames_data)

# Read shapefile once (if provided)
p = Path(shapefile_path)
if p.exists():
    gdf = gpd.read_file(shapefile_path)
    # create a Cartopy ShapelyFeature for fast drawing with transform
    shapefile_feature = ShapelyFeature(
        gdf.geometry,
        ccrs.PlateCarree(),
        facecolor=shapefile_color,  # or shapefile_color if you want filled polygons
        edgecolor=shapefile_color
    )


fig, ax = plt.subplots(
    figsize=(10,10),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
ax.set_extent(extent, crs=ccrs.PlateCarree())
osm = cimgt.GoogleTiles(style="satellite")
ax.add_image(osm, zoom_level)

quad = ax.pcolormesh(
    X, Y, frames_data[0],
    shading="auto", cmap=cmap,
    vmin=vmin, vmax=vmax,
    alpha=transparency_factor, zorder=2
)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(quad, cax=cax, orientation="vertical")
cbar.set_label(label, labelpad=25, rotation=270, fontsize=fontsize_label)
cbar.ax.tick_params(labelsize=fontsize_tick)
ax.set_title(f"{time_titles[0]} UTC", fontsize=fontsize_title)

# ----------------------------------------
# ANIMATION UPDATE FUNCTION
# ----------------------------------------
def update(i):
    # clear previous plot elements
    for coll in list(ax.collections):
        coll.remove()
    # draw scalar
    ax.pcolormesh(
        X, Y, frames_data[i],
        shading="auto", cmap=cmap,
        vmin=vmin, vmax=vmax,
        alpha=transparency_factor, zorder=2
    )

    ax.set_title(f"{time_titles[i]} UTC", fontsize=18)
    
    if p.exists():
        # add the feature 
        artist = ax.add_feature(shapefile_feature, zorder=4, linewidth=1, alpha=shapefile_transparency_factor)
        gdf.boundary.plot(ax=ax, color=shapefile_color, linewidth=1)

    return list(ax.collections)

# ----------------------------------------
# BUILD & SAVE ANIMATION
# ----------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(frames_data),
    interval=500, blit=False
)

date_span = f"{sd.strftime('%Y%m%d')}_{ed.strftime('%Y%m%d')}"
out_dir   = os.path.join(figures_folder, date_span, variable)
os.makedirs(out_dir, exist_ok=True)
mp4_path  = os.path.join(out_dir, f"{variable}.mp4")

ani.save(mp4_path, writer=animation.FFMpegWriter(fps=2), dpi=dpi)
print("Animation exported to", mp4_path)

if save_frames:
    for i in range(len(frames_data)):
        update(i)
        fn = os.path.join(out_dir, f"{variable}_{i:03d}.png")
        plt.savefig(fn, bbox_inches="tight", dpi=dpi)
    print("Image frames saved in", out_dir)