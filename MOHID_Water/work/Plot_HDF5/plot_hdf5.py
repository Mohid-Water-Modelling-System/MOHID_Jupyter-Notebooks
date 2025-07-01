#!/usr/bin/env python3
"""
Generate a pcolormesh animation of a scalar field (and optional
vector field) when these live in separate HDF5 file trees.
"""
import importlib
import Input_Plot_HDF5
importlib.reload(Input_Plot_HDF5)
from Input_Plot_HDF5 import *

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

def mask_water(data, water_points, is_3d):
    """
    Apply water mask: where water_points == 0 → NaN.
    Handles optional 3D data by dropping extra dims.
    """
    arr = np.squeeze(data)
    if is_3d and arr.ndim > 2:
        arr = arr[-1, :, :]
    return np.where(water_points == 0, np.nan, arr)

# ----------------------------------------
# PARSE DATES & COLLECT FILE LISTS
# ----------------------------------------
sd = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
ed = datetime.datetime.strptime(end_date_str,   "%Y-%m-%d").date()

scalar_files = collect_hdf5_paths(backup_root, hdf5_file, sd, ed)
vector_files = collect_hdf5_paths(backup_root, hdf5_file_vectors, sd, ed) if show_vectors else []

if not scalar_files:
    raise RuntimeError(f"No scalar HDF5s in {hdf5_file} between {start_date_str} and {end_date_str}")

scalar_index = index_by_time(scalar_files)
vector_index = index_by_time(vector_files) if show_vectors else {}

# ----------------------------------------
# INITIAL GRID & MASK FROM FIRST SCALAR FILE
# ----------------------------------------
with h5py.File(scalar_files[0], "r") as h5f:
    X = h5f["Grid"]["Longitude"][:]
    Y = h5f["Grid"]["Latitude"][:]
    if "WaterPoints2D" in h5f["Grid"]:
        water_points = np.squeeze(h5f["Grid"]["WaterPoints2D"][:])
        waterpoints_is_3d = False
    else:
        wp3 = np.squeeze(h5f["Grid"]["WaterPoints3D"][:])
        waterpoints_is_3d = (wp3.ndim == 3)
        water_points = wp3[-1, :, :] if waterpoints_is_3d else wp3

# ----------------------------------------
# SYNC TIMES & PREP FRAME CONTAINERS
# ----------------------------------------
all_times   = sorted(scalar_index.keys())
seen_times  = set()
frames_data = []
U_frames    = []
V_frames    = []
time_titles = []

for dt in all_times:
    if dt in seen_times:
        continue
    seen_times.add(dt)
    # pick the first scalar / vector match
    sfile, stkey = scalar_index[dt][0]
    vmatch = vector_index.get(dt, [])
    vfile, vtkey = (vmatch[0] if vmatch else (None, None))

    # read scalar
    with h5py.File(sfile, "r") as sh:
        dsname = f"{variable}_{stkey.split('_')[1]}"
        tmp = sh["Results"][variable][dsname][:]
        scalar_frame = mask_water(tmp, water_points, waterpoints_is_3d)

    # read vectors
    if show_vectors and vfile:
        with h5py.File(vfile, "r") as vh:
            Uds = f"{variable_vector[0]}_{vtkey.split('_')[1]}"
            Vds = f"{variable_vector[1]}_{vtkey.split('_')[1]}"
            Utmp = vh["Results"][variable_vector[0]][Uds][:]
            Vtmp = vh["Results"][variable_vector[1]][Vds][:]
        Uf = mask_water(Utmp, water_points, waterpoints_is_3d)
        Vf = mask_water(Vtmp, water_points, waterpoints_is_3d)
    else:
        Uf, Vf = None, None

    frames_data.append(scalar_frame)
    U_frames.append(Uf); V_frames.append(Vf)
    time_titles.append(dt.strftime("%d/%m/%Y %H:%M"))

# ----------------------------------------
# COMPUTE MAP EXTENT & ZOOM
# ----------------------------------------
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
global_vmin = min(np.nanmin(f) for f in frames_data)
global_vmax = max(np.nanmax(f) for f in frames_data)

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
    vmin=global_vmin, vmax=global_vmax,
    alpha=transparency_factor, zorder=2
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(quad, cax=cax, orientation="vertical")
cbar.set_label(label, labelpad=25, rotation=270, fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title(f"{time_titles[0]} UTC", fontsize=18)

# precompute cell centers for quiver
Xc = (X[:-1,:-1] + X[:-1,1:] + X[1:,:-1] + X[1:,1:]) / 4.0
Yc = (Y[:-1,:-1] + Y[:-1,1:] + Y[1:,:-1] + Y[1:,1:]) / 4.0

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
        vmin=global_vmin, vmax=global_vmax,
        alpha=transparency_factor, zorder=2
    )
    # draw vectors if available
    if show_vectors and U_frames[i] is not None:
        ax.quiver(
            Xc[::skip_vector, ::skip_vector],
            Yc[::skip_vector, ::skip_vector],
            U_frames[i][::skip_vector, ::skip_vector],
            V_frames[i][::skip_vector, ::skip_vector],
            color=vector_color, scale=vector_scale,
            alpha=0.8, zorder=3
        )
    ax.set_title(f"{time_titles[i]} UTC", fontsize=18)
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