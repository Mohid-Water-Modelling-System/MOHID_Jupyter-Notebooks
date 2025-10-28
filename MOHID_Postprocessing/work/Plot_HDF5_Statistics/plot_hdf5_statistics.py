import importlib
import input_plot_hdf5_statistics
importlib.reload(input_plot_hdf5_statistics)
from input_plot_hdf5_statistics import *

import os
import glob
import h5py
import datetime
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
from datetime import datetime as dt

import PIL
#%% Usar este pada adicionar outras fontes de mapas de fundo como o:  OSM  ou o: QuadtreeTiles
import io
from PIL import Image
from urllib.request import urlopen, Request
import scipy.ndimage

import rasterio
from rasterio.transform import from_origin

title = f'Percentil ' + str(percentil) + '_' + variable

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
            for tkey in sorted(h5f["Time"].keys()):
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
        arr = arr[-1, :, :]
    return np.where(openpoints == 0, np.nan, arr)
    
def image_spoof(self, tile):
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy


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

all_times   = sorted(scalar_index.keys())
seen_times  = set()
frames_data = []
U_frames    = []
V_frames    = []

for dt in all_times:
    if dt in seen_times:
        continue
    seen_times.add(dt)
    # pick the first scalar 
    sfile, stkey = scalar_index[dt][0]
    vmatch = vector_index.get(dt, [])
    vfile, vtkey = (vmatch[0] if vmatch else (None, None))
    
    # read scalar
    with h5py.File(sfile, "r") as sh:

        opname = f"OpenPoints_{stkey.split('_')[1]}"
        openpoints = (sh["Grid"]["OpenPoints"][opname][:])
        
        dsname = f"{variable}_{stkey.split('_')[1]}"
        tmp = sh["Results"][variable][dsname][:]
        scalar_frame = mask_water(tmp, openpoints)
        
    # read vectors
    if show_vectors and vfile:
        with h5py.File(vfile, "r") as vh:
            
            opname = f"OpenPoints_{stkey.split('_')[1]}"
            openpoints = (vh["Grid"]["OpenPoints"][opname][:])
        
            Uds = f"{variable_vector[0]}_{vtkey.split('_')[1]}"
            Vds = f"{variable_vector[1]}_{vtkey.split('_')[1]}"
            Utmp = vh["Results"][variable_vector[0]][Uds][:]
            Vtmp = vh["Results"][variable_vector[1]][Vds][:]
        Uf = mask_water(Utmp, openpoints)
        Vf = mask_water(Vtmp, openpoints)
    else:
        Uf, Vf = None, None

    frames_data.append(scalar_frame)
    U_frames.append(Uf); V_frames.append(Vf)

# 1. Stack into a single 4D array of shape (n_frames, d1, d2, d3)
stacked = np.stack(frames_data, axis=0)

# 2. Compute percentile along the first axis
percentil_3d = np.percentile(stacked, percentil, axis=0)

# If percentil_3d has shape (nz, nrows, ncols):
if percentil_3d.ndim == 3:
    if percentil_map == "max_value":
        percentil_2D = percentil_3d.max(axis=0)
    elif percentil_map == "layer":
        if 1 <= nlayer <= percentil_3d.shape[0]:
            percentil_2D = percentil_3d[nlayer-1, :, :]
        else:
            raise IndexError(f"nlayer {nlayer} out of range (1..{percentil_3d.shape[0]})")
    else : # percentil_map = "surface":
        percentil_2D = percentil_3d[-1,:,:]
else:
    percentil_2D = percentil_3d

Z = np.array(percentil_2D, dtype=np.float32)
#Z = np.where(np.isnan(Z), np.nan, Z)
#Z = np.where(Z <= 0.0, np.nan, Z)


if show_vectors and vfile:
    # 1. Stack into a single 4D array of shape (n_frames, d1, d2, d3)
    stacked_U = np.stack(U_frames, axis=0)
    stacked_V = np.stack(V_frames, axis=0)

    # 2. Compute percentile along the first axis
    percentil_U_3d = np.percentile(stacked_U, percentil, axis=0)
    percentil_V_3d = np.percentile(stacked_V, percentil, axis=0)

    # If percentil_3d has shape (nz, nrows, ncols):
    if percentil_U_3d.ndim == 3:
        if percentil_map == "max_value":
            percentil_U_2D = percentil_U_3d.max(axis=0)
            percentil_V_2D = percentil_V_3d.max(axis=0)
        elif percentil_map == "layer":
            if 1 <= nlayer <= percentil_U_3d.shape[0]:
                percentil_U_2D = percentil_U_3d[nlayer-1, :, :]
                percentil_V_2D = percentil_V_3d[nlayer-1, :, :]
            else:
                raise IndexError(f"nlayer {nlayer} out of range (1..{percentil_U_3d.shape[0]})")
        else : # percentil_map = "surface":
            percentil_U_2D = percentil_U_3d[-1,:,:]
            percentil_V_2D = percentil_V_3d[-1,:,:]
    else:
        percentil_U_2D = percentil_U_3d
        percentil_V_2D = percentil_V_3d

    U = np.array(percentil_U_2D, dtype=np.float32)
    V = np.array(percentil_V_2D, dtype=np.float32)

# INITIAL GRID
# ----------------------------------------
with h5py.File(scalar_files[0], "r") as h5f:
    X = h5f["Grid"]["Longitude"][:]
    Y = h5f["Grid"]["Latitude"][:]
    
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


Fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent)


## Title
ax.set_title(title, fontsize=18) 

cimgt.GoogleTiles.get_image = image_spoof # reformat web request for street map spoofing
osm_img = cimgt.GoogleTiles(style='satellite')
#osm_img = cimgt.GoogleTiles(style='street')
ax.add_image(osm_img, zoom_level)

SA = ax.pcolormesh(X,Y,Z[:,:],vmin = vmin,vmax = vmax,cmap=cmap)

## Colorbar
cbar = plt.colorbar(SA, shrink=0.95, pad=0.03) 
cbar.set_label(label,labelpad=25, rotation=270,fontsize=18)
cbar.ax.tick_params(labelsize=18)


# precompute cell centers for quiver
Xc = (X[:-1,:-1] + X[:-1,1:] + X[1:,:-1] + X[1:,1:]) / 4.0
Yc = (Y[:-1,:-1] + Y[:-1,1:] + Y[1:,:-1] + Y[1:,1:]) / 4.0

## Countour
contour = ax.contour(Xc,Yc,Z[:,:],levels=countour_levels,colors='grey', transform=ccrs.PlateCarree())
plt.clabel(contour, inline=False, fmt = '%2.1f', colors = 'white', fontsize=18) #contour line labels

if show_vectors:
    ax.quiver(
        Xc[::skip_vector, ::skip_vector],
        Yc[::skip_vector, ::skip_vector],
        U[::skip_vector, ::skip_vector],
        V[::skip_vector, ::skip_vector],
        color=vector_color, scale=vector_scale,
        alpha=0.8, zorder=3
    )
    
os.makedirs(out_dir, exist_ok=True)
#%%
figure_file = os.path.join(out_dir, f"{variable}_percentil_{percentil}.png")

plt.savefig(figure_file, format='png', dpi=dpi, bbox_inches='tight')
    
#Export to GeoTIFF
nrows, ncols = Z.shape

# Determine spatial extent and pixel size
x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()
pixel_width  = (x_max - x_min) / (ncols  - 1)
pixel_height = (y_max - y_min) / (nrows  - 1)

# Build an affine transform
transform = from_origin(
    x_min - pixel_width/2,  # West edge  (shift half a pixel)
    y_max + pixel_height/2, # North edge (shift half a pixel)
    pixel_width,
    pixel_height
)

# Flip the array so row 0 becomes north
Z_raster = Z.T
Z_raster = np.flipud(Z_raster)

raster_file = os.path.join(out_dir, f"{variable}_percentil_{percentil}.tif")

# Write GeoTIFF
with rasterio.open(
    raster_file,           # output filename
    'w',
    driver='GTiff',
    height=nrows,
    width=ncols,
    count=1,                # one band of data
    dtype=Z.dtype,
    crs='EPSG:4326',        # replace with your CRS
    transform=transform,
) as dst:
    dst.write(Z_raster, 1)         # write array into band 1

