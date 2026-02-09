import importlib
import input_plot_hdf5_residual
importlib.reload(input_plot_hdf5_residual)
from input_plot_hdf5_residual import *

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

import geopandas as gpd
from cartopy.feature import ShapelyFeature
from pathlib import Path

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
        arr = arr[:, :, :]
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

vector_files = collect_hdf5_paths(backup_root, hdf5_file_vectors, sd, ed) 

vector_index = index_by_time(vector_files)

all_times   = sorted(vector_index.keys())
seen_times  = set()
U_frames    = []
V_frames    = []

for dt in all_times:
    if dt in seen_times:
        continue
    seen_times.add(dt)
    vmatch = vector_index.get(dt, [])
    vfile, vtkey = (vmatch[0] if vmatch else (None, None))
    
    with h5py.File(vfile, "r") as vh:
        
        opname = f"OpenPoints_{vtkey.split('_')[1]}"
        openpoints = (vh["Grid"]["OpenPoints"][opname][:])
    
        Uds = f"{variable_vector[0]}_{vtkey.split('_')[1]}"
        Vds = f"{variable_vector[1]}_{vtkey.split('_')[1]}"
        Utmp = vh["Results"][variable_vector[0]][Uds][:]
        Vtmp = vh["Results"][variable_vector[1]][Vds][:]
    Uf = mask_water(Utmp, openpoints)
    Vf = mask_water(Vtmp, openpoints)


    U_frames.append(Uf); V_frames.append(Vf)

# 1. Stack into a single 4D array of shape (n_frames, d1, d2, d3)
stacked_U = np.stack(U_frames, axis=0)
stacked_V = np.stack(V_frames, axis=0)

# 2. Compute mean along the first axis
mean_U_3d = np.mean(stacked_U, axis=0)
mean_V_3d = np.mean(stacked_V, axis=0)

if mean_U_3d.ndim == 3:
    if mean_map == "layer":
        #if 1 <= nlayer <= stacked_U.shape[0]:
        mean_U_2D = mean_U_3d[nlayer, :, :]
        mean_V_2D = mean_V_3d[nlayer, :, :]
        #else:
        #    raise IndexError(f"nlayer {nlayer} out of range (1..{mean_U_3d.shape[0]})")
    else : # mean_map = "surface":
        mean_U_2D = mean_U_3d[-1,:,:]
        mean_V_2D = mean_V_3d[-1,:,:]
else:
    mean_U_2D = mean_U_3d
    mean_V_2D = mean_V_3d
    

U = np.array(mean_U_2D, dtype=np.float32)
V = np.array(mean_V_2D, dtype=np.float32)
Z = (U**2 + V**2)**0.5

# INITIAL GRID
# ----------------------------------------
with h5py.File(vector_files[0], "r") as h5f:
    X = h5f["Grid"]["Longitude"][:]
    Y = h5f["Grid"]["Latitude"][:]
    
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


Fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent)


## Title
ax.set_title(title, fontsize=fontsize_title) 

cimgt.GoogleTiles.get_image = image_spoof # reformat web request for street map spoofing
osm_img = cimgt.GoogleTiles(style='satellite')
#osm_img = cimgt.GoogleTiles(style='street')
ax.add_image(osm_img, zoom_level)

SA = ax.pcolormesh(X,Y,Z[:,:],vmin = vmin,vmax = vmax,cmap=cmap)

## Colorbar
cbar = plt.colorbar(SA, shrink=0.75, pad=0.03) 
cbar.set_label(label,labelpad=25, rotation=270,fontsize=fontsize_label)
cbar.ax.tick_params(labelsize=fontsize_tick)


# precompute cell centers for quiver
Xc = (X[:-1,:-1] + X[:-1,1:] + X[1:,:-1] + X[1:,1:]) / 4.0
Yc = (Y[:-1,:-1] + Y[:-1,1:] + Y[1:,:-1] + Y[1:,1:]) / 4.0

## Countour
contour = ax.contour(Xc,Yc,Z[:,:],levels=countour_levels,colors='grey', transform=ccrs.PlateCarree())
plt.clabel(contour, inline=False, fmt = '%2.1f', colors = 'white', fontsize=18) #contour line labels

ax.quiver(
    Xc[::skip_vector, ::skip_vector],
    Yc[::skip_vector, ::skip_vector],
    U[::skip_vector, ::skip_vector],
    V[::skip_vector, ::skip_vector],
    color=vector_color, scale=vector_scale,
    alpha=0.8, zorder=3
)

if p.exists():
    # add the feature 
    artist = ax.add_feature(shapefile_feature, zorder=4, linewidth=1, alpha=shapefile_transparency_factor)
    gdf.boundary.plot(ax=ax, color=shapefile_color, linewidth=1)
    
os.makedirs(out_dir, exist_ok=True)
#%%
figure_file = os.path.join(out_dir, f"{title}.png")

plt.savefig(figure_file, format='png', dpi=dpi, bbox_inches='tight')
    