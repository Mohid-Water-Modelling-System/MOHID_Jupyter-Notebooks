import importlib
import Input_Plot_HDF5
importlib.reload(Input_Plot_HDF5)
from Input_Plot_HDF5 import *
import os
import io
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import datetime
from IPython.display import HTML
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from urllib.request import Request, urlopen
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio_ffmpeg
import matplotlib as mpl

# Point Matplotlib to the ffmpeg executable provided by imageio-ffmpeg
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# ============================
# DEFINE VARIABLE-LABEL DICTIONARY
# ============================
# variable_label_dict = {
    # "wind velocity": "Wind Velocity (m/s)",
    # "air temperature": "Temperature (ºC)",
    # "solar radiation": "Solar Radiation (W/m²)"
# }

# variable_vector = ["wind velocity X", "wind velocity Y"]

# ============================
# SET-UP: Define paths and file names
# ============================
#dir_path = os.path.join(case_dir, "GeneralData", "BoundaryConditions", "ERA5", "backup", "20250101_20250105")
#hdf5_file = os.path.join(dir_path, "Meteo.hdf5")

#variable = "wind velocity"  # Change as needed
#label = variable_label_dict.get(variable, "Unknown Variable")  # Fetch label from dictionary

# Option to enable or disable vector overlay and image frame saving
#show_vectors = True      # Set to False to disable wind vectors in the animation
#save_frames = True       # Set to False to disable saving individual image frames

# User-specified parameters for skipping time steps, adjusting extent, vectors, etc.
# skip_time = 3           # Sample every nth time step
# extent_cells = 5        # Number of extra cells added to the plot extent
# increase_zoom_level = 1 # Increase computed zoom level by this amount
# skip_vector = 2         # Skip factor when plotting vectors (to reduce clutter)
# vector_scale = 250      # Scale for the wind vector arrows
# vector_color = 'white'  # Color for the wind vectors
# transparency_factor = 0.5

# ============================
# READ THE FILE AND COLLECT FRAMES
# ============================
with h5py.File(hdf5_file, 'r') as ah:
    # Read grid boundaries
    Y = ah["Grid"]["Latitude"][:]    # e.g., shape (47, 31)
    X = ah["Grid"]["Longitude"][:]     # e.g., shape (47, 31)
    
    # Check if a 2D or 3D water mask exists
    if "WaterPoints2D" in ah["Grid"]:
        water_points = np.squeeze(ah["Grid"]["WaterPoints2D"][:])
        waterpoints_is_3d = False
    elif "WaterPoints3D" in ah["Grid"]:
        water_points_3d = np.squeeze(ah["Grid"]["WaterPoints3D"][:])
        if water_points_3d.ndim == 3:
            water_points = water_points_3d[-1, :, :]  # Use the last vertical layer
        else:
            water_points = water_points_3d  # Already 2D; no further indexing needed
        waterpoints_is_3d = True
    else:
        raise ValueError("Neither WaterPoints2D nor WaterPoints3D found in the HDF5 file.")
    
    # Sample every nth time step
    time_keys = sorted(list(ah["Time"].keys()))
    time_keys = time_keys[::skip_time]
    
    frames_data = []   # Main variable frames (e.g., wind velocity)
    time_titles = []   # Time stamps for each frame
    
    for tkey in time_keys:
        # Create a human-readable timestamp
        d = list(ah["Time"][tkey])
        dt = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]))
        time_titles.append(dt.strftime("%d/%m/%Y %H:%M"))
        
        instant = tkey.split("_")[1]
        # Read main meteorological variable data
        temp_key = f"{variable}_{instant}"
        temp_data = np.squeeze(ah["Results"][variable][temp_key][:])
        # If waterpoints are 3D and the data has a vertical dimension, use the last layer.
        if waterpoints_is_3d and temp_data.ndim > 2:
            temp_data = temp_data[-1, :, :]
        masked_data = np.where(water_points == 0, np.nan, temp_data)
        frames_data.append(masked_data)

    if show_vectors:
        with h5py.File(hdf5_file_vectors, 'r') as ah:
            # Read grid boundaries
            Y = ah["Grid"]["Latitude"][:]    # e.g., shape (47, 31)
            X = ah["Grid"]["Longitude"][:]     # e.g., shape (47, 31)
            
            # Check if a 2D or 3D water mask exists
            if "WaterPoints2D" in ah["Grid"]:
                water_points = np.squeeze(ah["Grid"]["WaterPoints2D"][:])
                waterpoints_is_3d = False
            elif "WaterPoints3D" in ah["Grid"]:
                water_points_3d = np.squeeze(ah["Grid"]["WaterPoints3D"][:])
                if water_points_3d.ndim == 3:
                    water_points = water_points_3d[-1, :, :]  # Use the last vertical layer
                else:
                    water_points = water_points_3d  # Already 2D; no further indexing needed
                waterpoints_is_3d = True
            else:
                raise ValueError("Neither WaterPoints2D nor WaterPoints3D found in the HDF5 file.")
            
            # Sample every nth time step
            time_keys = sorted(list(ah["Time"].keys()))
            time_keys = time_keys[::skip_time]
            
            time_titles = []   # Time stamps for each frame
            U_frames = []      # Wind x-component frames
            V_frames = []      # Wind y-component frames
            
            for tkey in time_keys:
                # Create a human-readable timestamp
                d = list(ah["Time"][tkey])
                dt = datetime.datetime(int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]))
                time_titles.append(dt.strftime("%d/%m/%Y %H:%M"))
                
                instant = tkey.split("_")[1]
                
                # Extract vector components
                U_key = f"{variable_vector[0]}_{instant}"
                V_key = f"{variable_vector[1]}_{instant}"
                U_data = np.squeeze(ah["Results"][variable_vector[0]][U_key][:])
                V_data = np.squeeze(ah["Results"][variable_vector[1]][V_key][:])
                if waterpoints_is_3d and U_data.ndim > 2:
                    U_data = U_data[-1, :, :]
                if waterpoints_is_3d and V_data.ndim > 2:
                    V_data = V_data[-1, :, :]
                U_masked = np.where(water_points == 0, np.nan, U_data)
                V_masked = np.where(water_points == 0, np.nan, V_data)
                
                U_frames.append(U_masked)
                V_frames.append(V_masked)
        
# ============================
# CALCULATE THE EXTENT AND ZOOM LEVEL
# ============================
x_min, x_max = np.min(X), np.max(X)
y_min, y_max = np.min(Y), np.max(Y)

dx = (x_max - x_min) / X.shape[0]
dy = (y_max - y_min) / Y.shape[0]

lon_min, lon_max = x_min - extent_cells * dx, x_max + extent_cells * dx
lat_min, lat_max = y_min - extent_cells * dy, y_max + extent_cells * dy
extent = [lon_min, lon_max, lat_min, lat_max]

def calculate_zoom_level(increase_zoom_level):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    range_avg = max(lat_range, lon_range)
    zoom = int(np.log2(360 / range_avg))
    return max(1, min(zoom + increase_zoom_level, 19))

zoom_level = calculate_zoom_level(increase_zoom_level)

# ============================
# PATCH: Satellite Tile Request Function
# ============================
def image_spoof(self, tile):
    url = self._image_url(tile)
    req = Request(url)
    req.add_header("User-agent", "Anaconda 3")
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())
    fh.close()
    img = Image.open(im_data)
    img = img.convert(self.desired_tile_form)
    return img, self.tileextent(tile), "lower"

cimgt.GoogleTiles.get_image = image_spoof

# ============================
# SET-UP FOR ANIMATION
# ============================
global_vmin = min(np.nanmin(frame) for frame in frames_data)
global_vmax = max(np.nanmax(frame) for frame in frames_data)
print("Global Range: ", global_vmin, "to", global_vmax)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(extent)
osm_img = cimgt.GoogleTiles(style="satellite")
ax.add_image(osm_img, zoom_level)


# Plot the first frame using pcolormesh on the boundary grid
quad = ax.pcolormesh(X, Y, frames_data[0], shading="auto",
                     vmin=global_vmin, vmax=global_vmax, cmap="jet",
                     alpha=0.7, zorder=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(quad, cax=cax, orientation="vertical")
cbar.set_label(label, labelpad=25, rotation=270, fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_title(f"{time_titles[0]} UTC", fontsize=18)

# ============================
# COMPUTE CELL-CENTER COORDINATES FOR WIND VECTORS
# ============================
# X and Y are boundary arrays (shape (47, 31)); wind vector fields (U, V) lie at cell centers (e.g. shape (46, 30)).
X_center = (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:]) / 4.0
Y_center = (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:]) / 4.0

# ============================
# DEFINE UPDATE FUNCTION WITH OPTIONS
# ============================
def update(frame_index):
    # Remove previous plot elements (pcolormesh and quiver)
    for coll in list(ax.collections):
        coll.remove()

    # Update the pcolormesh for the current frame
    quad_new = ax.pcolormesh(X, Y, frames_data[frame_index], shading="auto",
                             vmin=global_vmin, vmax=global_vmax, cmap="jet",
                             alpha=transparency_factor, zorder=2)

    # Conditionally overlay wind vectors if enabled
    if show_vectors:
        ax.quiver(
            X_center[::skip_vector, ::skip_vector],
            Y_center[::skip_vector, ::skip_vector],
            U_frames[frame_index][::skip_vector, ::skip_vector],
            V_frames[frame_index][::skip_vector, ::skip_vector],
            color=vector_color, scale=vector_scale, alpha=0.8, zorder=3
        )

    # Set the title appended with "UTC"
    ax.set_title(f"{time_titles[frame_index]} UTC", fontsize=18)
 
    return [quad_new]

# ============================
# CREATE ANIMATION
# ============================

ani = animation.FuncAnimation(fig, update, frames=len(frames_data),
                              interval=500, blit=False)

animation_dir = os.path.join(dir_path, variable)
os.makedirs(animation_dir, exist_ok=True)
    
# EXPORT THE ANIMATION TO MP4
animation_name = os.path.join(animation_dir, f"{variable}.mp4")
ani.save(animation_name, writer=animation.FFMpegWriter(fps=2), dpi=dpi)
print("Animation exported to", animation_name)

# ============================
# OPTIONAL: SAVE INDIVIDUAL IMAGE FRAMES
# ============================
if save_frames:
    for idx in range(len(frames_data)):
        update(idx)  # Update the plot to the current frame
        frame_filename = os.path.join(animation_dir, f"{variable}_{idx:03d}.png")
        plt.savefig(frame_filename, bbox_inches='tight', dpi=dpi)
    print("Image frames saved in", animation_dir)

#plt.close()