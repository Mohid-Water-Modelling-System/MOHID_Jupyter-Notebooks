import importlib
import input_plot_hdf5_residual
importlib.reload(input_plot_hdf5_residual)
from input_plot_hdf5_residual import *

import os
import glob
import h5py
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend: figures are only saved to file (no Qt needed)
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
from datetime import datetime as dt
dt_now = dt.now()

import PIL
#%% Use this to add other background map sources such as OSM or QuadtreeTiles
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

# precompute cell centers for quiver
Xc = (X[:-1,:-1] + X[:-1,1:] + X[1:,:-1] + X[1:,1:]) / 4.0
Yc = (Y[:-1,:-1] + Y[:-1,1:] + Y[1:,:-1] + Y[1:,1:]) / 4.0

# ----------------------------------------
# VECTORS (subsampling + length scaling)
# ----------------------------------------
def vector_length(m):
    """Map velocity magnitude to arrow length according to vector_length_mode."""
    if vector_length_mode == 'normalized':
        return np.where(np.isnan(m), np.nan, 1.0)
    if vector_length_mode == 'power':
        return m ** vector_power
    if vector_length_mode == 'log':
        return np.log1p(m / vector_vref)
    return m  # 'linear'

Xs = Xc[::skip_vector, ::skip_vector]
Ys = Yc[::skip_vector, ::skip_vector]
Us = U[::skip_vector, ::skip_vector]
Vs = V[::skip_vector, ::skip_vector]
Ms = np.hypot(Us, Vs)                      # magnitude at vector points

Mdiv = np.where(Ms > 0, Ms, 1.0)           # avoid division by zero
Ls = vector_length(Ms)                     # transformed arrow length
Uq = Us / Mdiv * Ls                        # keep direction, rescale length
Vq = Vs / Mdiv * Ls

norm = plt.Normalize(vmin=vmin, vmax=vmax)

if plot_mode == 'colored_vectors':
    # Vectors colored by magnitude, without the field colormap
    Q = ax.quiver(
        Xs, Ys, Uq, Vq, Ms,
        cmap=cmap, norm=norm, scale=vector_scale,
        alpha=transparency_factor, zorder=3,
        transform=ccrs.PlateCarree()
    )
    SA = Q
else:  # plot_mode == 'field'
    # Magnitude colormap + single-color vectors
    SA = ax.pcolormesh(X, Y, Z[:,:], norm=norm, cmap=cmap,
                       alpha=transparency_factor, transform=ccrs.PlateCarree())
    Q = ax.quiver(
        Xs, Ys, Uq, Vq,
        color=vector_color, scale=vector_scale,
        alpha=0.8, zorder=3,
        transform=ccrs.PlateCarree()
    )

## Reference arrow (not meaningful when all arrows have the same length)
if quiverkey_speed is not None and vector_length_mode != 'normalized':
    key_len = float(vector_length(np.array(quiverkey_speed)))
    ax.quiverkey(
        Q, quiverkey_pos[0], quiverkey_pos[1], key_len,
        f"{quiverkey_speed:g} m/s", labelpos='E', coordinates='axes',
        color=quiverkey_color, labelcolor=quiverkey_color,
        fontproperties={'size': fontsize_tick}
    )

## Colorbar
cbar = plt.colorbar(SA, ax=ax, shrink=0.75, pad=0.03)
cbar.set_label(label, labelpad=25, rotation=270, fontsize=fontsize_label)
cbar.ax.tick_params(labelsize=fontsize_tick)

## Contour (only if levels are defined)
if len(countour_levels) > 0:
    contour = ax.contour(Xc, Yc, Z[:,:], levels=countour_levels, colors='grey',
                         transform=ccrs.PlateCarree())
    plt.clabel(contour, inline=False, fmt='%2.1f', colors='white', fontsize=18)

if p.exists():
    # add the feature 
    artist = ax.add_feature(shapefile_feature, zorder=4, linewidth=1, alpha=shapefile_transparency_factor)
    gdf.boundary.plot(ax=ax, color=shapefile_color, linewidth=1)
    
os.makedirs(out_dir, exist_ok=True)
#%%
figure_file = os.path.join(out_dir, f"{title}.png")

plt.savefig(figure_file, format='png', dpi=dpi, bbox_inches='tight')
    
#%%
# ----------------------------------------
# EXPORT FOR QGIS
# ----------------------------------------
out_base = os.path.join(out_dir, title.replace(' ', '_'))

if export_qgis_points:
    sk = qgis_skip_vector
    lon_p = Xc[::sk, ::sk].ravel()
    lat_p = Yc[::sk, ::sk].ravel()
    u_p = U[::sk, ::sk].ravel()
    v_p = V[::sk, ::sk].ravel()
    ok = np.isfinite(u_p) & np.isfinite(v_p)
    u_p, v_p, lon_p, lat_p = u_p[ok], v_p[ok], lon_p[ok], lat_p[ok]
    speed_p = np.hypot(u_p, v_p)
    # Azimuth the current flows towards, clockwise from north (QGIS marker rotation convention)
    dir_p = np.mod(np.degrees(np.arctan2(u_p, v_p)), 360.0)

    gdf_pts = gpd.GeoDataFrame(
        {'U': u_p.astype(float), 'V': v_p.astype(float),
         'speed': speed_p.astype(float), 'dir': dir_p.astype(float)},
        geometry=gpd.points_from_xy(lon_p, lat_p),
        crs='EPSG:4326'
    )
    gpkg_file = out_base + '_vectors.gpkg'
    gdf_pts.to_file(gpkg_file, layer='vectors', driver='GPKG')
    print(f'Points written: {gpkg_file} ({len(gdf_pts)} features)')

def regular_grid(lon2d, lat2d, fields):
    """
    Check whether the cell-center grid is a regular lon/lat grid.
    Returns (is_regular, lon_1d, lat_1d, fields) with longitude along axis 1
    and latitude increasing along axis 0 (south -> north).
    """
    lon_c, lat_c = lon2d, lat2d
    # Make longitude vary along columns (axis 1)
    if np.ptp(lon_c[:, 0]) > np.ptp(lon_c[0, :]):
        lon_c, lat_c = lon_c.T, lat_c.T
        fields = [f.T for f in fields]
    lon_1d, lat_1d = lon_c[0, :], lat_c[:, 0]
    dlon, dlat = np.diff(lon_1d), np.diff(lat_1d)
    is_regular = (np.allclose(lon_c, lon_1d[None, :]) and np.allclose(lat_c, lat_1d[:, None])
                  and np.allclose(dlon, dlon.mean(), rtol=1e-3)
                  and np.allclose(dlat, dlat.mean(), rtol=1e-3))
    # Latitude increasing (south -> north)
    if lat_1d[-1] < lat_1d[0]:
        lat_1d = lat_1d[::-1]
        lat_c = lat_c[::-1, :]
        lon_c = lon_c[::-1, :]
        fields = [f[::-1, :] for f in fields]
    return is_regular, lon_1d, lat_1d, lon_c, lat_c, fields

Dir = np.mod(np.degrees(np.arctan2(U, V)), 360.0)   # direction towards, clockwise from north
is_reg, lon_1d, lat_1d, lon_2d, lat_2d, (U_g, V_g, Z_g, D_g) = regular_grid(Xc, Yc, [U, V, Z, Dir])

if export_qgis_raster:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print('rasterio not installed: GeoTIFF export skipped (pip install rasterio)')
    else:
        if not is_reg:
            print('Grid is not a regular lon/lat grid: GeoTIFF export skipped (use the GeoPackage points)')
        else:
            # GeoTIFF rows must go from north to south
            bands = [b[::-1, :] for b in (U_g, V_g, Z_g)]
            lat_top = lat_1d[::-1]
            res_x = abs(np.diff(lon_1d).mean())
            res_y = abs(np.diff(lat_1d).mean())
            transform = from_origin(lon_1d[0] - res_x / 2, lat_top[0] + res_y / 2, res_x, res_y)
            tif_file = out_base + '_UV.tif'
            with rasterio.open(
                tif_file, 'w', driver='GTiff',
                height=bands[0].shape[0], width=bands[0].shape[1], count=3,
                dtype='float32', crs='EPSG:4326', transform=transform, nodata=np.nan
            ) as dst:
                for i, (b, name) in enumerate(zip(bands, ['U', 'V', 'speed']), start=1):
                    dst.write(np.asarray(b, dtype=np.float32), i)
                    dst.set_band_description(i, name)
            print(f'Raster written: {tif_file}')

if export_netcdf:
    try:
        import netCDF4
    except ImportError:
        print('netCDF4 not installed: NetCDF export skipped (pip install netCDF4)')
    else:
        nc_file = out_base + '_UV.nc'
        fill = np.float32(-9999.0)
        t0, t1 = all_times[0], all_times[-1]          # averaging period
        t_ref = datetime.datetime(t0.year, t0.month, t0.day)
        t_units = f"seconds since {t_ref:%Y-%m-%d %H:%M:%S}"
        t_mid = t0 + (t1 - t0) / 2

        with netCDF4.Dataset(nc_file, 'w', format='NETCDF4') as nc:
            # --- global attributes
            nc.Conventions = 'CF-1.8'
            nc.title = title
            nc.source = 'MOHID Water - ' + hdf5_file_vectors
            nc.history = f'Created {dt_now:%Y-%m-%d %H:%M} by plot_hdf5_residual.py'
            nc.comment = (f'Time-mean of {len(U_frames)} outputs between {t0:%Y-%m-%d %H:%M} '
                          f'and {t1:%Y-%m-%d %H:%M}; layer: '
                          + (f'{nlayer}' if mean_map == 'layer' else 'surface'))
            nc.time_coverage_start = f'{t0:%Y-%m-%dT%H:%M:%S}'
            nc.time_coverage_end = f'{t1:%Y-%m-%dT%H:%M:%S}'

            # --- dimensions and coordinates
            nc.createDimension('time', None)
            if is_reg:
                nc.createDimension('lat', len(lat_1d))
                nc.createDimension('lon', len(lon_1d))
                lat_v = nc.createVariable('lat', 'f8', ('lat',))
                lon_v = nc.createVariable('lon', 'f8', ('lon',))
                lat_v[:] = lat_1d
                lon_v[:] = lon_1d
                lat_v.axis, lon_v.axis = 'Y', 'X'
                hdims = ('lat', 'lon')
            else:
                nc.createDimension('y', lat_2d.shape[0])
                nc.createDimension('x', lat_2d.shape[1])
                lat_v = nc.createVariable('lat', 'f8', ('y', 'x'))
                lon_v = nc.createVariable('lon', 'f8', ('y', 'x'))
                lat_v[:] = lat_2d
                lon_v[:] = lon_2d
                hdims = ('y', 'x')
                print('Grid is curvilinear: NetCDF written with 2D lat/lon '
                      '(QGIS mesh layers may not open it; check before using)')
            lat_v.standard_name, lat_v.long_name, lat_v.units = 'latitude', 'latitude', 'degrees_north'
            lon_v.standard_name, lon_v.long_name, lon_v.units = 'longitude', 'longitude', 'degrees_east'

            time_v = nc.createVariable('time', 'f8', ('time',))
            time_v.standard_name = 'time'
            time_v.units = t_units
            time_v.calendar = 'standard'
            time_v.axis = 'T'
            time_v[:] = netCDF4.date2num([t_mid], t_units, 'standard')
            # No time_bnds variable: QGIS/MDAL treats it as a data grid and then ignores
            # the real variables. The averaging period is stored in global attributes.

            crs_v = nc.createVariable('crs', 'i4')
            crs_v.grid_mapping_name = 'latitude_longitude'
            crs_v.longitude_of_prime_meridian = 0.0
            crs_v.semi_major_axis = 6378137.0
            crs_v.inverse_flattening = 298.257223563
            crs_v.crs_wkt = 'EPSG:4326'

            # --- data variables
            def add_var(name, data, std_name, long_name, units):
                var = nc.createVariable(name, 'f4', ('time',) + hdims,
                                        fill_value=fill, zlib=True, complevel=4)
                var.standard_name = std_name
                var.long_name = long_name
                var.units = units
                var.grid_mapping = 'crs'
                var.cell_methods = 'time: mean'
                if not is_reg:
                    var.coordinates = 'lat lon'
                arr = np.where(np.isfinite(data), data, fill).astype(np.float32)
                var[0, :, :] = arr

            add_var(nc_u_name, U_g, 'eastward_sea_water_velocity',
                    'u-component of sea water velocity', 'm s-1')
            add_var(nc_v_name, V_g, 'northward_sea_water_velocity',
                    'v-component of sea water velocity', 'm s-1')
            add_var('speed', Z_g, 'sea_water_speed', 'sea water speed', 'm s-1')
            add_var('direction', D_g, 'direction_of_sea_water_velocity',
                    'direction the current flows towards (clockwise from north)', 'degree')
        print(f'NetCDF written: {nc_file}')
