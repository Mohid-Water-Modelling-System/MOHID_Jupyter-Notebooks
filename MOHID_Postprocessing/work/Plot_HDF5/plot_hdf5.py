#!/usr/bin/env python3
"""
Generate a pcolormesh animation of a scalar field (and optional
vector field) when these live in separate HDF5 file trees.

Options (see Input_Plot_HDF5.py):
  - vector length scaling: linear, normalized, power or log
  - plot mode: scalar field + single-color vectors, or vectors colored by speed
  - export of all time steps to GeoTIFF (one file per time step),
    CF NetCDF (one file with a time dimension) and GeoPackage points
"""
import importlib
import Input_Plot_HDF5
importlib.reload(Input_Plot_HDF5)
from Input_Plot_HDF5 import *

import os
import re
import glob
import io
import h5py
import numpy as np
import datetime
from urllib.request import Request, urlopen
from PIL import Image

import matplotlib as mpl
mpl.use('Agg')   # non-interactive backend: figures are only saved to file (no Qt needed)
from matplotlib import pyplot as plt, animation
import imageio_ffmpeg

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Point Matplotlib to the ffmpeg executable provided by imageio_ffmpeg
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import geopandas as gpd
from cartopy.feature import ShapelyFeature
from pathlib import Path

# ----------------------------------------
# DEFAULTS FOR OPTIONS MISSING IN THE INPUT FILE
# ----------------------------------------
_defaults = dict(
    group='Results', group_vector='Results',
    map='surface', nlayer=-1,
    vmin=None, vmax=None, extent=None,
    shapefile_path='None', shapefile_color='black', shapefile_transparency_factor=0.5,
    fontsize_label=14, fontsize_title=18, fontsize_tick=12,
    background_style='satellite', fps=2,
    # vectors
    plot_mode='field', vector_length_mode='linear', vector_power=0.5, vector_vref=0.05,
    vector_vmin=None, vector_vmax=None,
    quiverkey_speed=None, quiverkey_pos=(0.75, 0.04), quiverkey_color='white',
    # exports
    export_geotiff=False, export_netcdf=False, export_qgis_points=False,
    qgis_skip_vector=3, nc_u_name='uo', nc_v_name='vo', scalar_units=None,
)
for _k, _v in _defaults.items():
    globals().setdefault(_k, _v)


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
    # If truly 3D, pick surface layer
    if openpoints.ndim == 3:
        openpoints = openpoints[-1, :, :]

    arr = np.squeeze(data)
    # If truly 3D, pick the requested layer
    if arr.ndim == 3:
        if map == "layer":
            arr = arr[nlayer, :, :]
        else:  # map == "surface"
            arr = arr[-1, :, :]
    return np.where(openpoints == 0, np.nan, arr).astype(np.float32)

def vector_length(m):
    """Map velocity magnitude to arrow length according to vector_length_mode."""
    if vector_length_mode == 'normalized':
        return np.where(np.isnan(m), np.nan, 1.0)
    if vector_length_mode == 'power':
        return m ** vector_power
    if vector_length_mode == 'log':
        return np.log1p(m / vector_vref)
    return m  # 'linear'

def prepare_vectors(Uf, Vf):
    """Subsample U, V and rescale arrow length. Returns (Uq, Vq, speed)."""
    Us = Uf[::skip_vector, ::skip_vector]
    Vs = Vf[::skip_vector, ::skip_vector]
    Ms = np.hypot(Us, Vs)
    Mdiv = np.where(Ms > 0, Ms, 1.0)       # avoid division by zero
    Ls = vector_length(Ms)
    return Us / Mdiv * Ls, Vs / Mdiv * Ls, Ms

def regular_grid(lon2d, lat2d, fields):
    """
    Check whether the cell-center grid is a regular lon/lat grid.
    fields are arrays whose LAST TWO axes are the grid (2D or 3D time stacks).
    Returns (is_regular, lon_1d, lat_1d, lon_2d, lat_2d, fields) with longitude
    along the last axis and latitude increasing (south -> north).
    """
    lon_c, lat_c = lon2d, lat2d
    # Make longitude vary along columns
    if np.ptp(lon_c[:, 0]) > np.ptp(lon_c[0, :]):
        lon_c, lat_c = lon_c.T, lat_c.T
        fields = [np.swapaxes(f, -1, -2) for f in fields]
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
        fields = [f[..., ::-1, :] for f in fields]
    return is_regular, lon_1d, lat_1d, lon_c, lat_c, fields


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
# INITIAL GRID
# ----------------------------------------
with h5py.File(scalar_files[0], "r") as h5f:
    X = h5f["Grid"]["Longitude"][:]
    Y = h5f["Grid"]["Latitude"][:]

# cell centers (for vectors and exports)
Xc = (X[:-1,:-1] + X[:-1,1:] + X[1:,:-1] + X[1:,1:]) / 4.0
Yc = (Y[:-1,:-1] + Y[:-1,1:] + Y[1:,:-1] + Y[1:,1:]) / 4.0

# ----------------------------------------
# SYNC TIMES & PREP FRAME CONTAINERS
# ----------------------------------------
all_times   = sorted(scalar_index.keys())
seen_times  = set()
frames_data = []
U_frames    = []
V_frames    = []
frame_times = []
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
        opname = f"OpenPoints_{stkey.split('_')[1]}"
        openpoints = sh["Grid"]["OpenPoints"][opname][:]

        dsname = f"{variable}_{stkey.split('_')[1]}"
        tmp = sh[group][variable][dsname][:]
        scalar_frame = mask_water(tmp, openpoints)

    # read vectors
    if show_vectors and vfile:
        with h5py.File(vfile, "r") as vh:
            opname = f"OpenPoints_{vtkey.split('_')[1]}"
            openpoints = vh["Grid"]["OpenPoints"][opname][:]

            Uds = f"{variable_vector[0]}_{vtkey.split('_')[1]}"
            Vds = f"{variable_vector[1]}_{vtkey.split('_')[1]}"
            Utmp = vh[group_vector][variable_vector[0]][Uds][:]
            Vtmp = vh[group_vector][variable_vector[1]][Vds][:]
        Uf = mask_water(Utmp, openpoints)
        Vf = mask_water(Vtmp, openpoints)
    else:
        Uf, Vf = None, None

    frames_data.append(scalar_frame)
    U_frames.append(Uf); V_frames.append(Vf)
    frame_times.append(dt)
    time_titles.append(dt.strftime("%d/%m/%Y %H:%M"))

has_vectors = show_vectors and any(u is not None for u in U_frames)
if plot_mode == 'colored_vectors' and not has_vectors:
    print("plot_mode = 'colored_vectors' but no vectors were found: using 'field'")
    plot_mode = 'field'

# ----------------------------------------
# OUTPUT FOLDER
# ----------------------------------------
date_span = f"{sd.strftime('%Y%m%d')}_{ed.strftime('%Y%m%d')}"
out_dir   = os.path.join(figures_folder, date_span, variable)
os.makedirs(out_dir, exist_ok=True)
var_safe  = re.sub(r'\W+', '_', variable).strip('_')

# ----------------------------------------
# EXPORT FOR QGIS (GeoTIFF / NetCDF / GeoPackage)
# ----------------------------------------
nt = len(frames_data)
S_all = np.stack(frames_data, axis=0)                       # (nt, ny, nx)
if has_vectors:
    nan_frame = np.full_like(frames_data[0], np.nan)
    U_all = np.stack([u if u is not None else nan_frame for u in U_frames], axis=0)
    V_all = np.stack([v if v is not None else nan_frame for v in V_frames], axis=0)
    M_all = np.hypot(U_all, V_all)
    D_all = np.mod(np.degrees(np.arctan2(U_all, V_all)), 360.0)   # towards, clockwise from north
    fields = [S_all, U_all, V_all, M_all, D_all]
else:
    fields = [S_all]

if scalar_units is None:
    _m = re.search(r'\(([^)]*)\)\s*$', label)
    scalar_units = _m.group(1).strip() if _m else ''

if export_geotiff or export_netcdf:
    is_reg, lon_1d, lat_1d, lon_2d, lat_2d, fields_g = regular_grid(Xc, Yc, fields)

if export_geotiff:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print('rasterio not installed: GeoTIFF export skipped (pip install rasterio)')
    else:
        if not is_reg:
            print('Grid is not a regular lon/lat grid: GeoTIFF export skipped')
        else:
            tif_dir = os.path.join(out_dir, 'geotiff')
            os.makedirs(tif_dir, exist_ok=True)
            res_x = abs(np.diff(lon_1d).mean())
            res_y = abs(np.diff(lat_1d).mean())
            # GeoTIFF rows go from north to south
            transform = from_origin(lon_1d[0] - res_x / 2, lat_1d[-1] + res_y / 2, res_x, res_y)
            names = [var_safe] + (['U', 'V', 'speed'] if has_vectors else [])
            for i, t in enumerate(frame_times):
                bands = [f[i, ::-1, :] for f in fields_g[:len(names)]]
                tif_file = os.path.join(tif_dir, f"{var_safe}_{t:%Y%m%d_%H%M}.tif")
                with rasterio.open(
                    tif_file, 'w', driver='GTiff',
                    height=bands[0].shape[0], width=bands[0].shape[1], count=len(bands),
                    dtype='float32', crs='EPSG:4326', transform=transform, nodata=np.nan
                ) as dst:
                    for b_i, (b, name) in enumerate(zip(bands, names), start=1):
                        dst.write(np.asarray(b, dtype=np.float32), b_i)
                        dst.set_band_description(b_i, name)
                    dst.update_tags(TIFFTAG_DATETIME=f"{t:%Y:%m:%d %H:%M:%S}")
            print(f'GeoTIFFs written: {tif_dir} ({nt} files, bands: {", ".join(names)})')

if export_netcdf:
    try:
        import netCDF4
    except ImportError:
        print('netCDF4 not installed: NetCDF export skipped (pip install netCDF4)')
    else:
        nc_file = os.path.join(out_dir, f"{var_safe}_{date_span}.nc")
        fill = np.float32(-9999.0)
        t0 = frame_times[0]
        t_units = f"seconds since {t0:%Y-%m-%d} 00:00:00"

        with netCDF4.Dataset(nc_file, 'w', format='NETCDF4') as nc:
            # --- global attributes
            nc.Conventions = 'CF-1.8'
            nc.title = f"{variable} {date_span}"
            nc.source = 'MOHID Water - ' + hdf5_file + (
                (' / ' + hdf5_file_vectors) if has_vectors else '')
            nc.history = f'Created {datetime.datetime.now():%Y-%m-%d %H:%M} by plot_hdf5.py'
            nc.comment = 'Layer: ' + (f'{nlayer}' if map == 'layer' else 'surface')
            nc.time_coverage_start = f'{frame_times[0]:%Y-%m-%dT%H:%M:%S}'
            nc.time_coverage_end = f'{frame_times[-1]:%Y-%m-%dT%H:%M:%S}'

            # --- dimensions and coordinates
            # (no time_bnds or other auxiliary arrays: QGIS/MDAL would read them as data grids)
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
            time_v[:] = netCDF4.date2num(frame_times, t_units, 'standard')

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
                if std_name:
                    var.standard_name = std_name
                var.long_name = long_name
                var.units = units
                var.grid_mapping = 'crs'
                if not is_reg:
                    var.coordinates = 'lat lon'
                var[:] = np.where(np.isfinite(data), data, fill).astype(np.float32)

            add_var(var_safe, fields_g[0], None, variable, scalar_units)
            if has_vectors:
                # long_name with "u-component"/"v-component": pattern QGIS/MDAL uses to pair vectors
                add_var(nc_u_name, fields_g[1], 'eastward_sea_water_velocity',
                        'u-component of sea water velocity', 'm s-1')
                add_var(nc_v_name, fields_g[2], 'northward_sea_water_velocity',
                        'v-component of sea water velocity', 'm s-1')
                add_var('speed', fields_g[3], 'sea_water_speed', 'sea water speed', 'm s-1')
                add_var('direction', fields_g[4], 'direction_of_sea_water_velocity',
                        'direction the current flows towards (clockwise from north)', 'degree')
        print(f'NetCDF written: {nc_file} ({nt} time steps)')

if export_qgis_points and has_vectors:
    sk = qgis_skip_vector
    lon_p = Xc[::sk, ::sk].ravel()
    lat_p = Yc[::sk, ::sk].ravel()
    parts = []
    for i, t in enumerate(frame_times):
        u_p = U_all[i, ::sk, ::sk].ravel()
        v_p = V_all[i, ::sk, ::sk].ravel()
        ok = np.isfinite(u_p) & np.isfinite(v_p)
        parts.append(gpd.GeoDataFrame(
            {'time': [t] * int(ok.sum()),
             'U': u_p[ok].astype(float), 'V': v_p[ok].astype(float),
             'speed': np.hypot(u_p[ok], v_p[ok]).astype(float),
             'dir': np.mod(np.degrees(np.arctan2(u_p[ok], v_p[ok])), 360.0).astype(float)},
            geometry=gpd.points_from_xy(lon_p[ok], lat_p[ok]), crs='EPSG:4326'))
    import pandas as pd
    gdf_pts = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    gpkg_file = os.path.join(out_dir, f"{var_safe}_{date_span}_vectors.gpkg")
    gdf_pts.to_file(gpkg_file, layer='vectors', driver='GPKG')
    print(f'Points written: {gpkg_file} ({len(gdf_pts)} features, field "time" for the Temporal Controller)')

# ----------------------------------------
# COMPUTE MAP EXTENT & ZOOM
# ----------------------------------------
if extent is None:
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
# COLOR LIMITS
# ----------------------------------------
if plot_mode == 'colored_vectors':
    # colors refer to vector speed
    cvmin = vector_vmin if vector_vmin is not None else float(np.nanmin(M_all))
    cvmax = vector_vmax if vector_vmax is not None else float(np.nanmax(M_all))
else:
    cvmin = vmin if vmin is not None else float(np.nanmin(S_all))
    cvmax = vmax if vmax is not None else float(np.nanmax(S_all))
norm = plt.Normalize(vmin=cvmin, vmax=cvmax)

# ----------------------------------------
# INITIALIZE FIGURE
# ----------------------------------------
# Read shapefile once (if provided)
p = Path(shapefile_path)
if p.exists():
    gdf = gpd.read_file(shapefile_path)
    shapefile_feature = ShapelyFeature(
        gdf.geometry,
        ccrs.PlateCarree(),
        facecolor=shapefile_color,
        edgecolor=shapefile_color
    )

fig, ax = plt.subplots(
    figsize=(10,10),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
ax.set_extent(extent, crs=ccrs.PlateCarree())
if background_style:
    osm = cimgt.GoogleTiles(style=background_style)
    ax.add_image(osm, zoom_level)

# static layers (drawn once)
if p.exists():
    ax.add_feature(shapefile_feature, zorder=4, linewidth=1, alpha=shapefile_transparency_factor)
    gdf.boundary.plot(ax=ax, color=shapefile_color, linewidth=1, zorder=4)

# colorbar
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
cbar.set_label(label, labelpad=25, rotation=270, fontsize=fontsize_label)
cbar.ax.tick_params(labelsize=fontsize_tick)

Xs = Xc[::skip_vector, ::skip_vector]
Ys = Yc[::skip_vector, ::skip_vector]

# ----------------------------------------
# ANIMATION UPDATE FUNCTION
# ----------------------------------------
dynamic_artists = []

def update(i):
    # remove only the elements drawn in the previous frame
    for a in dynamic_artists:
        a.remove()
    dynamic_artists.clear()

    # scalar field
    if plot_mode == 'field':
        dynamic_artists.append(ax.pcolormesh(
            X, Y, frames_data[i],
            shading="auto", cmap=cmap, norm=norm,
            alpha=transparency_factor, zorder=2,
            transform=ccrs.PlateCarree()
        ))

    # vectors
    if show_vectors and U_frames[i] is not None:
        Uq, Vq, Ms = prepare_vectors(U_frames[i], V_frames[i])
        if plot_mode == 'colored_vectors':
            Q = ax.quiver(Xs, Ys, Uq, Vq, Ms,
                          cmap=cmap, norm=norm, scale=vector_scale,
                          alpha=transparency_factor, zorder=3,
                          transform=ccrs.PlateCarree())
        else:
            Q = ax.quiver(Xs, Ys, Uq, Vq,
                          color=vector_color, scale=vector_scale,
                          alpha=0.8, zorder=3,
                          transform=ccrs.PlateCarree())
        dynamic_artists.append(Q)

        # reference arrow (same length rule as the arrows)
        if quiverkey_speed is not None and vector_length_mode != 'normalized':
            key_len = float(vector_length(np.array(quiverkey_speed)))
            K = ax.quiverkey(Q, quiverkey_pos[0], quiverkey_pos[1], key_len,
                             f"{quiverkey_speed:g} m/s", labelpos='E', coordinates='axes',
                             color=quiverkey_color, labelcolor=quiverkey_color,
                             fontproperties={'size': fontsize_tick})
            dynamic_artists.append(K)

    ax.set_title(f"{time_titles[i]} UTC", fontsize=fontsize_title)
    return dynamic_artists

update(0)

# ----------------------------------------
# BUILD & SAVE ANIMATION
# ----------------------------------------
ani = animation.FuncAnimation(
    fig, update, frames=nt,
    interval=500, blit=False
)

mp4_path = os.path.join(out_dir, f"{variable}.mp4")
ani.save(mp4_path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
print("Animation exported to", mp4_path)

if save_frames:
    for i in range(nt):
        update(i)
        fn = os.path.join(out_dir, f"{variable}_{i:03d}.png")
        plt.savefig(fn, bbox_inches="tight", dpi=dpi)
    print("Image frames saved in", out_dir)
