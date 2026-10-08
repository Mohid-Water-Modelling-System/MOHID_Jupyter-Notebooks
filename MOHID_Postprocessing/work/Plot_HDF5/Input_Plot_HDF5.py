backup_root       = r'C:\Users\aquaf\OneDrive\Projetos\Aquaflow\Maretec\Jupyter Notebooks\MOHID_Postprocessing\res'
hdf5_file         =r'Hydrodynamic_2_Surface.hdf5'
hdf5_file_vectors=r'Hydrodynamic_2_Surface.hdf5'
figures_folder    = r'C:\Users\aquaf\OneDrive\Projetos\Aquaflow\Maretec\Jupyter Notebooks\MOHID_Postprocessing\out\maps'
start_date_str    = '2025-9-25'
end_date_str      = '2025-9-27'
group             = 'Results'
variable          = 'velocity modulus'
label             = 'Velocity Modulus(m/s)'
group_vector      = 'Results'
variable_vector   = ['velocity U', 'velocity V']
show_vectors      = True
save_frames       = True
skip_time         = 3
map               = 'surface'   # 'surface' or 'layer' (for 3D results)
nlayer            = -1          # layer index used when map = 'layer'
vmin              = None        # None -> computed from the data
vmax              = None
extent            = None        # [lon_min, lon_max, lat_min, lat_max] or None (whole grid)
extent_cells      = 1
increase_zoom_level = 1
background_style  = 'satellite' # Google tiles style ('satellite', 'street', ...) or None for no background
skip_vector       = 5
vector_scale      = 10
vector_color      = 'white'
transparency_factor = 1.0
dpi               = 150
fps               = 2
cmap               = 'jet'
fontsize_label    = 14
fontsize_title    = 18
fontsize_tick     = 12
shapefile_path    = r'None'
shapefile_color   = 'black'
shapefile_transparency_factor = 0.5

# ----------------------------------------
# VECTOR OPTIONS
# ----------------------------------------
# plot_mode = 'field'           -> scalar colormap (pcolormesh) + single-color vectors (vector_color)
# plot_mode = 'colored_vectors' -> no scalar map; vectors are colored by speed (uses cmap, vector_vmin, vector_vmax)
plot_mode         = 'field'
vector_vmin       = None        # color limits for 'colored_vectors' (None -> computed from the data)
vector_vmax       = None

# vector_length_mode controls how arrow length relates to velocity magnitude |v|:
#   'linear'     -> length proportional to |v| (original behaviour)
#   'normalized' -> all arrows have the same length (direction only)
#   'power'      -> length proportional to |v|**vector_power (0 < vector_power < 1 compresses the range;
#                   0.5 = square root, smaller values compress more)
#   'log'        -> length proportional to log(1 + |v|/vector_vref) (strong compression of high velocities)
vector_length_mode = 'linear'
vector_power       = 0.5
vector_vref        = 0.05   # reference velocity (m/s) for 'log' mode

# Reference arrow (quiverkey). Set to None to disable.
# Its length is transformed with the same rule as the arrows, so it stays consistent.
quiverkey_speed    = 0.5    # m/s
quiverkey_pos      = (0.75, 0.04)   # position in axes coordinates (x, y)
quiverkey_color    = 'white'

# ----------------------------------------
# QGIS EXPORT (all time steps)
# ----------------------------------------
# export_geotiff     = True -> one GeoTIFF per time step in <out_dir>/geotiff, bands: scalar [, U, V, speed]
#                              (requires rasterio and a regular lon/lat grid)
# export_netcdf      = True -> one CF-1.8 NetCDF with a time dimension: scalar [, U, V, speed, direction]
#                              (requires netCDF4)
# export_qgis_points = True -> GeoPackage of vector points with a "time" field (can be large)
export_geotiff     = True
export_netcdf      = True
export_qgis_points = False
qgis_skip_vector   = 3          # subsampling for the point layer (1 = every cell)
scalar_units       = None       # units of the scalar in the NetCDF; None -> taken from label "(...)"
# Variable names of the U and V components in the NetCDF
nc_u_name          = 'uo'
nc_v_name          = 'vo'
