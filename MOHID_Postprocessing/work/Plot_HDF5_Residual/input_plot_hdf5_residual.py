backup_root       = r'D:\Aplica\Proj_575_Consulgal_Soyo\Aplica\MOHID_Water\run_cases\Soyo\backup\Level_3_Bat2024'
hdf5_file_vectors=r'Hydrodynamic_2.hdf5'
out_dir           = r'D:\Aplica\Proj_575_Consulgal_Soyo\Aplica\MOHID_Water\run_cases\Soyo\Figures\Level_3_40m_Bat2024\MOHID_Postprocessing\out\residual'
start_date_str    = '2025-1-11'
end_date_str      = '2025-1-25'
label             = 'Mean Velocity (m/s)'
countour_levels   = []
vmin              = 0.0
vmax              = 1.0
mean_map          = 'surface'
nlayer            = 10
extent_cells      = 1
increase_zoom_level = 3
transparency_factor = 1.0
dpi               = 150
cmap              = 'jet'
skip_vector       = 3
vector_scale      = 10
vector_color      = 'white'
variable_vector   = ['velocity U', 'velocity V']
title             = 'Mean velocity at surface layer'
extent             = None
shapefile_path     = r'None'
shapefile_color     = 'black'
shapefile_transparency_factor     = 0.5
fontsize_label     = 20
fontsize_title     = 18
fontsize_tick     = 20

# ----------------------------------------
# VECTOR OPTIONS
# ----------------------------------------
# vector_length_mode controls how arrow length relates to velocity magnitude |v|:
#   'linear'     -> length proportional to |v| (original behaviour)
#   'normalized' -> all arrows have the same length (direction only)
#   'power'      -> length proportional to |v|**vector_power (0 < vector_power < 1 compresses the range;
#                   0.5 = square root, smaller values compress more)
#   'log'        -> length proportional to log(1 + |v|/vector_vref) (strong compression of high velocities)
vector_length_mode = 'power'
vector_power       = 0.5
vector_vref        = 0.05   # reference velocity (m/s) for 'log' mode: below it ~linear, above it ~logarithmic

# Reference arrow (quiverkey). Set to None to disable.
# Its length is transformed with the same rule as the arrows, so it stays consistent.
quiverkey_speed    = 0.5    # m/s
quiverkey_pos      = (0.85, 0.05)   # position in axes coordinates (x, y)
quiverkey_color    = 'white'

# plot_mode = 'field'           -> magnitude colormap (pcolormesh) + single-color vectors (vector_color)
# plot_mode = 'colored_vectors' -> no magnitude map; vectors are colored by velocity (uses cmap, vmin, vmax)
plot_mode         = 'field'
