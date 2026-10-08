root_A       = r'D:\Aplica\Proj_575_Consulgal_Soyo\Aplica\MOHID_Water\run_cases\Soyo\backup\Level_3_Bat2024'
root_B       = r'D:\Aplica\Proj_575_Consulgal_Soyo\Aplica\MOHID_Water\run_cases\Soyo\backup\Level_3_Bat2024_dragagem'
hdf5_file         =r'InterfaceSedimentWater_2.hdf'
figures_folder    = r'D:\Aplica\Proj_575_Consulgal_Soyo\Aplica\MOHID_Water\run_cases\Soyo\Figures\Level_3_40m_Bat2024_dragagem\MOHID_Postprocessing\out\maps_difference'
start_date_str    = '2025-1-24'
end_date_str      = '2025-1-25'
variable          = 'cohesive sediment'
group             = 'Results'
vmin              = -1
vmax              = 1
extent            = None
map               = 'surface'
nlayer            = 9
label             = 'Cohesive Sediment (kg/m2)'
save_frames       = True
skip_time         = 1
extent_cells      = 1
increase_zoom_level = 3
transparency_factor = 1.0
dpi               = 150
# Diverging colormap: blue = B < A (less), white = no change, red = B > A (more).
# Use symmetric limits (vmin = -vmax) so that zero falls on the white center.
# Other options: 'RdBu' (reversed colors), 'coolwarm', 'BrBG', 'PuOr_r'.
cmap               = 'RdBu_r'
shapefile_path     = r'None'
shapefile_color     = 'black'
shapefile_transparency_factor     = 0.5
fontsize_label     = 16
fontsize_title     = 18
fontsize_tick     = 14

# ----------------------------------------
# GEOTIFF EXPORT
# ----------------------------------------
# export_geotiff = True -> one GeoTIFF per time step in <out_dir>/geotiff with the difference (B - A)
#                          (requires rasterio and a regular lon/lat grid)
# geotiff_include_scenarios = True -> also write scenarios A and B as bands 2 and 3
export_geotiff            = True
geotiff_include_scenarios = False
