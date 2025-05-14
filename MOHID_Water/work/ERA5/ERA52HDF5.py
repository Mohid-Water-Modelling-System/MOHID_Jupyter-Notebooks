import importlib
import Input_ERA52HDF5
importlib.reload(Input_ERA52HDF5)
from Input_ERA52HDF5 import *
import os
import datetime
import cdsapi
import zipfile
import shutil
import subprocess
import h5py
import glob

era5_dir = os.getcwd()
print(era5_dir)
# Define the executable path.
exe_path = os.path.join("..","..","releases","ConvertToHdf5","ConvertToHDF5.exe")

# List of input data files for conversion.
input_convert_file_names = [
    "ConvertToHDF5Action_instant.dat",
    "ConvertToHDF5Action_accum.dat",
    "ConvertToHDF5Action_avg.dat"
]
#Merge
# Define directories and file paths
file1 = os.path.join(era5_dir, "Meteo_instant.hdf5")
file2 = os.path.join(era5_dir, "Meteo_accum.hdf5")
file3 = os.path.join(era5_dir, "Meteo_avg.hdf5")
merged_file = os.path.join(era5_dir, "Meteo.hdf5")

# List of HDF5 files to merge
files_to_merge = [file1, file2, file3]

#####################################################
def next_date (run,initial_date):
         
        next_start_date = initial_date + datetime.timedelta(days = run)
        next_end_date = next_start_date + datetime.timedelta(days = 1)
        
        return (next_start_date,next_end_date)
#####################################################

def download_era5(min_lon,max_lon,min_lat,max_lat,next_start_date,next_end_date):

    target = os.path.join(era5_dir, "ERA5.zip")

    # Extract days across months correctly
    days = []
    current_date = next_start_date
    while current_date <= next_end_date:
        days.append(str(current_date.day))
        current_date += datetime.timedelta(days=1)
        
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
                    '10m_u_component_of_wind', 
                    '10m_v_component_of_wind', 
                    '2m_dewpoint_temperature',
                    '2m_temperature', 
                    'boundary_layer_height', 
                    'forecast_albedo',
                    'mean_sea_level_pressure',
                    'mean_surface_downward_short_wave_radiation_flux', 
                    'surface_thermal_radiation_downwards', 
                    'total_cloud_cover',
                    'total_precipitation',
        ],
        "year": sorted([str(next_start_date.year), str(next_end_date.year)]),
        "month": sorted([str(next_start_date.month), str(next_end_date.month)]),
        "day": days,
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [max_lat, min_lon, min_lat, max_lon]
    }

    #target = 'ERA5.zip'
    client = cdsapi.Client()

    try:
        client.retrieve(dataset, request, target)
        print("Download completed successfully!")
    except Exception as e:
        print("An error occurred:", e)

    # Open and extract the zip file
    with zipfile.ZipFile(target, 'r') as zip_ref:
        zip_ref.extractall(era5_dir)

    print(f"Files extracted to {era5_dir}")


#####################################################
def convert_era5():       
#Convert to hdf5
    # Loop over each file in the list.
    for file_name in input_convert_file_names:
        # Construct the full path to the input file.
        src_file = os.path.join(era5_dir, file_name)
        # Define the destination file, which the conversion program expects.
        dst_file = os.path.join(era5_dir, "ConvertToHDF5Action.dat")
        
        try:
            # Copy the input file to the destination file.
            shutil.copy(src_file, dst_file)
            #print(f"Copied {src_file} to {dst_file}.")
        except Exception as e:
            print(f"Error copying {src_file} to {dst_file}: {e}")
            continue  # If there's an error with this file, skip to the next iteration
        
        try:
            # Define the executable path.
            #exe_path = os.path.join(era5_dir, "ConvertToHDF5.exe")
            print(f"Executing {file_name}...")
            # Execute the conversion process.            
            with open(dst_file, 'r') as infile:
                result = subprocess.run([exe_path],
                                        stdin=infile,
                                        capture_output=True,
                                        text=True,
                                        cwd=era5_dir)
                                        
        except Exception as e:
            print(f"Error executing {file_name}: {e}")
#####################################################
def merge_era5():

    with h5py.File(merged_file, "w") as out_file:
        # Iterate through each file to copy and merge groups
        for file_name in files_to_merge:
            with h5py.File(file_name, "r") as in_file:
                # Iterate over all top-level groups in the current file
                for top_group in in_file.keys():
                    if top_group == "Results":
                        # Special handling for "Results": merge subgroups individually.
                        out_results = out_file.require_group("Results")
                        in_results = in_file["Results"]
                        for subgroup in in_results:
                            if subgroup not in out_results:
                                # Copy the subgroup to the merged file under "Results"
                                in_file.copy(in_results[subgroup], out_results, subgroup)
                            #else:
                                #print(f"Subgroup '{subgroup}' under 'Results' already exists. Skipping duplicate.")
                    else:
                        # For any other top-level group, check if it already exists
                        if top_group not in out_file:
                            in_file.copy(in_file[top_group], out_file, top_group)
                        #else:
                            #print(f"Top-level group '{top_group}' already exists. Skipping duplicate.")

    #print("Merging complete while maintaining all groups!")

#####################################################
if forecast == 1:

        start_date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min) + datetime.timedelta(days = refday_to_start)
        runs = number_of_runs
        daily = 1
else:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if daily == 1:
            interval = end_date - start_date
            runs = interval.days  
        else:
            runs = 1
   
for run in range (0, runs):    

        #Update dates
        if daily == 1:
            next_start_date, next_end_date = next_date(run, start_date)
        else:
            next_start_date = start_date
            next_end_date = end_date
        
        hdf_files = glob.iglob(os.path.join(era5_dir,"*.hdf*"))
        nc_files = glob.iglob(os.path.join(era5_dir,"*.nc"))
        
        for filename in hdf_files:
            os.remove(filename)

        for filename in nc_files:
            os.remove(filename)  

        download_era5(min_lon,max_lon,min_lat,max_lat,next_start_date,next_end_date)
            
        convert_era5()
        
        merge_era5()
                           
        output_dir = backup_path+"//"+str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))
            
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        shutil.copy(merged_file, output_dir)
    