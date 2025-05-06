import os
import datetime
from Input_CMEMS2HDF5 import *
import copernicusmarine
import shutil
import subprocess
import h5py
import glob

#cmems_dir = os.path.join(os.getcwd(),"work/CMEMS")
cmems_dir = os.getcwd()
print(cmems_dir)
# Define the executable path.
#exe_path = os.path.join(os.getcwd(),"releases/ConvertToHdf5", "ConvertToHDF5.exe")
exe_path = os.path.join("..","..","releases","ConvertToHdf5","ConvertToHDF5.exe")

# Mapping each product into its specific subset parameters:
product_parameters = {
    "cmems_mod_glo_phy_anfc_0.083deg_PT6H-i": {'variables': ['zos'], 'filename': "CMEMS_zos.nc"},
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m": {'variables': ['zos'], 'filename': "CMEMS_zos.nc"},
    "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i": {'variables': ['uo', 'vo'], 'filename': "CMEMS_cur.nc"},
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m": {'variables': ['uo', 'vo'], 'filename': "CMEMS_cur.nc"},
    "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i": {'variables': ['so'], 'filename': "CMEMS_so.nc"},
    "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m": {'variables': ['so'], 'filename': "CMEMS_so.nc"},
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i": {'variables': ['thetao'], 'filename': "CMEMS_thetao.nc"},
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m": {'variables': ['thetao'], 'filename': "CMEMS_thetao.nc"}
}

# List of input data files for conversion.
input_convert_file_names = [
    "ConvertToHDF5Action_zos.dat",
    "ConvertToHDF5Action_cur.dat",
    "ConvertToHDF5Action_so.dat",
    "ConvertToHDF5Action_thetao.dat"
]

#Merge
# Define directories and file paths
file1 = os.path.join(cmems_dir, "CMEMS_cur.hdf5")
file2 = os.path.join(cmems_dir, "CMEMS_so.hdf5")
file3 = os.path.join(cmems_dir, "CMEMS_zos.hdf5")
file4 = os.path.join(cmems_dir, "CMEMS_thetao.hdf5")
merged_file = os.path.join(cmems_dir, "CMEMS.hdf5")

# List of HDF5 files to merge
files_to_merge = [file1, file2, file3, file4]

#####################################################
def next_date (run,initial_date):
         
        next_start_date = initial_date + datetime.timedelta(days = run)
        next_end_date = next_start_date + datetime.timedelta(days = 1)
        
        return (next_start_date,next_end_date)
#####################################################

def download_cmems(product,start_depth,end_depth,min_lon,max_lon,min_lat,max_lat,next_start_date,next_end_date):
    """
    Download a data file for a given product between start_date and end_date.
    
    The start_date is adjusted by subtracting one day, which is a domain-specific
    requirement to ensure compatibility with Mohid.
    
    Parameters:
        product (str): The product identifier from Copernicus Marine Service.
        start_date (datetime): The original start date.
        end_date (datetime): The end date.
    """
    # Subtract one day from the start date.
    adj_start_date = next_start_date - datetime.timedelta(days=1)
    
    # Retrieve mapping parameters for the product.
    if product not in product_parameters:
        print(f"Product {product} not recognized. Skipping download.")
        return

    params = product_parameters[product]
    variable = params['variables']
    output_file_name = params['filename']

    print(f"Downloading: {output_file_name} for {next_start_date.strftime('%Y-%m-%d')} to {next_end_date.strftime('%Y-%m-%d')}")

    # Call the subset function and handle potential exceptions.
    try:
        copernicusmarine.subset(
            dataset_id=product,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_depth=start_depth,
            maximum_depth=end_depth,
            start_datetime=adj_start_date.strftime('%Y-%m-%d') + ' 00:00:00',
            end_datetime=next_end_date.strftime('%Y-%m-%d') + ' 00:00:00',
            variables=variable,
            output_directory=cmems_dir,
            output_filename=output_file_name,
            netcdf3_compatible = True
        )
        print(f"Download successful: {output_file_name}")
    except Exception as e:
        print(f"Error downloading {output_file_name}: {e}")

#####################################################

def convert_cmems():       
#Convert to hdf5

    # Loop over each file in the list.
    for file_name in input_convert_file_names:
        # Construct the full path to the input file.
        src_file = os.path.join(cmems_dir, file_name)
        # Define the destination file, which the conversion program expects.
        dst_file = os.path.join(cmems_dir, "ConvertToHDF5Action.dat")
        
        try:
            # Copy the input file to the destination file.
            shutil.copy(src_file, dst_file)
            #print(f"Copied {src_file} to {dst_file}.")
        except Exception as e:
            #print(f"Error copying {src_file} to {dst_file}: {e}")
            continue  # If there's an error with this file, skip to the next iteration
        
        try:
            print(f"Executing {file_name}...")
            # Execute the conversion process.
            
            with open(dst_file, 'r') as infile:
                result = subprocess.run([exe_path],
                                        stdin=infile,
                                        capture_output=True,
                                        text=True,
                                        cwd=cmems_dir)
        except Exception as e:
            print(f"Error executing {file_name}: {e}")
#####################################################
       
def merge_cmems():

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
#def main_cmems(daily,forecast,number_of_runs,refday_to_start,backup_path,product_id,start_depth,end_depth,min_lon,max_lon,min_lat,max_lat,start_date,end_date):

start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

if forecast == 1:

        initial_date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min) + datetime.timedelta(days = refday_to_start)
        
else:
        interval = end_date - start_date
        number_of_runs = interval.days
        initial_date = datetime.datetime.combine(start_date, datetime.time.min)
        
for run in range (0,number_of_runs):    

        #Update dates
        if daily == 1:
            next_start_date, next_end_date = next_date(run, initial_date)
        else:
            next_start_date = start_date
            next_end_date = end_date
        
        hdf_files = glob.iglob(os.path.join(cmems_dir,"*.hdf*"))
        nc_files = glob.iglob(os.path.join(cmems_dir,"*.nc"))
        
        for filename in hdf_files:
            os.remove(filename)

        for filename in nc_files:
            os.remove(filename)  

        # Process each product.
        for product in product_id:
            download_cmems(product,start_depth,end_depth,min_lon,max_lon,min_lat,max_lat,next_start_date,next_end_date)
            
        convert_cmems()
        
        merge_cmems()
                           
        output_dir = backup_path+"//"+str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))
            
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        shutil.copy(merged_file, output_dir)
    