import importlib
import Input_XMART
importlib.reload(Input_XMART)
from Input_XMART import *
import re, datetime, time
import glob, os, shutil
import subprocess, sys
from ftplib import FTP
import requests
import logging

mohid_log = (exe_dir+"//mohid.log")

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
                    
#To be considered later
convert2netcdf = 0

f_min_meteo = 100
f_min_hydro = 100
f_min_wp = 100


#number_of_meteo = 1
#number_of_hydro = 1
number_of_wp = 0
rivers = 0

send_ftp = 0
telegram_messages = 0
model_name = ""

timeseries_backup = 0
#####################################################
def next_date (run,initial_date):
         
        next_start_date = initial_date + datetime.timedelta(days = run)
        next_end_date = next_start_date + datetime.timedelta(days = 1)
        
        return (next_start_date,next_end_date)

#####################################################
def write_date(file_name):
        
    with open(file_name) as file:
        file_lines = file.readlines()
        
    number_of_lines = len(file_lines)
    
    for n in range(0,number_of_lines):
        line = file_lines[n]        
        if re.search("^START.+:", line):
            file_lines[n] = "START " + ": " + str(next_start_date.strftime("%Y %m %d ")) + "0 0 0\n"

        elif re.search("^END.+:", line):    
            file_lines[n] = "END " + ": " + str(next_end_date.strftime("%Y %m %d ")) + "0 0 0\n"
            
    with open(file_name,"w") as file:
        for n in range(0,number_of_lines) :
            file.write(file_lines[n])

#####################################################
def copy_initial_files():
    """
    Copies initial '.fin*' files from the backup directory to the results directory.
    
    Process:
      1. Searches for a directory in backup_dir that matches the pattern "*_YYYYMMDD",
         with YYYYMMDD taken from old_end_date.
      2. Clears any existing '.fin*' files in the destination directory.
      3. Copies '.fin*' files from the matching backup directory to the destination.
      4. Renames files in the destination ending with '_2.fin*' to use '_1.fin*' instead.

    """


    # Format the date string to match the naming convention.
    date_str = old_end_date.strftime("%Y%m%d")

    # Build a source directory pattern.
    # Note: This creates a glob pattern that matches any directory name ending with '_YYYYMMDD'
    source_dir = os.path.join(backup_dir, "*_" + date_str)

    dest_dir = results_dir

    # Remove existing '.fin*' files in the destination directory.
    dest_fin_files = glob.glob(os.path.join(dest_dir, "*.fin*"))
    for filepath in dest_fin_files:
        os.remove(filepath)

    # Initialize an empty list to hold the source '.fin*' files we want to copy.
    source_fin_files = []

    # Try to match the pattern '*_2.fin*' in the source directory pattern.
    fin2 = glob.iglob(os.path.join(source_dir, "*_2.fin*"))
    try:
        first_item = next(fin2)  # Ensure at least one file exists
        # Combine the first file with the rest of the iterator results.
        source_fin_files = [first_item] + list(fin2)
    except StopIteration:
        # If no '*_2.fin*' files are found, try matching the '*_1.fin*' pattern.
        fin1 = glob.iglob(os.path.join(source_dir, "*_1.fin*"))
        try:
            first_item = next(fin1)
            source_fin_files = [first_item] + list(fin1)
        except StopIteration:
            print("No initial files found. Define continuos = 0.")

    # Only proceed with copying if found any files.
    if source_fin_files:
        for file in source_fin_files:
            # Confirm that the found item is actually a file.
            if os.path.isfile(file):
                try:
                    shutil.copy(file, dest_dir)
                    print(f"Copied {file} to {dest_dir}")
                except Exception as e:
                    print(f"Error copying {file}: {e}")

    # Rename files in the destination directory from '_2.fin' to '_1.fin'.
    dest_two_fin_files = glob.iglob(os.path.join(dest_dir, "*_2.fin*"))
    for file in dest_two_fin_files:
        if os.path.isfile(file):
            new_name = file.replace("_2.fin", "_1.fin")
            try:
                os.rename(file, new_name)
                print(f"Renamed {file} to {new_name}")
            except Exception as e:
                print(f"Error renaming {file}: {e}")

#####################################################

def backup():
    # Construct backup directory name using os.path.join() and appropriate formatting
    backup_dir_date = os.path.join(
        backup_dir,
        f"{next_start_date.strftime('%Y%m%d')}_{next_end_date.strftime('%Y%m%d')}"
    )
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir_date):
        os.makedirs(backup_dir_date)
        print(f"Created backup directory: {backup_dir_date}")
    
    # Remove all MPI*.* files in the results_dir using absolute paths
    mpi_files = glob.glob(os.path.join(results_dir, "MPI*.*"))
    for file in mpi_files:
        os.remove(file)

    # Process '*.hdf*' files: copy them to backup and then remove them from results_dir
    hdf_files = glob.iglob(os.path.join(results_dir, "*.hdf*"))
    for file in hdf_files:
        shutil.copy(file, backup_dir_date)
        print(f"Copied HDF file: {file} to {backup_dir_date}")
        os.remove(file)
    
    # Determine the pattern based on the 'continuous' flag
    fin_pattern = "*_1.fin*" if continuous == 0 else "*_2.fin*"
    fin_files = glob.iglob(os.path.join(results_dir, fin_pattern))
    
    # Copy the designated '.fin*' files to the backup directory
    for file in fin_files:
        shutil.copy(file, backup_dir_date)
        print(f"Copied fin file: {file} to {backup_dir_date}")

    # Remove all '*.fin*' files from results_dir
    all_fin_files = glob.iglob(os.path.join(results_dir, "*.fin*"))
    for file in all_fin_files:
        os.remove(file)
   
    # If timeseries backup is enabled, copy files from the appropriate run directory
    if timeseries_backup == 1:
        # Use os.path.join to create the timeseries directory
        run_subdir = "run1" if continuous == 0 else "run2"
        timeseries_dir = os.path.join(results_dir, run_subdir)
        ts_files = glob.iglob(os.path.join(timeseries_dir, "*.*"))
        for file in ts_files:
            shutil.copy(file, backup_dir_date)
            print(f"Copied timeseries file: {file} to {backup_dir_date}")
#####################################################
def convert(date, hdf_file):
        
    convert2netcdf_file = convert2netcdf_dir + "//Convert2netcdf.dat"
    
    with open(convert2netcdf_file) as file:
        file_lines = file.readlines()
        
    number_of_lines = len(file_lines)
    
    for n in range(0,number_of_lines):
        line = file_lines[n]        
        if re.search("^HDF_FILE.+:", line):
            backup_dir_date = (backup_dir+"\\" + date)
            file_lines[n] = "HDF_FILE " + ": " + backup_dir_date + "\\" + hdf_file + "\n"

        elif re.search("^NETCDF_FILE.+:", line):    
            file_lines[n] = "NETCDF_FILE " + ": " + backup_dir_date + "\\" + hdf_file + ".nc\n"
            
        elif re.search("^REFERENCE_TIME.+:", line):
            file_lines[n] = "REFERENCE_TIME " + ": " + str(next_start_date.strftime("%Y %m %d ")) + "0 0 0\n"
            
    with open(convert2netcdf_file,"w") as file:
        for n in range(0,number_of_lines) :
            file.write(file_lines[n])
    
    os.chdir(convert2netcdf_dir)
    output = subprocess.call(["Convert2netcdf.exe"])


#####################################################
#Funcao para envio de mensagem pelo Bot do Telegram
def telegram_msg(message):
        if telegram_messages == 1:
                #message = "hello from your telegram bot"
                urlbot = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
                print(requests.get(urlbot).json()) # this sends the message
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
        
for run in range (0,runs):    

    #Update dates
    if daily == 1:
        next_start_date, next_end_date = next_date(run, start_date)
    else:
        next_start_date = start_date
        next_end_date = end_date
    
    #Pre-processing
    pattern = os.path.join(boundary_conditions_dir, "*.hdf*")
    files = glob.glob(pattern)
    for filename in files:
        os.remove(filename)
                
    #Copy Meteo boundary conditions
    f_missing = True


    f_meteo = (dir_meteo+"//"+str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))+"//"+file_name_meteo)

    f_exists = os.path.exists(f_meteo)
    
    if f_exists:
        f_size = os.path.getsize(f_meteo)
    
        if f_size > f_min_meteo:

            destination_file = os.path.join(boundary_conditions_dir, "meteo.hdf5")
            shutil.copy(f_meteo, destination_file)
            print("Get meteo from: " + f_meteo)
            f_missing = False

        
    if f_missing == True:
        msg = "Message from XMART: model " + model_name + "\nMeteo file is missing or is too small for " + str(next_start_date.strftime("%Y%m%d"))
        telegram_msg(msg)
        sys.exit (msg)
        
    #Copy ocean boundary conditions
    #Hydrodynamics

    f_hydro = (dir_hydro+"//"+str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))+"//"+file_hydro)
    f_exists = os.path.exists(f_hydro)

    if f_exists:
        f_size = os.path.getsize(f_hydro)
        if f_size > f_min_hydro:
            destination_file = os.path.join(boundary_conditions_dir, "ocean.hdf5")
            shutil.copy(f_hydro, destination_file)
            print("Get oceanic boundary conditions from: " + f_hydro)
        else:
            msg = "Message from XMART: model " + model_name + "\nHydrodynamic BC file is too small for " + str(next_start_date.strftime("%Y%m%d"))
            telegram_msg(msg)
            sys.exit (msg)
    else:
        msg = "Message from XMART: model " + model_name + "\nHydrodynamic BC file is missing for " + str(next_start_date.strftime("%Y%m%d"))
        telegram_msg(msg)
        sys.exit (msg)
    
    #Water properties
    if number_of_wp > 0:
        for n in range(0, number_of_wp):
            f_wp = (dir_wp[n]+"//"+str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))+"//"+file_wp[n])
            f_exists = os.path.exists(f_wp)
            
            if f_exists:
                f_size = os.path.getsize(f_wp)
                if f_size > f_min_wp:
                    shutil.copy(f_wp, boundary_conditions_dir)
                else:
                    msg = "Message from XMART: model " + model_name + "\nWater Properties BC file is too small for " + str(next_start_date.strftime("%Y%m%d"))
                    telegram_msg(msg)
                    sys.exit (msg)
            else:
                msg = "Message from XMART: model " + model_name + "\nWater Properties BC file is missing for " + str(next_start_date.strftime("%Y%m%d"))
                telegram_msg(msg)
                sys.exit (msg)
            
    #Copy rivers boundary conditions
    if rivers == 1:    
    
        river_files = glob.iglob(os.path.join(dir_rivers_average,"*.dat"))
        for file in river_files:
            shutil.copy(file, boundary_conditions_dir)
            
        if forecast_mode == 1:
            dir_rivers_date = (dir_rivers_forecast+"//"+str(initial_date.strftime("%Y%m%d")))
        
        else:
            dir_rivers_date = (dir_rivers_forecast+"//"+str(next_start_date.strftime("%Y%m%d")))
        
        f_exists = os.path.exists(dir_rivers_date)
        if f_exists:
            river_files = glob.iglob(os.path.join(dir_rivers_date,"*.dat"))
            for file in river_files:
                shutil.copy(file, boundary_conditions_dir)
        
    ##############################################
    #MOHID
    
    #Update dates
    if continuous == 0:
        model = os.path.join(data_dir, "Model_1.dat")
    else:
        model = os.path.join(data_dir, "Model_2.dat")

    write_date(model)
    
    #Copy initial files (.fin)
    #old_start_date = next_start_date - datetime.timedelta(days = 1)
    old_end_date = next_end_date - datetime.timedelta(days = 1)
    

    if continuous == 1:
        copy_initial_files()            

    #Run
    os.chdir(exe_dir)
    
    with open("Tree.dat","w") as file:
        file.write("Automatic Generated Tree File\n")
        file.write("by MOHID_Jupyter-Notebooks\n")
        file.write(rf"+..\exe : {np}")

    with open("run.bat","w") as file:
        # Write the mpiexec command. The inner quotes around {release} ensure the file path is enclosed in quotes.
        file.write(rf'mpiexec -np {np} "{release}" >> mohid.log 2>&1' + "\n")
        # Write the DomainConsolidation command, also enclosed in quotes.
        file.write(rf'"{DomainConsolidation}"')

    os.remove("mohid.log")   
    output = subprocess.call([exe_dir+"/run.bat"])
    #output = subprocess.call([exe_dir+"/run.sh"])
    
    if not ("Program Mohid Water successfully terminated") in open(mohid_log, encoding='latin-1').read():
        msg = "Message from XMART: model " + model_name + "\nProgram Mohid Water was not successfully terminated for " + str(next_start_date.strftime("%Y%m%d"))
        telegram_msg(msg)
        sys.exit (msg)
        
    #Backup    
    backup()
    
    date = str(next_start_date.strftime("%Y%m%d")) + "_" + str(next_end_date.strftime("%Y%m%d"))
    
    if convert2netcdf == 1:
    
        for file in range (0, len(convert_list)):
            convert(date, convert_list[file])

    #Send ftp
    if send_ftp == 1:
        
        ftp=FTP(server)
        ftp.login(user,password)
        ftp.cwd(cwd)
        
        if not date in ftp.nlst():
            ftp.mkd(date)
            
        ftp.cwd(date)
        
        backup_dir_date = (backup_dir+"\\" + date)
        os.chdir(backup_dir_date)
        
        for file in range (0, len(ftp_list)):
            #ftp.set_pasv(False)
            ftp.storbinary('STOR '+ftp_list[file],open(ftp_list[file],'rb'))

        ftp.quit()

