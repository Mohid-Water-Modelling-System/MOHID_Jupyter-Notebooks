@echo off
REM === Path to your Anaconda installation ===
set CONDAPATH=%USERPROFILE%\anaconda3

REM === Name of your environment ===
set ENVNAME=MOHID_Water_environment

REM === Path to your environment YAML file ===
set ENVFILE=MOHID_Water_environment.yaml

REM === Activate Conda ===
call "%CONDAPATH%\Scripts\activate.bat"

REM === Check if environment exists ===
conda env list | findstr /C:"%ENVNAME%" >nul
if errorlevel 1 (
    echo Environment %ENVNAME% not found. Creating it...
    conda env create --file %ENVFILE%
    call conda activate %ENVNAME%
) else (
    echo Environment %ENVNAME% already exists.
    call conda activate %ENVNAME%
)

REM === Launch Jupyter Lab with your notebook ===
jupyter lab "MOHID_Water.ipynb"

pause