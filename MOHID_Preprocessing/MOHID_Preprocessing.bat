@echo off
REM === Path to your Anaconda installation ===
set CONDAPATH=%USERPROFILE%\anaconda3

REM === Name of your environment ===
set ENVNAME=MOHID_Preprocessing_environment

REM === Path to your environment YAML file ===
set ENVFILE=MOHID_Preprocessing_environment.yaml

REM === Copy of the YAML from the last successful install/update (skips the slow update when unchanged) ===
set STAMP=.env_installed.yaml

REM === Activate Conda ===
call "%CONDAPATH%\Scripts\activate.bat"

REM === Check if environment exists ===
conda env list | findstr /C:"%ENVNAME%" >nul
if errorlevel 1 (
    echo Environment %ENVNAME% not found. Creating it...
    call conda env create --file %ENVFILE%
    if not errorlevel 1 copy /y "%ENVFILE%" "%STAMP%" >nul
    call conda activate %ENVNAME%
) else (
    fc /b "%ENVFILE%" "%STAMP%" >nul 2>&1
    if errorlevel 1 (
        echo %ENVFILE% has changed since the last update. Updating %ENVNAME%...
        call conda env update --name %ENVNAME% --file %ENVFILE%
        if not errorlevel 1 copy /y "%ENVFILE%" "%STAMP%" >nul
    ) else (
        echo Environment %ENVNAME% is up to date.
    )
    call conda activate %ENVNAME%
)

REM === Launch Jupyter Lab with your notebook ===
jupyter lab "MOHID_Preprocessing.ipynb"

pause