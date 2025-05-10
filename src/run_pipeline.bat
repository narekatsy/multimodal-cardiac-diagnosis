@echo off
echo Running ECG preprocessing...
python src\ecg_preprocessing.py
if errorlevel 1 exit /b %errorlevel%

echo Running MRI preprocessing...
python src\mri_preprocessing.py
if errorlevel 1 exit /b %errorlevel%

echo Running ECG metadata preprocessing...
python src\ecg_metadata_preprocessing.py
if errorlevel 1 exit /b %errorlevel%

echo Running MRI metadata preprocessing...
python src\mri_metadata_preprocessing.py
if errorlevel 1 exit /b %errorlevel%

echo All scripts completed successfully.
