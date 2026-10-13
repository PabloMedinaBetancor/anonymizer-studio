@echo off
rem Starts Anonymizer Studio from the folder it was installed in. NOT TESTED ON WINDOWS.
setlocal
set "ROOT=%~dp0..\.."
if not defined ANONYMIZER_MODELS set "ANONYMIZER_MODELS=%ROOT%\models"
"%ROOT%\.venv\Scripts\python.exe" -m anonymizer.ui %*
