@echo off
title ASL Sign Language Translator
echo ============================================================
echo         Starting ASL Sign Language Translator...
echo ============================================================

set "VENV_PY=C:\Users\Admin\Desktop\U2\.venv\Scripts\python.exe"
set "SYS_PY=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"

if exist "%VENV_PY%" (
    echo Using Virtual Environment Python...
    "%VENV_PY%" signlanguage.py
    goto end
)

if exist "%SYS_PY%" (
    echo Using System Python 3.12...
    "%SYS_PY%" signlanguage.py
    goto end
)

echo Using standard Python...
python signlanguage.py

:end
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Program exited with an error code.
    pause
)
