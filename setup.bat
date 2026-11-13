@echo off
rem ==========================================================================================
rem  SIMGuard AI - one-command setup and start (Windows)
rem
rem    setup.bat               set up whatever is missing, then start the app and open the browser
rem    setup.bat --no-start    set up only
rem    setup.bat --skip-tests  do not run the test suite after (re)installing packages
rem    setup.bat --reinstall   rebuild the Python environment from scratch
rem    setup.bat --no-update   do not pull the latest code from GitHub / model from Hugging Face
rem
rem  Safe to run any number of times: finished steps are skipped, so a normal start takes seconds.
rem  Optional environment variables: SIMGUARD_TORCH=cpu|cu132  SIMGUARD_PORT=8000
rem                                  SIMGUARD_VENV=<folder>     QWEN_PASSWORD=<used for a new .env>
rem ==========================================================================================
setlocal EnableExtensions
title SIMGuard AI setup

rem `git pull` may replace this very file, and cmd reads a running batch file line by line.
rem So the real work always runs from a temporary copy; the copy is never touched by git.
if /i "%~1"=="--from-copy" goto running_from_copy
del "%TEMP%\simguard_setup_*.bat" >nul 2>&1
set "RUN_COPY=%TEMP%\simguard_setup_%RANDOM%%RANDOM%.bat"
copy /y "%~f0" "%RUN_COPY%" >nul
rem call and exit on ONE line: cmd must not read this file again after the copy returns
call "%RUN_COPY%" --from-copy "%~dp0" %* & exit /b
:running_from_copy
set "ROOT=%~2"
shift
shift
cd /d "%ROOT%"

set "NO_START="
set "SKIP_TESTS="
set "REINSTALL="
set "NO_UPDATE="
set "PASS_ARGS="
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--no-start" set "NO_START=1"
if /i "%~1"=="--skip-tests" set "SKIP_TESTS=1"
if /i "%~1"=="--reinstall" set "REINSTALL=1"
if /i "%~1"=="--no-update" set "NO_UPDATE=1"
if /i not "%~1"=="--no-update" set "PASS_ARGS=%PASS_ARGS% %~1"
shift
goto parse_args
:args_done

set "VENV=%SIMGUARD_VENV%"
if not defined VENV set "VENV=%ROOT%safe"
set "PY=%VENV%\Scripts\python.exe"
set "PORT=%SIMGUARD_PORT%"
if not defined PORT set "PORT=8000"
set "UI_PORT=%SIMGUARD_UI_PORT%"
if not defined UI_PORT set "UI_PORT=8501"
rem the API uses FRONTEND_URL to know where the Streamlit app lives
set "FRONTEND_URL=http://127.0.0.1:%UI_PORT%"
set "STAMP=%VENV%\.simguard_deps"
set "LAYA_DIR=%ROOT%finetune\fraud_v2\runs\20261113_171802_a1_6ep\final"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo.
echo  SIMGuard AI setup
echo  =================

rem ---------------------------------------------------------------- 0. latest code
echo.
echo [0/7] Latest code from GitHub
if defined NO_UPDATE (
    echo   skipped with --no-update
    goto code_ready
)
git --version >nul 2>&1
if errorlevel 1 (
    echo   git not found - using the code as it is
    goto code_ready
)
if not exist "%ROOT%.git" (
    echo   not a git clone - using the code as it is
    goto code_ready
)
rem only pull when no tracked file has local edits, so nobody's work is ever overwritten
set "DIRTY="
for /f "delims=" %%s in ('git status --porcelain --untracked-files=no') do set "DIRTY=1"
if defined DIRTY (
    echo   you have local changes - not pulling. Commit or discard them to get updates.
    goto code_ready
)
set "OLD_HEAD="
for /f "delims=" %%h in ('git rev-parse HEAD') do set "OLD_HEAD=%%h"
git pull --ff-only --quiet
if errorlevel 1 (
    echo   could not update - offline, or your branch has commits GitHub does not have.
    echo   Continuing with the current code.
    goto code_ready
)
set "NEW_HEAD="
for /f "delims=" %%h in ('git rev-parse HEAD') do set "NEW_HEAD=%%h"
if "%OLD_HEAD%"=="%NEW_HEAD%" (
    echo   already up to date
    goto code_ready
)
echo   updated to the latest version
git diff --quiet %OLD_HEAD% %NEW_HEAD% -- setup.bat
if errorlevel 1 (
    echo   setup.bat itself changed - restarting with the new version
    call "%ROOT%setup.bat" --no-update %PASS_ARGS% & exit /b
)
:code_ready

rem ---------------------------------------------------------------- 1. Python environment
echo.
echo [1/7] Python environment
if defined REINSTALL if exist "%VENV%" (
    echo   removing old environment
    rmdir /s /q "%VENV%"
)
if not exist "%PY%" goto venv_checked
"%PY%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto venv_checked
echo   environment is broken - rebuilding it
rmdir /s /q "%VENV%"
:venv_checked
if exist "%PY%" goto python_ready
call :make_venv
if errorlevel 1 goto fail
:python_ready
for /f "delims=" %%v in ('""%PY%" tools\setup_helpers.py py-version"') do echo   %%v

rem ---------------------------------------------------------------- 2. PyTorch
echo.
echo [2/7] PyTorch
set "TORCH_FLAVOR=%SIMGUARD_TORCH%"
if not defined TORCH_FLAVOR (
    nvidia-smi >nul 2>&1
    if errorlevel 1 (set "TORCH_FLAVOR=cpu") else (set "TORCH_FLAVOR=cu132")
)
"%PY%" -c "import torch,sys;want='%TORCH_FLAVOR%';ok=torch.__version__.startswith('2.15.0') and (torch.cuda.is_available() or want=='cpu');sys.exit(0 if ok else 1)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%v in ('""%PY%" tools\setup_helpers.py torch-info"') do echo   already installed: torch %%v
    goto torch_ready
)
echo   installing torch 2.15.0 [%TORCH_FLAVOR%] - the GPU build is a large download, please wait
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install "torch==2.15.0" --index-url "https://download.pytorch.org/whl/%TORCH_FLAVOR%"
if errorlevel 1 (
    if /i "%TORCH_FLAVOR%"=="cpu" goto fail
    echo   GPU build failed - falling back to the CPU build
    set "TORCH_FLAVOR=cpu"
    "%PY%" -m pip install "torch==2.15.0" --index-url "https://download.pytorch.org/whl/cpu"
    if errorlevel 1 goto fail
)
:torch_ready

rem ---------------------------------------------------------------- 3. packages
echo.
echo [3/7] Project packages
set "WANT_HASH="
for /f "delims=" %%h in ('""%PY%" tools\setup_helpers.py deps-hash"') do set "WANT_HASH=%%h"
set "HAVE_HASH="
if exist "%STAMP%" set /p HAVE_HASH=<"%STAMP%"
if "%WANT_HASH%"=="%HAVE_HASH%" (
    echo   up to date
    set "FRESH_DEPS="
    goto deps_ready
)
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r requirements.txt -c constraints.txt
if errorlevel 1 goto fail
"%PY%" -m pip check
> "%STAMP%" echo %WANT_HASH%
set "FRESH_DEPS=1"
:deps_ready

rem ---------------------------------------------------------------- 4. configuration
echo.
echo [4/7] Configuration
if not exist ".env" if not defined QWEN_PASSWORD (
    echo   No .env yet. Enter the Qwen service password - the QWEN_PASSWORD Kaggle secret.
    echo   Press Enter to skip; the app then uses template reports until you add it to .env.
    set /p "QWEN_PASSWORD=  Qwen password: "
)
"%PY%" tools\setup_helpers.py ensure-env
if errorlevel 1 goto fail

rem ---------------------------------------------------------------- 5. scoring engine
echo.
echo [5/7] LightGBM + Isolation Forest
if exist "ml\artifacts\model.joblib" if exist "ml\artifacts\iforest.joblib" if exist "ml\artifacts\scaler.json" (
    echo   trained models present
    goto ml_ready
)
echo   training on sim_swap_fraud_dataset_100000.csv - about 10 seconds
"%PY%" -m ml.train > "%TEMP%\simguard_train.log" 2>&1
if errorlevel 1 (
    type "%TEMP%\simguard_train.log"
    goto fail
)
echo   done
:ml_ready

rem ---------------------------------------------------------------- 6. Laya model
echo.
echo [6/7] Fine-tuned Laya model
rem downloaded from https://huggingface.co/Kyoiske/simguard-laya-fraud when missing
set "LAYA_FLAGS="
if defined NO_UPDATE set "LAYA_FLAGS=--no-update"
"%PY%" tools\setup_helpers.py laya-ensure "%LAYA_DIR%" %LAYA_FLAGS%

rem ---------------------------------------------------------------- 7. self-test
echo.
echo [7/7] Self-test
if defined SKIP_TESTS (
    echo   skipped with --skip-tests
    goto tests_done
)
if not defined FRESH_DEPS (
    echo   skipped - packages unchanged since the last passing run
    goto tests_done
)
"%PY%" -m pytest -q -p no:cacheprovider
if errorlevel 1 (
    echo   TESTS FAILED - see the output above
    > "%STAMP%" echo tests-failed
    goto fail
)
:tests_done

echo.
echo  Setup complete.
if defined NO_START goto end

rem ---------------------------------------------------------------- start
echo.
echo  Starting the API on http://127.0.0.1:%PORT% ...
"%PY%" -c "import requests,sys;requests.get('http://127.0.0.1:%PORT%/health',timeout=2);sys.exit(0)" >nul 2>&1
if not errorlevel 1 (
    echo   already running
    goto api_ready
)
start "SIMGuard AI API" "%PY%" -m uvicorn api.main:app --host 127.0.0.1 --port %PORT%
"%PY%" tools\setup_helpers.py wait-health %PORT%
if errorlevel 1 goto fail
:api_ready
"%PY%" tools\setup_helpers.py status %PORT%

echo.
echo  Starting the analyst frontend on http://127.0.0.1:%UI_PORT% ...
"%PY%" -c "import requests,sys;requests.get('http://127.0.0.1:%UI_PORT%/_stcore/health',timeout=2);sys.exit(0)" >nul 2>&1
if not errorlevel 1 (
    echo   already running
    goto ui_ready
)
set "API_URL=http://127.0.0.1:%PORT%"
start "SIMGuard AI frontend" /d "%ROOT%app" "%PY%" -m streamlit run app.py --server.address 127.0.0.1 --server.port %UI_PORT% --server.headless true
"%PY%" tools\setup_helpers.py wait-url "http://127.0.0.1:%UI_PORT%/_stcore/health"
if errorlevel 1 goto fail
:ui_ready
start "" "http://127.0.0.1:%UI_PORT%/"
echo.
echo  SIMGuard AI is open in your browser at http://127.0.0.1:%UI_PORT%
echo  To stop it, close the "SIMGuard AI API" and "SIMGuard AI frontend" windows.
goto end

rem ==========================================================================================
:make_venv
rem Prefer an installed Python 3.15 / 3.12 / 3.13; otherwise install one with uv (no admin needed).
for %%v in (3.15 3.12 3.13) do (
    py -%%v -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        echo   creating environment with Python %%v
        py -%%v -m venv "%VENV%"
        if exist "%PY%" exit /b 0
    )
)
python -c "import sys;sys.exit(0 if (3,12)<=sys.version_info[:2]<=(3,15) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo   creating environment with the python on PATH
    python -m venv "%VENV%"
    if exist "%PY%" exit /b 0
)
set "UV=uv"
uv --version >nul 2>&1
if errorlevel 1 set "UV=%USERPROFILE%\.local\bin\uv.exe"
"%UV%" --version >nul 2>&1
if errorlevel 1 (
    echo   no suitable Python found - installing uv to get one
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
"%UV%" --version >nul 2>&1
if errorlevel 1 (
    echo   could not install uv. Install Python 3.12 from https://www.python.org and run setup.bat again.
    exit /b 1
)
echo   installing Python 3.12 with uv
"%UV%" python install 3.12
if errorlevel 1 exit /b 1
"%UV%" venv --seed --python 3.12 "%VENV%"
if exist "%PY%" exit /b 0
exit /b 1

:fail
echo.
echo  SETUP FAILED - see the messages above. Fix the problem and run setup.bat again.
echo  If you are stuck, call Srijan.
endlocal
pause
exit /b 1

:end
endlocal
exit /b 0
