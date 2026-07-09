@echo off
cd /d "%~dp0"

echo Train model...

if exist .venv (
    echo .venv already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    uv venv .venv
)

call .venv\Scripts\activate.bat

uv pip install -r requirements.txt

echo Preparing dataset...
python -u src/prepare_forecasting_dataset.py

echo Training SARIMAX...
python -u src/train_sarimax.py --n-jobs -1

echo Done.
pause