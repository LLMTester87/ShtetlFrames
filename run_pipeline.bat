@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
call .venv\Scripts\activate.bat
python src\download.py
python src\discover_ia.py --download-top 10
python src\calibrate.py
python src\scan.py --fresh --threshold 0.05 --fps 1.0
python src\export_review.py
echo Done. Open output\review_queue.csv
