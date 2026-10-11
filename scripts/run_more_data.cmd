@echo off
rem More data for the vocabulary target, then Phase 2 over everything with the fixed segments.py.
cd /d "C:\Users\Can Zorlu\Desktop\turkish-lipreading-dataset"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if exist data\run_all.log move /y data\run_all.log data\run_all.2026-10-09.log
if exist data\run_all.err move /y data\run_all.err data\run_all.2026-10-09.err

echo [%date% %time%] step 1/3: pending pipeline-2 rows except cuneyt_ozdemir, osman_gokcek
uv run python -u scripts\run_all.py --pipeline 2 --exclude-reporters cuneyt_ozdemir osman_gokcek > data\run_all.log 2> data\run_all.err

echo [%date% %time%] step 2/3: cuneyt_ozdemir up to 85 h of clips
uv run python -u scripts\run_all.py --pipeline 2 --reporters cuneyt_ozdemir --max-hours 85 >> data\run_all.log 2>> data\run_all.err

echo [%date% %time%] step 3/3: Phase 2 over all clips (old output kept in data\phase2_v1)
if exist data\phase2 if not exist data\phase2_v1 move data\phase2 data\phase2_v1
uv run python -u scripts\phase2\run_phase2.py > data\phase2_run3.log 2>&1

echo [%date% %time%] all done
