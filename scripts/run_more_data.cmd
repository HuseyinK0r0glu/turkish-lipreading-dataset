@echo off
rem Six new presenters (speaker/channel diversity), then Phase 2 over every clip with
rem the fixed segments.py. Safe to re-run after a crash: finished rows are skipped,
rem Phase 2 resumes from its face_track_report.csv, logs are appended to.
cd /d "C:\Users\Can Zorlu\Desktop\turkish-lipreading-dataset"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [%date% %time%] step 1/2: pipeline 2 for the six new presenters
uv run python -u scripts\run_all.py --pipeline 2 --reporters selcuk_tepeli ekrem_acikel sinem_fistikoglu seda_selek irfan_degirmenci sule_aydin >> data\run_all.log 2>> data\run_all.err

echo [%date% %time%] step 2/2: Phase 2 over all clips (v1 output kept in data\phase2_v1)
if exist data\phase2 if not exist data\phase2_v1 move data\phase2 data\phase2_v1
uv run python -u scripts\phase2\run_phase2.py >> data\phase2_run3.log 2>&1

echo [%date% %time%] all done
