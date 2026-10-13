@echo off
rem Five new presenters (speaker/channel diversity), then Phase 2 over every clip with
rem the fixed segments.py. Safe to re-run after a crash: finished rows are skipped,
rem Phase 2 resumes from its face_track_report.csv, logs are appended to.
cd /d "C:\Users\Can Zorlu\Desktop\turkish-lipreading-dataset"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem selcuk_tepeli (NOW) is left out: NOW keeps a sign-language interpreter box in every
rem frame, so no sample is ever solo (11 episodes gave 1 clip).
echo [%date% %time%] step 1/2: pipeline 2 for the five new presenters
uv run python -u scripts\run_all.py --pipeline 2 --reporters ekrem_acikel sinem_fistikoglu irfan_degirmenci sule_aydin >> data\run_all.log 2>> data\run_all.err
rem seda_selek: only her first 10 of 27 newscasts (1h50 each, ~6 h a row); --limit
rem counts completed rows too, so a re-run still stops at the same 10.
uv run python -u scripts\run_all.py --pipeline 2 --reporters seda_selek --limit 10 >> data\run_all.log 2>> data\run_all.err

echo [%date% %time%] step 2/2: Phase 2 over all clips (v1 output kept in data\phase2_v1)
if exist data\phase2 if not exist data\phase2_v1 move data\phase2 data\phase2_v1
uv run python -u scripts\phase2\run_phase2.py >> data\phase2_run3.log 2>&1

echo [%date% %time%] all done
