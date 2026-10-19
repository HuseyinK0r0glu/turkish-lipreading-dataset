@echo off
rem Five new presenters (speaker/channel diversity), then Phase 2 over every clip with
rem the fixed segments.py. Safe to re-run after a crash: finished rows are skipped,
rem Phase 2 resumes from its face_track_report.csv, logs are appended to.
rem The errorlevel checks matter: a Windows Update restart killed step 1 on 2026-10-14
rem and, without them, cmd carried on into step 2 while rows were still pending.
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem selcuk_tepeli (NOW) is left out: NOW keeps a sign-language interpreter box in every
rem frame, so no sample is ever solo (11 episodes gave 1 clip).
echo [%date% %time%] step 1/2: pipeline 2 for the five new presenters
rem Capped 2026-10-17 to shorten the run: all of ekrem_acikel (11) and sinem_fistikoglu
rem (19), the first 20 rows of irfan_degirmenci and sule_aydin, the first 5 of seda_selek
rem (1h50 newscasts, ~6 h a row). --limit counts completed rows too, so a re-run still
rem stops at the same rows.
uv run python -u scripts\run_all.py --pipeline 2 --reporters ekrem_acikel sinem_fistikoglu >> data\run_all.log 2>> data\run_all.err
if errorlevel 1 (echo [%date% %time%] step 1 stopped with an error, NOT starting Phase 2. Re-run this file. & exit /b 1)
uv run python -u scripts\run_all.py --pipeline 2 --reporters irfan_degirmenci --limit 20 >> data\run_all.log 2>> data\run_all.err
if errorlevel 1 (echo [%date% %time%] step 1 stopped with an error, NOT starting Phase 2. Re-run this file. & exit /b 1)
uv run python -u scripts\run_all.py --pipeline 2 --reporters sule_aydin --limit 20 >> data\run_all.log 2>> data\run_all.err
if errorlevel 1 (echo [%date% %time%] step 1 stopped with an error, NOT starting Phase 2. Re-run this file. & exit /b 1)
uv run python -u scripts\run_all.py --pipeline 2 --reporters seda_selek --limit 5 >> data\run_all.log 2>> data\run_all.err
if errorlevel 1 (echo [%date% %time%] step 1 stopped with an error, NOT starting Phase 2. Re-run this file. & exit /b 1)

echo [%date% %time%] step 2/2: Phase 2 over all clips (resumes from data\phase2)
uv run python -u scripts\phase2\run_phase2.py >> data\phase2_run3.log 2>&1

echo [%date% %time%] all done
