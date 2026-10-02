@echo off
title Autonomous Self-Healing Data Analyst Agent
echo =====================================================================
echo Launching Autonomous Self-Healing Data Analyst Agent...
echo =====================================================================
python -m streamlit run app.py --server.headless false
pause
