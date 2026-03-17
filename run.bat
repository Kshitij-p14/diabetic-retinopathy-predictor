@echo off
REM Quick Start Script for Diabetic Retinopathy Predictor

echo.
echo ================================
echo Diabetic Retinopathy Predictor
echo ================================
echo.

REM Check if model files exist
if not exist "model.pkl" (
    echo ERROR: model.pkl not found in current directory
    echo Please ensure model.pkl and scaler.pkl are present
    pause
    exit /b 1
)

if not exist "scaler.pkl" (
    echo ERROR: scaler.pkl not found in current directory
    echo Please ensure model.pkl and scaler.pkl are present
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo After startup, your browser will open at: http://localhost:8501
echo.
echo To stop the application, press Ctrl+C in this window
echo.

python -m streamlit run app_enhanced.py

pause
