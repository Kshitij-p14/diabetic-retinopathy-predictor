#!/bin/bash

# Quick Start Script for Diabetic Retinopathy Predictor

echo ""
echo "================================"
echo "Diabetic Retinopathy Predictor"
echo "================================"
echo ""

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo "ERROR: model.pkl not found in current directory"
    echo "Please ensure model.pkl and scaler.pkl are present"
    exit 1
fi

if [ ! -f "scaler.pkl" ]; then
    echo "ERROR: scaler.pkl not found in current directory"
    echo "Please ensure model.pkl and scaler.pkl are present"
    exit 1
fi

echo "Starting Streamlit application..."
echo ""
echo "After startup, your browser will open at: http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

python -m streamlit run app_enhanced.py
