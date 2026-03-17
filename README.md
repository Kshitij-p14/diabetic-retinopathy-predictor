# 🩺 Diabetic Retinopathy Risk Predictor - Advanced Edition

**AI-powered comprehensive assessment for diabetic retinopathy detection**

## 📊 Features

### Core Functionality
- **🔬 Risk Prediction** - Individual patient risk assessment using Logistic Regression
- **📊 Advanced Analytics** - Risk factor analysis with parameter comparisons
- **📚 Education & Info** - Comprehensive clinical information and guidelines
- **📁 Batch Analysis** - Bulk patient analysis from CSV files

### Advanced Features
- **📈 Real-Time Dashboard** - KPI metrics, risk distribution, trend analysis
- **👥 Risk Stratification** - Age group segmentation and comparative analysis
- **📋 Report Generation** - PDF, Excel, and CSV export capabilities
- **💬 Recommendations** - Personalized action plans based on risk level

## 🎯 Technology Stack

- **Framework**: Streamlit 1.55.0
- **ML Model**: Logistic Regression (scikit-learn)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Report Generation**: ReportLab (PDF), openpyxl (Excel)

## 📋 Model Details

- **Algorithm**: Logistic Regression
- **Training Data**: 6,000 patient records
- **Features**: 4 clinical parameters
  - Age (years)
  - Systolic Blood Pressure (mmHg)
  - Diastolic Blood Pressure (mmHg)
  - Cholesterol (mg/dL)
- **Performance**:
  - Accuracy: ~85%
  - AUC-ROC: ~0.89

## 🚀 Local Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone or download the repository
2. Navigate to the project directory:
   ```bash
   cd diabetic-retinopathy-predictor
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the model files:
   - `model.pkl` (trained Logistic Regression model)
   - `scaler.pkl` (feature scaler)

5. Run the app:
   ```bash
   streamlit run app_enhanced.py
   ```

6. Open your browser to `http://localhost:8501`

## ☁️ Streamlit Cloud Deployment

### Quick Deploy in 3 Steps:

#### Step 1: Prepare GitHub Repository
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Add Diabetic Retinopathy Predictor app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

#### Step 2: Sign Up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Deploy an app"
3. Sign in with GitHub (authenticate if needed)

#### Step 3: Deploy
1. Select your repository
2. Select the branch: `main`
3. Select the main file path: `app_enhanced.py`
4. Click "Deploy"

Your app will be live in 2-3 minutes! 🎉

### Important Notes:
- Free tier allows 1 public app
- Keep your GitHub repo public for easy deployment
- Streamlit Cloud automatically redeploys on GitHub push
- Environment variables can be set in Settings if needed

## 📁 File Structure

```
diabetic-retinopathy-predictor/
├── app_enhanced.py          # Main Streamlit application
├── requirements.txt         # Python dependencies
├── model.pkl               # Pre-trained model
├── scaler.pkl              # Feature scaler
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── README.md               # This file
└── P653_pronostico_dataset.csv  # Sample dataset
```

## 👤 Usage

### Individual Prediction
1. Go to **Risk Prediction** tab
2. Enter patient parameters
3. Click "Analyze Risk Profile"
4. View risk assessment and recommendations

### Batch Analysis
1. Go to **Batch Analysis** tab
2. Download sample CSV template
3. Upload your patient CSV
4. Click "Analyze All Patients"
5. Export results as CSV

### Generate Reports
1. Make individual predictions
2. Go to **Reports & Export** tab
3. Choose format: PDF, Excel, or CSV
4. Download your report

## 📊 Risk Categories

| Risk Level | Probability | Action |
|----------|-----------|--------|
| 🟢 Low | < 50% | Continue regular monitoring |
| 🟡 Medium | 50-70% | Schedule follow-up |
| 🔴 High | ≥ 70% | Immediate consultation |

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals before making medical decisions.

## 🔒 Privacy & Security

- All data is processed locally on your device (Streamlit Cloud)
- No personal data is stored on servers
- Model predictions are not logged
- Compliance with standard web application security practices

## 🤝 Support

For issues, questions, or suggestions:
- Check the Education & Info tab for clinical information
- Review the Recommendations tab for guidance

## 📝 License

Educational Project - P653 Machine Learning
Built with ❤️ using Streamlit

## 🙏 Acknowledgments

- Dataset: 6,000 patient records
- Model: Logistic Regression (scikit-learn)
- Framework: Streamlit
- Visualization: Matplotlib

---

**Last Updated**: March 2026
**Version**: 2.0 - Advanced Edition with 8 Features
