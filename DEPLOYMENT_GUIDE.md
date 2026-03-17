# 🚀 DEPLOYMENT GUIDE

## Streamlit Cloud Deployment (Recommended)

### Setup Instructions:

#### 1. **Initialize Git Repository**

```bash
cd d:\college\excelr

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Diabetic Retinopathy Predictor v2.0"
```

#### 2. **Create GitHub Repository**

1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `diabetic-retinopathy-predictor`
4. Choose "Public" (required for free Streamlit deployment)
5. Click "Create repository"

#### 3. **Push to GitHub**

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/diabetic-retinopathy-predictor.git

# Rename branch to main if needed
git branch -M main

# Push code
git push -u origin main
```

#### 4. **Deploy to Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up / Sign in"
3. Authenticate with GitHub
4. Click "Deploy an app"
5. Fill in:
   - **GitHub repo**: YOUR_USERNAME/diabetic-retinopathy-predictor
   - **Branch**: `main`
   - **File path**: `app_enhanced.py`
6. Click "Deploy"

**Your app will be live at:**
```
https://diabetic-retinopathy-predictor.streamlit.app
```

---

## Deployment Checklist

✅ Files Required:
- [x] app_enhanced.py (main app)
- [x] requirements.txt (dependencies)
- [x] .streamlit/config.toml (configuration)
- [x] model.pkl (trained model)
- [x] scaler.pkl (feature scaler)
- [x] README.md (documentation)
- [x] .gitignore (git configuration)

✅ Configuration:
- [x] Streamlit config.toml created
- [x] Requirements.txt ready
- [x] All dependencies specified

✅ Git Setup:
- [ ] GitHub account created
- [ ] Repository initialized
- [ ] Code pushed to GitHub

✅ Deployment:
- [ ] Streamlit Cloud account created
- [ ] App deployed to cloud
- [ ] Live URL accessed and tested

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
→ Ensure `scikit-learn` is in requirements.txt

### "FileNotFoundError: model.pkl"
→ Add model.pkl and scaler.pkl to the repository

### "Permission denied on GitHub"
→ Check that repository is Public and you have proper GitHub permissions

### "Streamlit app takes too long to load"
→ Check internet connection and Streamlit Cloud server status

### "Model predictions seem off"
→ Verify model.pkl and scaler.pkl are correct versions

---

## Post-Deployment

### Access Your App:
- Share the Streamlit Cloud URL with others
- App updates automatically when you push to GitHub
- Monitor app logs in Streamlit Cloud dashboard

### Manage App:
- Settings: Configure visibility, email notifications
- Settings > Reboot app: Force restart if needed
- Advanced settings: Add environment variables if needed

### Share With Others:
- Public link is automatically generated
- No additional setup required for viewers
- Works on mobile and desktop

---

## Alternative Deployment Options

If you need more control or want different deployment:

### Docker + Google Cloud Run
1. Create `Dockerfile`
2. Build Docker image
3. Push to Google Cloud Registry
4. Deploy to Cloud Run

### Heroku (*requires credit card*)
1. Create Procfile
2. Authenticate with Heroku CLI
3. Deploy with `git push heroku main`

### AWS EC2
1. Launch EC2 instance
2. Install Python and dependencies
3. Clone repository and run with systemd service
4. Use Nginx as reverse proxy

### DigitalOcean Apps
1. Connect GitHub repository
2. Configure buildpack
3. Deploy one-click

---

## Environment Variables (if needed)

For Streamlit Cloud:
1. Go to App Settings
2. Click "Secrets"
3. Add in TOML format:
```toml
[database]
url = "postgresql://..."

[api]
key = "your-api-key"
```

Access in code:
```python
import streamlit as st
db_url = st.secrets["database"]["url"]
api_key = st.secrets["api"]["key"]
```

---

## Monitoring & Maintenance

### Weekly
- Check app performance
- Review logs for errors
- Verify model accuracy

### Monthly
- Update dependencies
- Check for Streamlit updates
- Monitor cloud costs

### Quarterly
- Retrain model with new data
- Update documentation
- Test all features

---

## Support Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [GitHub Help](https://docs.github.com)
- [Streamlit Community](https://discuss.streamlit.io)

---

**Good luck with your deployment! 🚀**
