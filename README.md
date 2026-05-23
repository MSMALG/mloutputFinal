# 🌿 Plant Disease Classifier — Streamlit Cloud Deployment

> This is the **deployment repository** for the Plant Disease Classification project developed for CS435 — Machine Learning.  
> For the training notebook and original `app.py`, see the [Main Project Repository](https://github.com/MSMALG/PlantsDiseaseClassification/tree/main).

---

## 🔗 Live App

**[Open the Streamlit App](https://plantleavesclassification.streamlit.app/)**

---

##  What Is This Repository?

This repository is what Streamlit Cloud actually runs. It contains the trained model artifacts alongside a modified version of `app.py` adapted for cloud deployment.

---

## Repository Structure

```
├── app.py                  # Modified Streamlit app (see changes below)
├── best_model.keras        # Trained EfficientNetB0 weights
├── svm_model.pkl           # Trained SVM classifier
├── hog_scaler.pkl          # StandardScaler fitted on HOG training features
├── hog_pca.pkl             # PCA projection matrix (1,764 → 497 dims)
├── label_encoder.pkl       # Integer ↔ class name mapping
├── class_names.pkl         # Ordered list of 38 class label strings
├── dl_accuracy.pkl         # Accuracy scalars displayed in the model selector UI
├── svm_accuracy.pkl
├── rf_accuracy.pkl
├── requirements.txt        # Includes gdown for RF model download
└── .streamlit/
    └── secrets.toml        # GROQ_API_KEY (Streamlit Cloud Secrets — not committed)
```

> `rf_model.pkl` is **not** stored here — it is downloaded from Google Drive at runtime on first load (see below).

---

## Key Difference from the Main Repository

The only functional difference between this `app.py` and the one in the main repository is how `rf_model.pkl` is loaded.

### Why the change was needed

The trained Random Forest model (`rf_model.pkl`) is a 200-tree ensemble that exceeds GitHub's 100 MB file size limit and cannot be stored directly in the repository. All other model files are within limits and are stored normally.

### Solution: download via `gdown`

The `load_classical_assets()` function was modified to check whether `rf_model.pkl` exists locally at startup. If it does not, the app downloads it from Google Drive using `gdown` before proceeding. Streamlit's `@st.cache_resource` ensures this download only happens once per session.

```python
@st.cache_resource
def load_classical_assets():
    rf_filename = "rf_model.pkl"
    
    if not os.path.exists(rf_filename):
        with st.spinner("Downloading Random Forest Model from Google Drive... Please wait."):
            google_drive_id = '1oqtjMZw0L5QyJNeuOUMB8dOPcs10zw_J'
            try:
                gdown.download(id=google_drive_id, output=rf_filename, quiet=False)
            except Exception as e:
                st.error(f"Download failed. Error details: {e}")
                st.stop()
            
    return {
        "svm": joblib.load("svm_model.pkl"),
        "rf": joblib.load(rf_filename),
        "scaler": joblib.load("hog_scaler.pkl"),
        "pca": joblib.load("hog_pca.pkl"),
        "le": joblib.load("label_encoder.pkl"),
    }
```

### Cold start behaviour

On the very first load after a fresh deployment or session restart, users will see a spinner while `rf_model.pkl` downloads from Google Drive. All subsequent interactions within the same session use the cached version and are instantaneous.

---

## Secrets Setup (Streamlit Cloud)

The Groq API key is configured via **Streamlit Cloud → App Settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

It is not stored in any file in this repository.

---


## Team

| Name | Student ID |
|---|---|
| Muzna Abdelgadir | 441211827 |
| Bedor Alharbi | 432205469 |
| Raghad Mesleh | 441203195 |
| Horiah Algofidi | 441203342 |

**Course:** CS435 — Machine Learning | **Lecturer:** Dr. Renad Alsweed | **Section:** 5500
