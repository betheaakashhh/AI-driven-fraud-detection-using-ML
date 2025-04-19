


# 💳 AI-Driven Credit Card Fraud Detection

This project is a machine learning-based ### **Credit Card Fraud Detection System** built using **Python** and an interactive **Streamlit UI**. It allows users to explore, analyze, and predict fraudulent transactions using trained ML models with easy-to-use visualizations and real-time results.



## 📁 Project Overview

- **Frontend (UI)**: `app.py` – Built with Streamlit  
- **Backend (ML logic)**: `model_utils.py`  
- **ML Models Used**:
  - Logistic Regression
  - Decision Tree Classifier
  - Isolation Forest



## ⚙️ Setup Instructions

Follow the steps below to get the project up and running on your system.

### ✅ Step 1: Create a Python Virtual Environment

```bash
python -m venv venv
```

### ✅ Step 2: Activate the Environment

#### For Windows:

```bash
cd venv/Scripts
activate
```

If the above doesn't work:

```bash
./activate
```

> ⚠️ If you see an error like **"execution of scripts is disabled"**, then:

1. Open **PowerShell as Administrator**  
2. Run:
   ```powershell
   Set-ExecutionPolicy Unrestricted
   ```
3. Type `A` to allow all

---

### ✅ Step 3: Install Required Libraries

Make sure your environment is activated, then run:

```bash
pip install streamlit pandas matplotlib seaborn numpy scikit-learn notebooks
```

---

## ▶️ How to Run the App

After installing the dependencies:

1. Navigate to the project folder where `app.py` and `model_utils.py` are located
2. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Features

- Multi-model fraud detection engine
- Real-time classification and predictions
- Interactive graphs and data visualizations
- Built with Streamlit – clean and offline-capable UI

---

## 📂 Folder Structure

```
fraud_detection_project/
│
├── app.py                # Main UI (Streamlit)
├── model_utils.py        # ML logic and backend
├── venv/                 # Python virtual environment
├── data/                 # Optional dataset folder
└── notebooks/            # Jupyter notebooks (EDA, experiments)
```

---

## 📌 Notes

- Works fully **offline** after installation
- You can plug in custom datasets via the `data/` folder
- Easily extendable with new models via `model_utils.py`

---

## 👨‍💻 Author

Made with 💙 using Python & Streamlit  
Feel free to fork this repo, contribute, or suggest improvements!

---
```
