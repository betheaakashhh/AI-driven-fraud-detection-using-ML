# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model_utils import load_data, preprocess_data, split_and_scale, train_models, evaluate_model

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: #ff8510;'>💳 Credit Card Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Powered by Machine Learning · Built with Streamlit</h4>", unsafe_allow_html=True)
st.markdown("---")

# Load & preprocess data
data = load_data('venv\data\creditcard.csv')  # ← Correct relative path
X, Y, new_data = preprocess_data(data)
X_train, X_test, Y_train, Y_test, scaler = split_and_scale(X, Y)
log_model, tree_model, iso_model = train_models(X_train, Y_train)

# Tabs for dashboard layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Dataset Overview", "📊 Visualizations", "📈 Model Evaluation", "🔍 Predict Fraud", "👨‍💻 Meet Our Team"])

# ----- Tab 1: Dataset Overview -----
with tab1:
    st.header("Dataset Info")
    st.write(data.head())
    st.write("Original Class Distribution:")
    st.bar_chart(data['Class'].value_counts())

# ----- Tab 2: Visualizations -----
with tab2:
    st.header("Balanced Data Visualizations")
    fig1, ax1 = plt.subplots()
    ax1.pie(new_data['Class'].value_counts(), labels=['Legit', 'Fraud'], autopct='%1.1f%%')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(data=new_data, x="Amount", hue="Class", bins=30, ax=ax2)
    st.pyplot(fig2)

# ----- Tab 3: Model Evaluation -----
with tab3:
    st.header("Compare ML Models")
    model_option = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Isolation Forest"])
    
    if model_option == "Logistic Regression":
        acc, conf, report = evaluate_model(log_model, X_test, Y_test)
    elif model_option == "Decision Tree":
        acc, conf, report = evaluate_model(tree_model, X_test, Y_test)
    else:
        acc, conf, report = evaluate_model(iso_model, X_test, Y_test, model_type="unsupervised")
    
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.write("Confusion Matrix", conf)
    st.write("Classification Report:", report)

# ----- Tab 4: Real-time Prediction -----
with tab4:
    st.header("Fraud Prediction on New Transaction")
    st.info("Paste 30 comma-separated values of a new transaction:")

    input_text = st.text_area("Input transaction data:")
    selected_model = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Isolation Forest"])

    if st.button("Predict"):
        try:
            input_data = np.array([float(x) for x in input_text.split(',')]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            if selected_model == "Logistic Regression":
                result = log_model.predict(input_scaled)
            elif selected_model == "Decision Tree":
                result = tree_model.predict(input_scaled)
            else:
                result = iso_model.predict(input_scaled)
                result = [0 if X == 1 else 1]

            st.success("🚨 Fraudulent Transaction" if result[0] == 1 else "✅ Legit Transaction")
        except Exception as e:
            st.error("Invalid input! Please ensure 30 comma-separated numbers are entered.")

# ----- Tab 5: Meet Our Team -----
with tab5:
    st.markdown("## 🕰️ OverClockers")
    st.markdown("We're a team of passionate developers working on AI-powered fraud detection 🚀")

    team = [
        {
            "name": "Aakash Kumar Sahu",
            "role": "Full Stack Developer with ML Expertise",
            "linkedin": "https://www.linkedin.com/in/aakashkumarsahu",
            "instagram": "https://www.instagram.com/aakash.codes"
        },
        {
            "name": "Somesh Chaukde",
            "role": "Full Stack Developer & Entrepreneur",
            "linkedin": "https://www.linkedin.com/in/someshchaukde",
            "instagram": "https://www.instagram.com/somesh.dev"
        },
        {
            "name": "Piyush Sinha",
            "role": "UI/UX Designer & Full Stack Developer",
            "linkedin": "https://www.linkedin.com/in/piyushsinha",
            "instagram": "https://www.instagram.com/piyush.design"
        },
        {
            "name": "Prakhar Dewangan",
            "role": "AI/ML Engineer & Full Stack Developer",
            "linkedin": "https://www.linkedin.com/in/prakhardewangan",
            "instagram": "https://www.instagram.com/prakhar.ai"
        },
        {
            "name": "Brijesh",
            "role": "Computer Vision Engineer & Full Stack Developer",
            "linkedin": "https://www.linkedin.com/in/brijeshvision",
            "instagram": "https://www.instagram.com/brijesh.cv"
        }
    ]

    cols = st.columns(len(team))

    for index, member in enumerate(team):
        with cols[index]:
            st.image("https://source.unsplash.com/100x100/?face,portrait", width=100)
            st.markdown(f"### {member['name']}")
            st.markdown(f"**{member['role']}**")
            st.markdown(f"[🔗 LinkedIn]({member['linkedin']})", unsafe_allow_html=True)
            st.markdown(f"[📸 Instagram]({member['instagram']})", unsafe_allow_html=True)

