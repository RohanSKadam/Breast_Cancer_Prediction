import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load and clean dataset
csv_filepath = "Breast_Cancer.csv"
df = pd.read_csv(csv_filepath)
df = df.apply(lambda col: col.str.strip().str.capitalize() if col.dtypes == 'object' else col)

# Page Config
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")

# App Title
st.markdown("<h1 style='text-align: center; color: #e91e63;'>🎗️ Breast Cancer Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by Rohan Kadam • For Educational Use Only</p>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📋 Patient Information")

age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 50)
race = st.sidebar.selectbox("Race", df["Race"].unique())
marital_status = st.sidebar.selectbox("Marital Status", df["Marital Status"].unique())
t_stage = st.sidebar.selectbox("T Stage", df["T Stage"].unique())
n_stage = st.sidebar.selectbox("N Stage", df["N Stage"].unique())
sixth_stage = st.sidebar.selectbox("6th Stage", df["6th Stage"].unique())
differentiate = st.sidebar.selectbox("Differentiation", df["differentiate"].unique())
grade = st.sidebar.selectbox("Grade", df["Grade"].unique())
tumor_size = st.sidebar.slider("Tumor Size", int(df["Tumor Size"].min()), int(df["Tumor Size"].max()), 20)
estrogen_status = st.sidebar.selectbox("Estrogen Status", df["Estrogen Status"].unique())
progesterone_status = st.sidebar.selectbox("Progesterone Status", df["Progesterone Status"].unique())
regional_node_examined = st.sidebar.slider("Regional Node Examined", int(df["Regional Node Examined"].min()), int(df["Regional Node Examined"].max()), 10)
reginol_node_positive = st.sidebar.slider("Reginol Node Positive", int(df["Reginol Node Positive"].min()), int(df["Reginol Node Positive"].max()), 5)

if st.sidebar.button("🔍 Predict Survival"):
    user_data = pd.DataFrame({
        "Age": [age],
        "Race": [race],
        "Marital Status": [marital_status],
        "T Stage": [t_stage],
        "N Stage": [n_stage],
        "6th Stage": [sixth_stage],
        "differentiate": [differentiate],
        "Grade": [grade],
        "Tumor Size": [tumor_size],
        "Estrogen Status": [estrogen_status],
        "Progesterone Status": [progesterone_status],
        "Regional Node Examined": [regional_node_examined],
        "Reginol Node Positive": [reginol_node_positive],
    })

    # -----------------------
    # Predict Survival Months
    # -----------------------
    X_reg = df.drop(["Survival Months", "Status"], axis=1)
    y_reg = df["Survival Months"]
    X_reg = pd.get_dummies(X_reg, drop_first=True)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    model_reg = RandomForestRegressor(random_state=42)
    model_reg.fit(X_train_reg, y_train_reg)

    user_data_reg = pd.get_dummies(user_data, drop_first=True)
    user_data_reg = user_data_reg.reindex(columns=X_train_reg.columns, fill_value=0)
    prediction_reg = model_reg.predict(user_data_reg)

    # -----------------------
    # Predict Survival Status
    # -----------------------
    X_clf = df.drop("Status", axis=1)
    y_clf = df["Status"]
    X_clf = pd.get_dummies(X_clf, drop_first=True)

    user_data_clf = pd.get_dummies(user_data, drop_first=True)
    user_data_clf = user_data_clf.reindex(columns=X_clf.columns, fill_value=0)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    classifiers = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "SVC": SVC(probability=True, class_weight='balanced', random_state=42),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=4),
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "GradientBoost": GradientBoostingClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
    }

    best_classifier = None
    best_accuracy = 0
    for name, clf in classifiers.items():
        clf.fit(X_train_clf, y_train_clf)
        y_pred = clf.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_classifier = clf

    prediction_clf = best_classifier.predict(user_data_clf)[0]
    proba = best_classifier.predict_proba(user_data_clf)[0] if hasattr(best_classifier, "predict_proba") else None

    # -----------------------
    # Results Display
    # -----------------------
    st.markdown("## 🧾 Results")
    col1, col2 = st.columns(2)

    survival_months_pred = prediction_reg[0]
    if survival_months_pred < 24:
        category = "Short-term"
        emoji = "🔴"
    elif survival_months_pred < 60:
        category = "Mid-term"
        emoji = "🟡"
    else:
        category = "Long-term"
        emoji = "🟢"

    with col1:
        st.metric(label="Predicted Survival Months", value=f"{survival_months_pred:.2f}", delta=category)
        st.markdown(f"### {emoji} {category} Survival")

    with col2:
        status_icon = "✅" if prediction_clf == "Alive" else "⚠️"
        st.metric(label="Predicted Survival Status", value=prediction_clf)
        if proba is not None:
            st.caption(f"Confidence: {proba.max():.2%}")
        st.markdown(f"### {status_icon} Status")

    st.subheader("📌 Model Performance")
    st.write(f"Best Classifier: **{type(best_classifier).__name__}**")
    st.success(f"Model Accuracy: {best_accuracy:.2%}")

    st.subheader("⚠️ Disclaimer")
    st.write("These predictions are intended for **educational and exploratory use only**. Please consult healthcare professionals for real-life decisions.")

    # -----------------------
    # Dataset Visuals
    # -----------------------
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview & Visualizations")

    st.write("### Tumor Size Distribution")
    st.bar_chart(df["Tumor Size"])

    st.write("### Survival Months Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(df["Survival Months"], bins=20, color="#03a9f4")
    ax1.set_xlabel("Months")
    ax1.set_ylabel("Patients")
    st.pyplot(fig1)

    st.write("### Survival Months Boxplot")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(df["Survival Months"])
    st.pyplot(fig2)

    st.write("### Status Counts")
    st.dataframe(df["Status"].value_counts().rename_axis("Status").reset_index(name="Count"))
