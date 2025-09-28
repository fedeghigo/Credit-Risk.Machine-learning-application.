import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    st.title("Credit Scoring")
    st.sidebar.title("Credit Scoring Web App")
    st.markdown("Credit analysis")

    # ----------------------------
    # Load Data
    # ----------------------------
    @st.cache_data
    def load_data():
        url = "https://github.com/fedeghigo/Credit_Risk.Machine_learning-application/releases/download/credit_scoring.ml/LendingClub.csv"
        data = pd.read_csv(url)

        cols = [
            "loan_amnt",
            "term",
            "int_rate",
            "funded_amnt",
            "grade",
            "annual_inc",
            "dti",
            "hardship_loan_status",
            "delinq_2yrs",
            "last_pymnt_amnt",
            "emp_length",
            "loan_status",
            "home_ownership",
            "tax_liens",
        ]
        data = data[cols]
        data = data.drop(["hardship_loan_status"], axis=1)
        data = data.dropna()
        data = data[
            (data["loan_status"] == "Fully Paid")
            | (data["loan_status"] == "Charged Off")
        ]

        binary = {"Fully Paid": 0, "Charged Off": 1}
        data["defaulted"] = data.loan_status.map(binary)
        data = data.drop("loan_status", axis=1)

        emp_map = {
            "7 years": 7,
            "4 years": 4,
            "1 year": 1,
            "3 years": 3,
            "< 1 year": 0,
            "6 years": 6,
            "5 years": 5,
            "2 years": 2,
            "10+ years": 10,
            "9 years": 9,
            "8 years": 8,
        }
        data.emp_length = data.emp_length.map(emp_map)

        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        data.grade = data.grade.map(grade_map)

        data.term = pd.to_numeric(data.term.str.slice(1, 3))

        home_map = {
            "MORTGAGE": 1,
            "RENT": 2,
            "OWN": 3,
            "ANY": 4,
            "OTHER": 5,
            "NONE": 6,
        }
        data.home_ownership = data.home_ownership.map(home_map)

        return data

    # ----------------------------
    # Correlation heatmap
    # ----------------------------
    def corr_mt(df):
        corr_mtx = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # ----------------------------
    # Train/test split
    # ----------------------------
    @st.cache_data
    def split(df):
        x = df.iloc[:, 0:11]
        y = df.iloc[:, 12]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )
        return x_train, x_test, y_train, y_test

    # ----------------------------
    # Metrics plotting
    # ----------------------------
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(plt.gcf())

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())

    # ----------------------------
    # Load and split data
    # ----------------------------
    df = load_data()
    class_names = [0, 1]
    x_train, x_test, y_train, y_test = split(df)

    # ----------------------------
    # Sidebar UI
    # ----------------------------
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Random Forest", "Logistic Regression", "Linear Discriminant Analysis")
    )

    # ----------------------------
    # Random Forest
    # ----------------------------
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 5000, step=10, key="n_estimators"
        )
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 100, step=1, key="max_depth"
        )
        max_features = st.sidebar.radio("Max Features", ("auto", "sqrt"), key="max_features")
        bootstrap = st.sidebar.radio(
            "Bootstrap samples when building trees", ("True", "False"), key="bootstrap"
        )
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify_rf"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap == "True",
                max_features=max_features,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # ----------------------------
    # Logistic Regression
    # ----------------------------
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        solver = st.sidebar.radio(
            "Which solver?", ("newton-cg", "lbfgs", "liblinear", "sag", "saga"), key="solver"
        )
        penalty = st.sidebar.radio("Which penalty?", ("l1", "l2"), key="penalty")
        C = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 100.0, step=0.01, key="C_LR"
        )
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify_lr"):
            st.subheader("Logistic Regression Results")
            if penalty == "l1" and solver not in ("liblinear", "saga"):
                solver = "liblinear"  # fallback
            if penalty == "l2" and solver not in ("lbfgs", "newton-cg", "sag", "saga"):
                solver = "lbfgs"  # fallback
            model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter, solver=solver)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            precision_val = precision_score(y_test, y_pred, labels=class_names)
            st.write("Precision: ", round(precision_val, 2))

            recall_val = recall_score(y_test, y_pred, labels=class_names)
            st.write("Recall: ", round(recall_val, 2))

            st.write("Accuracy: ", round(accuracy, 2))
            
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # ----------------------------
    # Linear Discriminant Analysis
    # ----------------------------
    if classifier == "Linear Discriminant Analysis":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Shrinkage parameter)", 0.01, 1.0, step=0.01, key="C_LDA"
        )
        n_components = st.sidebar.number_input(
            "Number of dimensions", 1, 8, step=1, key="dim"
        )
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify_lda"):
            st.subheader("Linear Discriminant Analysis Results")
            max_components = min(x_train.shape[1], len(np.unique(y_train)) - 1)
            if n_components > max_components:
                n_components = max_components

            model = LinearDiscriminantAnalysis(
                solver="eigen",
                shrinkage=C,
                n_components=n_components,
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(
                "Precision: ",
                round(
                    precision_score(y_test, y_pred, average="weighted", labels=class_names),
                    2
                ),
            )
            st.write(
                "Recall: ",
                round(
                    recall_score(y_test, y_pred, average="weighted", labels=class_names),
                    2
                ),
            )
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # ----------------------------
    # Raw Data Viewer
    # ----------------------------
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Credit Data Set (Classification)")
        st.subheader("{Fully Paid:0, Charged Off/Defaulted:1}")
        st.write(df)
        st.write(df.info())
        st.write(df.describe())
        st.write(df.defaulted.value_counts())
        corr_mt(df)


if __name__ == "__main__":
    main()
