import streamlit as st
import pandas as pd
# import numpy as np
# from bokeh.layouts import column
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
# from statsmodels.sandbox.panel.sandwich_covariance_generic import kernel


def main():
    st.set_page_config(
        page_title="Model Classification",
        page_icon="🧠",  # or use an emoji or a path to a .png/.ico
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Binary Classification Web App")
    st.sidebar.subheader("Binary Classification Web App")
    st.markdown("Are your mushroom edible or poisonous ?🍄🍄")
    st.sidebar.markdown("Are your mushroom edible or poisonous ?🍄🍄")
    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label_encoder = LabelEncoder()
        for column in data.columns:
            data[column] = label_encoder.fit_transform(data[column])
        return data

    @st.cache_data
    def split_data(data):
        y= data.type
        X= data.drop(columns=["type"])
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names).plot(ax=ax)
            st.pyplot(fig)

        if 'Precision Recall Curve' in metrics_list:
            st.subheader("Precision Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test).plot(ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test).plot(ax=ax)
            st.pyplot(fig)


    df=load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine(SVM)","Random Forest","Logistic Regression"))

    if classifier=="Support Vector Machine(SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C= st.sidebar.number_input("C (Regularization parameter)", min_value=0.0, max_value=10.0,step=0.01,key="C")
        kernel=st.sidebar.radio("Kernel",("linear","rbf"),key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernal Coefficient)", ("auto","scale"), key="gamma")
        matrices = st.sidebar.multiselect('What matrix to plot',('Confusion Matrix','Precision Recall Curve','ROC Curve'),key="matrices_svm")
        if st.sidebar.button("Classify",key="classify"):
            st.subheader("Support Vector Machine(SVM) Classification")
            model = SVC(kernel=kernel,C=C,gamma=gamma)
            model.fit(X_train,y_train)
            y_predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            st.write("Accuracy:",round(accuracy,2))
            st.write("Precision:",round(precision_score(y_test,y_predictions,labels=class_names),2))
            st.write("Recall:",round(recall_score(y_test,y_predictions,labels=class_names),2))
            plot_metrics(matrices)

    if classifier=="Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C= st.sidebar.number_input("C (Regularization parameter)", min_value=0.0, max_value=10.0,step=0.01,key="C_LR")
        max_iter = st.sidebar.slider("Maximum number of iterations", min_value=100, max_value=500, step=10, key="max_iter")
        matrices = st.sidebar.multiselect('What matrix to plot',('Confusion Matrix','Precision Recall Curve','ROC Curve'),key="matrices_LR")
        if st.sidebar.button("Classify",key="classify_LR"):
            st.subheader("Logistic Regression Classification")
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            y_predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            st.write("Accuracy:",round(accuracy,2))
            st.write("Precision:",round(precision_score(y_test,y_predictions,labels=class_names),2))
            st.write("Recall:",round(recall_score(y_test,y_predictions,labels=class_names),2))
            plot_metrics(matrices)

    if classifier=="Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimetor= st.sidebar.number_input("The no of trees in forest", min_value=100, max_value=5000, step=10, key="n_est")
        max_depth= st.sidebar.number_input("Maximum depth of tree", min_value=1, max_value=20, step=1, key="max_depth")
        bootstrap=st.sidebar.radio("Bootstrap sample",("True","False"),key="bootstrap")
        matrices = st.sidebar.multiselect('What matrix to plot',('Confusion Matrix','Precision Recall Curve','ROC Curve'))
        if st.sidebar.button("Classify",key="classify_rf"):
            st.subheader("Random Forest Classification")
            model = RandomForestClassifier(n_estimators=n_estimetor, max_depth=max_depth, bootstrap=bootstrap=="True",n_jobs=-1)
            model.fit(X_train,y_train)
            y_predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            st.write("Accuracy:",round(accuracy,2))
            st.write("Precision:",round(precision_score(y_test,y_predictions,labels=class_names),2))
            st.write("Recall:",round(recall_score(y_test,y_predictions,labels=class_names),2))
            plot_metrics(matrices)

    if st.sidebar.checkbox( label="Show Dataset",value=False):
        st.dataframe(df)



if __name__ == '__main__':
    main()


