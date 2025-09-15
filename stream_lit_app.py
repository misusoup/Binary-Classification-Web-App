import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# -------------------------------
# Main function
# -------------------------------
def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Are your mushrooms edible or poisonous?')
    st.sidebar.markdown('Are your mushrooms edible or poisonous?')

    @st.cache_data(persist=True)
    def load_data():
        df = pd.read_csv('mushrooms.csv')  # Kaggle file in your repo
        df = df.replace('?', df.mode().iloc[0])  # Replace missing values
        # Encode categorical columns
        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])
        return df

    @st.cache_data(persist=True)
    def split(df):
        y = df['class']  # target column
        X = df.drop(columns=['class'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list, model, X_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names)
            st.pyplot(plt.gcf())
        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(plt.gcf())
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            st.pyplot(plt.gcf())

    # Load and split data
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    # Sidebar - Classifier selection
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Logistic Regression", "Random Forest"))

    # -------------------------------
    # SVM
    # -------------------------------
    if classifier == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button("Classify", key='Classify_SVM'):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(model.score(X_test, y_test), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    # -------------------------------
    # Logistic Regression
    # -------------------------------
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button("Classify", key='Classify_LR'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(model.score(X_test, y_test), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    # -------------------------------
    # Random Forest
    # -------------------------------
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("Max depth", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples?", ('True', 'False'), key='bootstrap')
        bootstrap = True if bootstrap == 'True' else False
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        if st.sidebar.button("Classify", key='Classify_RF'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", round(model.score(X_test, y_test), 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    # Show raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

# Run the app
if __name__ == '__main__':
    main()
