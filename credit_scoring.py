import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score



def main():
    st.title("Credit Scoring")
    st.sidebar.title("Credit Scoring Web App")
    st.markdown("Credit analysis")
    

    @st.cache(persist=True)
    def load_data():
        url = 'https://github.com/fedeghigo/Credit_Risk.Machine_learning-application/releases/download/credit_scoring.ml/LendingClub.csv'
        data = pd.read_csv(url)

        #data = pd.read_csv(r"C:\Users\federico\Documents\Python lending club\LendingClub.csv")
        #data = pd.read_csv(r"C:\Users\federico\Desktop\Markovitz\credit_scoring_v1\LendingClub.csv")
        #cols = ['loan_amnt', 'term',  'funded_amnt',  'annual_inc', 'dti', 'delinq_2yrs', 'last_pymnt_amnt', 'emp_length','tax_liens']
        cols = ['loan_amnt', 'term', 'int_rate', 'funded_amnt', 'grade', 'annual_inc', 'dti', 'hardship_loan_status', 'delinq_2yrs', 'last_pymnt_amnt', 'emp_length','loan_status','home_ownership','tax_liens']
        data = data[cols]
        data = data.drop(['hardship_loan_status'], axis=1)
        data =data.dropna()
        data=data[(data["loan_status"]=="Fully Paid") | (data['loan_status'] == 'Charged Off')]
        binary={"Fully Paid":0, "Charged Off":1}
        data["defaulted"]=data.loan_status.map(binary)
        data=data.drop("loan_status",axis=1)
        emp_map={'7 years':7, '4 years':4, '1 year':1, '3 years':3, '< 1 year':0, '6 years':6,
        '5 years':5, '2 years':2, '10+ years':10, '9 years':9, '8 years':8}
        data.emp_length=data.emp_length.map(emp_map)
        grade_map={'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        data.grade=data.grade.map(grade_map)
        
        data.term = pd.to_numeric(data.term.str.slice(1,3))
        home_map={'MORTGAGE':1, 'RENT':2, 'OWN':3, 'ANY':4, 'OTHER':5, 'NONE':6}
        data.home_ownership=data.home_ownership.map(home_map)
        print(data)
        print(data.info)
        print(data.describe())
        print(data.defaulted.value_counts())
        return data
    
    
        

    def corr_mt(df):
        corr_mtx = df.corr()
        
        plt.figure(figsize = (10,10))

        sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
        return st.pyplot()




    @st.cache(persist=True)
    def split(df):
        x=df.iloc[:,0:11]
        y=df.iloc[:,12]
        #y = df.type
        #x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names,normalize="pred")
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            
            st.pyplot()

    df = load_data()
    #class_names = ['edible', 'poisonous']
    # {"Fully Paid":0, "Charged Off/Defaulted":1}
    class_names=[0, 1]
    x_train, x_test, y_train, y_test = split(df)
   

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Random Forest", "Logistic Regression","Linear Discriminant Analysis" ))

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=1, key='n_estimators')
        max_features=st.sidebar.radio("Max Features", ('auto', 'sqrt'), key='max_features')
        #min_samples_leaf=st.sidebar.number_input("min_samples_leaf", 1, 20, step=1, key='min_samples_leaf')
        #min_samples_split=st.sidebar.number_input("min_samples_split", 1, 20, step=1, key='min_samples_split')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(ccp_alpha=0.0,random_state = 42,n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,max_features=max_features,            n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))

            #cross_val=cross_val_score(estimator=model, X=x_train, y=y_train, cv=5)
            #st.write("Cv k=5", cross_val)
            #st.write("Cv mean ",cross_val.mean())
            #st.write("Cv std.dev ",cross_val.std())

            #st.write("Cross Validation 5k:",cross_val_score(model, x, y, cv=5))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        solver = st.sidebar.radio("Which solver?", ("newton-cg", "lbfgs", "liblinear", "sag", "saga"), key='solver')
        penalty = st.sidebar.radio("Which penalty?", ("l1", "l2"), key='penalty')

        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 100.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        

        




        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty= penalty, max_iter=max_iter, solver= solver)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test,)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names, average=None).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            #st.write("Cross Validation 5k:",cross_val_score(model, x, y, cv=5))
            plot_metrics(metrics)
  
  
  
    if classifier == "Linear Discriminant Analysis":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 1.0, step=0.01, key='C_LR')
        #max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        n_components=st.sidebar.number_input("Number of dimension", 1, 8, step=1, key='dim')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Linear Discriminant Analysis Results")
            model = LinearDiscriminantAnalysis(solver="eigen",shrinkage=C,n_components=n_components)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            

            st.write("Accuracy: ", accuracy.round(2))                   #labels=class_names
            st.write("Precision: ", precision_score(y_test, y_pred.round(),average="weighted",labels=class_names ).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred.round(),average="weighted",labels=class_names).round(2))
            #st.write("Cross Validation 5k:",cross_val_score(model, x, y, cv=5))
            plot_metrics(metrics)



   
   
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Credit Data Set (Classification)")
        st.subheader("{Fully Paid:0, Charged Off/Defaulted:1}")
        st.write(df)
        st.write(df.info())
        st.write(df.describe())
        st.write(df.defaulted.value_counts())
        st.write(corr_mt(df))
if __name__ == '__main__':
    main()


