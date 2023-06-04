import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import time
import requests

#Import Libraries
# Math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Pre-Preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Modeling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

# Evaluation and comparision of all the models

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,auc,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve 

st.set_page_config(page_title="Breast cancer Detection", page_icon=None, layout='centered', initial_sidebar_state='auto')
# Load data
data = pd.read_csv('data.csv') 
# Remove 'Unnamed: 32' column
data = data.drop('Unnamed: 32', axis=1)
 
with st.sidebar:
    choose = option_menu("Main Menu", ["Home", "Data", "Models", "Accuracy"],
                         icons=['house', 'files', 'kanban', 'easel'],
                         menu_icon="display", default_index=0, 
                         styles={
        "container": {"padding": "5!important", "background-color": "#213555"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#F3BCC8"},
        "nav-link-selected": {"background-color": "#E893CF"},
    
    }
    )
    
if choose == "Home":
    st.markdown("<h1 style='text-align: center; color: pink;'>Breast Cancer Detection</h1>", unsafe_allow_html=True)
    st.image("breast-cancer-prevention-detection-thumb.jpg",use_column_width = "auto")
    st.divider()
    st.markdown(
        """
        Breast cancer is a type of cancer that develops in the breast cells. It primarily affects women, but men can also develop breast cancer, although it is less common. 
	Breast cancer is characterized by the uncontrolled growth and division of abnormal cells in the breast tissue.""")
    st.markdown("""  	Here are some key points to describe breast cancer:""")
    st.markdown(""" #### Types of Breast Cancer: """)
    st.markdown( """
          
        - **Ductal Carcinoma**: The most common type, starts in the milk ducts.
        - **Lobular Carcinoma**: Begins in the milk-producing glands (lobules).
        - **Invasive vs. In Situ**: Breast cancer can be invasive, meaning it spreads to nearby tissues, or in situ, confined to the original site.
        - **Lobular Carcinoma**: Begins in the milk-producing glands (lobules). 
        #### Risk Factors:""")
    st.markdown( """               Several risk factors are associated with an increased likelihood of developing breast cancer, including:

        - **Gender**: Being a woman is the primary risk factor.
        - **Age**: The risk of breast cancer increases with age.
        - **Family History**: Having a family history of breast cancer, especially in first-degree relatives.
        - **Genetic Mutations**: Inherited mutations in genes such as BRCA1 and BRCA2.
        - **Hormonal Factors**: Early menstruation, late menopause, hormone replacement therapy, etc.
        - **Lifestyle Factors**: Obesity, alcohol consumption, lack of physical activity, etc. 
    """
    )
    st.markdown(""" #### Symptoms and Detection: """)
    st.markdown("""  Common symptoms of breast cancer may include a lump or thickening in the breast, changes in breast size or shape, nipple abnormalities, skin changes, and breast pain. Early detection is crucial and can be done through regular breast self-exams, clinical breast exams, mammography, and other imaging techniques.""")
    st.markdown(""" #### Diagnosis: """)
    st.markdown("""  If an abnormality is detected, further diagnostic tests may be conducted, including breast biopsy, which involves removing a sample of tissue for laboratory analysis. Imaging tests like ultrasound, MRI, or PET scans can help determine the extent of the cancer.""")
    st.markdown(""" #### Stages and Treatment: """)
    st.markdown("""   Breast cancer is categorized into stages based on the size of the tumor, lymph node involvement, and metastasis. Treatment options depend on the stage and may include surgery (lumpectomy or mastectomy), radiation therapy, chemotherapy, hormone therapy, targeted therapy, or a combination of these.""")
    st.markdown(""" #### Prognosis: """)
    st.markdown("""  Prognosis varies depending on several factors, including the stage, type of breast cancer, age, overall health, and response to treatment. Early detection and treatment increase the chances of successful outcomes.""")
    st.markdown("""  It's important to note that each individual's experience with breast cancer is unique, and treatment plans are personalized based on various factors. Regular screenings, awareness, and understanding the risk factors can help in the early detection and management of breast cancer.""")

    
elif choose == "Data":
    # Set up Streamlit app
    st.title('Data Processing and Visualizing the Data')
        
    # Display the raw data
    st.subheader('Raw Data')
    st.write(data)
       
    # Selectbox for visualization options
    visualization_option = st.selectbox('Select Visualization', ('Countplot - Diagnosis', 'Histogram'))
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    # Countplot creation
    if visualization_option == 'Countplot - Diagnosis':
        st.subheader('Countplot - Diagnosis')
        
        # Create the countplot using Seaborn
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='diagnosis', ax=ax)
        
        # Display the plot using Streamlit
        st.pyplot(fig)

    # Histogram creation
    elif visualization_option == 'Histogram':
        st.subheader('Histogram')
        
        # Create the histogram using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))
        data.hist(bins=50, ax=ax)
        plt.tight_layout()
        
        # Display the plot using Streamlit
        st.pyplot(fig)
    
    # Convert 'diagnosis' column to categorical codes
    data['diagnosis'] = data['diagnosis'].astype('category').cat.codes

    # Correlation matrix
    st.subheader('Correlation Matrix')

    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Display the correlation matrix
    st.write(correlation_matrix)
    
    # Define the center value for the heatmaps
    center_value = 0  # Replace with your desired center value

    # Selectbox for choosing the heatmap type
    heatmap_type = st.selectbox('Select Heatmap Type', ('Mean Features', 'Standard Error Features', 'Worst Features', 'All Features'))
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    # Visualize correlation for mean features using a heatmap
    if heatmap_type == 'Mean Features':
        st.subheader('Correlation Heatmap -- Mean')

        # Select only the mean features columns
        mean_features = data.iloc[:, 2:12]

        # Calculate the correlation matrix for mean features
        corr_matrix = mean_features.corr()

        # Create a heatmap using Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, center=center_value, ax=ax)

        # Display the heatmap using Streamlit
        st.pyplot(fig)

    # Visualize correlation for standard error features using a heatmap
    elif heatmap_type == 'Standard Error Features':
        st.subheader('Correlation Heatmap -- Standard Error')

        # Select only the standard error features columns
        standard_error_features = data.iloc[:, 12:22]

        # Calculate the correlation matrix for standard error features
        corr_matrix = standard_error_features.corr()

        # Create a heatmap using Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, center=center_value, ax=ax)

        # Display the heatmap using Streamlit
        st.pyplot(fig)

    # Visualize correlation for worst features using a heatmap
    elif heatmap_type == 'Worst Features':
        st.subheader('Correlation Heatmap -- Worst')

        # Select only the worst features columns
        worst_features = data.iloc[:, 22:32]

        # Calculate the correlation matrix for worst features
        corr_matrix = worst_features.corr()

        # Create a heatmap using Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, center=center_value, ax=ax)

        # Display the heatmap using Streamlit
        st.pyplot(fig)

    # Visualize correlation for all features using a heatmap
    else:
        st.subheader('Correlation Heatmap')

        # Calculate the correlation matrix for all features
        corr_matrix = data.corr()

        # Create a heatmap using Seaborn
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(corr_matrix, annot=True, center=center_value, ax=ax)

        # Display the heatmap using Streamlit
        st.pyplot(fig)
elif choose == "Models":  
        
    # Splitting dataset into independent and dependent data
    X = data.iloc[:, 2:32].values
    Y = data.iloc[:, 1].values

    # Label Encoding for dependent variable
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training and Prediction
    models = []
    algorithms = ["SVC", "DecisionTreeClassifier", "LogisticRegression", "KNeighborsClassifier", "XGB",
                  "RandomForestClassifier", "GradientBoostingClassifier", "GaussianNB"]
    classifiers = [SVC(), DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier(), XGBClassifier(),
                   RandomForestClassifier(), GradientBoostingClassifier(), GaussianNB()]

    for algorithm, classifier in zip(algorithms, classifiers):
        if isinstance(classifier, LinearRegression):
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_pred = [round(val) for val in y_pred]  # Round predictions to the nearest integer for comparison
        else:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        models.append((algorithm, accuracy))


    # Selectbox for confusion matrix
    models_dict = {
        "Support Vector Classifier (SVC)": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "KNeighbors Classifier": KNeighborsClassifier(),
        "XGBoost Classifier": XGBClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        }

    selected_model = st.selectbox('Select Model for Confusion Matrix', options=models_dict.keys())

    # Display algorithm explanation
    st.subheader(selected_model + " Algorithm")
    if selected_model == "Support Vector Classifier (SVC)":
        st.write("Support Vector Classifier is a powerful classification algorithm that works well with high-dimensional data. It tries to find the best hyperplane that separates different classes of data. It aims to maximize the margin between the classes, allowing for better generalization to unseen data.")
    elif selected_model == "Decision Tree Classifier":
        st.write("Decision Tree Classifier is a simple and interpretable algorithm that builds a tree-like model for decision-making. It splits the data based on different features to create branches and make predictions. The decision tree learns a set of if-then rules based on the training data, where each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents the class label.")
    elif selected_model == "Logistic Regression":
        st.write("Logistic Regression is a popular algorithm for binary classification. It models the relationship between the dependent variable and independent variables using the logistic function. Logistic Regression estimates the probability of an instance belonging to a particular class and makes predictions based on a decision boundary.")
    elif selected_model == "KNeighbors Classifier":
        st.write("K-Nearest Neighbors Classifier is a simple and versatile algorithm that classifies new data points based on the majority class of their k nearest neighbors. The value of k determines the number of neighbors considered for classification. KNN does not make any assumptions about the underlying data distribution and can handle multi-class classification tasks as well.")
    elif selected_model == "XGBoost Classifier":
        st.write("XGBoost (Extreme Gradient Boosting) is an optimized implementation of the gradient boosting algorithm. It is known for its excellent performance and scalability. XGBoost builds an ensemble of weak prediction models (usually decision trees) and iteratively improves the model by minimizing a specific loss function. It combines the predictions from multiple weak models to make the final prediction.")
    elif selected_model == "Random Forest Classifier":
        st.write("Random Forest is an ensemble learning algorithm that constructs a collection of decision trees and combines their predictions to make a final prediction. It introduces randomness in two ways: by using random subsets of the training data for each tree and by randomly selecting a subset of features for each split in the tree. Random Forest improves upon the decision tree algorithm by reducing overfitting and increasing robustness.")
    elif selected_model == "Gradient Boosting Classifier":
        st.write("Gradient Boosting Classifier is another ensemble learning algorithm that combines weak prediction models to make a final prediction. It builds the models in a sequential manner, where each new model corrects the mistakes made by the previous models. Gradient Boosting works by minimizing a loss function through gradient descent optimization, resulting in an overall strong predictive model.")
    elif selected_model == "Gaussian Naive Bayes":
        st.write("Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the assumption of independence between features. It models the likelihood of each class by assuming that the features follow a Gaussian distribution. Naive Bayes is simple, fast, and performs well on datasets with high dimensionality.")
    
    # Calculate and display confusion matrix
    st.subheader("Confusion Matrix for " + selected_model)
    model = models_dict[selected_model]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    confusion_mat = pd.DataFrame(cm, index=['cancer', 'healthy'], columns=['predicted_cancer', 'predicted_healthy'])
    st.write(confusion_mat)

    # Calculate and display accuracy for the selected model
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy for " + selected_model + ": ", accuracy)
elif choose == "Accuracy":
    # Set up Streamlit app
    st.title('Model Evaluation Metrics')

    # Splitting dataset into independent and dependent data
    X = data.iloc[:, 2:32].values
    Y = data.iloc[:, 1].values

    # Label Encoding for dependent variable
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Train and Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=101)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Copy and paste the code snippet with necessary modifications
    models = []
    models1 = []
    models2 = []
    models3 = []
    models4 = []

    # Define the classifiers list
    classifiers = [
        SVC(),
        DecisionTreeClassifier(),
        LogisticRegression(),
        KNeighborsClassifier(),
        XGBClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearRegression()
    ]

    # Define the algorithms list based on the classifier names
    algorithms = [type(classifier).__name__ for classifier in classifiers]

    for i in range(len(classifiers)):
        model = classifiers[i]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)  # Obtain binary predictions directly
        
        # Ensure y_test and pred contain only binary values
        binary_y_test = y_test.astype(int)
        binary_pred = pred.astype(int)
        
        models.append(accuracy_score(binary_y_test, binary_pred))
        models1.append(precision_score(binary_y_test, binary_pred))
        models2.append(recall_score(binary_y_test, binary_pred))
        models3.append(f1_score(binary_y_test, binary_pred))
        models4.append(roc_auc_score(binary_y_test, binary_pred))

    # Create a dictionary for the DataFrame
    d = {
        "Algorithm": algorithms,
        "Accuracy": models,
        "Precision": models1,
        "Recall": models2,
        "F1-Score": models3,
        "AUC": models4
    }

    # Create the DataFrame
    data_frame = pd.DataFrame(d)
    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Display the DataFrame
    st.subheader('Model Evaluation Metrics')
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(data_frame)

    # Add the bar plots in tabs
    st.subheader("Bar Plots")
    tabs = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    selected_tab = st.radio("Select Evaluation Metric", tabs)
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1, text=progress_text)

    if selected_tab == "Accuracy":
        fig, ax = plt.subplots()
        sns.barplot(x=data_frame['Accuracy'], y=data_frame['Algorithm'], palette="husl", ax=ax)
        ax.set_title('Accuracy of all Algorithms')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Algorithm')
        st.pyplot(fig)
    elif selected_tab == "Precision":
        fig, ax = plt.subplots()
        sns.barplot(x=data_frame['Precision'], y=data_frame['Algorithm'], palette="husl", ax=ax)
        ax.set_title('Precision of all Algorithms')
        ax.set_xlabel('Precision')
        ax.set_ylabel('Algorithm')
        st.pyplot(fig)
    elif selected_tab == "Recall":
        fig, ax = plt.subplots()
        sns.barplot(x=data_frame['Recall'], y=data_frame['Algorithm'], palette="husl", ax=ax)
        ax.set_title('Recall of all Algorithms')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Algorithm')
        st.pyplot(fig)
    elif selected_tab == "F1-Score":
        fig, ax = plt.subplots()
        sns.barplot(x=data_frame['F1-Score'], y=data_frame['Algorithm'], palette="husl", ax=ax)
        ax.set_title('F1-Score of all Algorithms')
        ax.set_xlabel('F1-Score')
        ax.set_ylabel('Algorithm')
        st.pyplot(fig)
    elif selected_tab == "AUC":
        fig, ax = plt.subplots()
        sns.barplot(x=data_frame['AUC'], y=data_frame['Algorithm'], palette="husl", ax=ax)
        ax.set_title('AUC of all Algorithms')
        ax.set_xlabel('AUC')
        ax.set_ylabel('Algorithm')
        st.pyplot(fig)
        
    

