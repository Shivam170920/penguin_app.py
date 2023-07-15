# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

# Create a function that accepts 'model', island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g' and 'sex' as inputs and returns the species name.
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    data = pd.DataFrame({'island': [island],'bill_length_mm': [bill_length_mm],'bill_depth_mm': [bill_depth_mm],'flipper_length_mm': [flipper_length_mm],'body_mass_g': [body_mass_g],'sex': [sex]})

    prediction = model.predict(data)[0]

    species = ['Adelie', 'Chinstrap', 'Gentoo'][prediction]

    return species

# Design the App
st.title("Penguin Species Prediction")

bill_length_mm = st.sidebar.slider("Bill Length (mm)", min_value=32.1, max_value=59.6, value=45.0, step=0.1)
bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", min_value=13.1, max_value=21.5, value=17.0, step=0.1)
flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", min_value=172.0, max_value=231.0, value=200.0, step=1.0)
body_mass_g = st.sidebar.slider("Body Mass (g)", min_value=2700.0, max_value=6300.0, value=4200.0, step=100.0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
island = st.sidebar.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])

sex_mapping = {"Male": 0, "Female": 1}
island_mapping = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
sex_numeric = sex_mapping[sex]
island_numeric = island_mapping[island]

classifier = st.sidebar.selectbox("Classifier", ["SVC", "Logistic Regression", "Random Forest"])

if st.sidebar.button("Predict"):
    if classifier == "SVC":
        prediction = prediction(svc_model, island_numeric, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_numeric)
        score = svc_score
    elif classifier == "Logistic Regression":
        prediction = prediction(log_reg, island_numeric, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_numeric)
        score = log_reg_score
    elif classifier == "Random Forest":
        prediction = prediction(rf_clf, island_numeric, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_numeric)
        score = rf_clf_score

    st.write("Predicted Species:", prediction)
    st.write("Accuracy Score:", score)
