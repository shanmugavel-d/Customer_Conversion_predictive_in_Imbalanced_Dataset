import streamlit as st
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_filename = "xgb_model.pkl"
with open(model_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("XGBoost Classifier App")

# User input section
st.header("User Input")

# Mapping dictionaries (including 'mon' mapping)
job_mapping = {
    'blue-collar': 0,
    'management': 7,
    'technician': 4,
    'admin.': 6,
    'services': 3,
    'retired': 9,
    'self-employed': 5,
    'entrepreneur': 1,
    'unemployed': 8,
    'housemaid': 2,
    'student': 10
}

marital_mapping = {
    'married': 0,
    'single': 2,
    'divorced': 1
}

education_mapping = {
    'secondary': 1,
    'tertiary': 2,
    'primary': 0
}

call_type_mapping = {
    'cellular': 2,
    'unknown': 0,
    'telephone': 1
}

prev_outcome_mapping = {
    'unknown': 0,
    'failure': 1,
    'other': 2,
    'success': 3
}

mon_mapping = {
    'may': 0,
    'jul': 1,
    'aug': 5,
    'jun': 4,
    'nov': 3,
    'apr': 7,
    'feb': 6,
    'jan': 2,
    'oct': 8,
    'sep': 9,
    'mar': 11,
    'dec': 10
}

# Reverse mapping dictionaries
reverse_job_mapping = {v: k for k, v in job_mapping.items()}
reverse_marital_mapping = {v: k for k, v in marital_mapping.items()}
reverse_education_mapping = {v: k for k, v in education_mapping.items()}
reverse_call_type_mapping = {v: k for k, v in call_type_mapping.items()}
reverse_prev_outcome_mapping = {v: k for k, v in prev_outcome_mapping.items()}
reverse_mon_mapping = {v: k for k, v in mon_mapping.items()}

# Get user input
import streamlit as st
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_filename = "xgb_model.pkl"
with open(model_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("XGBoost Classifier App")

# User input section in the sidebar
st.sidebar.header("User Input")

# Mapping dictionaries (including 'mon' mapping)
# ... (same as before)

# Reverse mapping dictionaries
# ... (same as before)

# Get user input
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
job = st.sidebar.selectbox("Job", list(reverse_job_mapping.values()))
marital = st.sidebar.selectbox("Marital Status", list(reverse_marital_mapping.values()))
education_qual = st.sidebar.selectbox("Education Qualification", list(reverse_education_mapping.values()))
call_type = st.sidebar.selectbox("Call Type", list(reverse_call_type_mapping.values()))
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1)
mon = st.sidebar.selectbox("Month", list(reverse_mon_mapping.values()))
dur_label = st.sidebar.number_input("Duration Label", min_value=-221.0, max_value=643.0, value=0.0)
num_calls = st.sidebar.number_input("Number of Calls", min_value=0, max_value=10, value=2)
prev_outcome = st.sidebar.selectbox("Previous Outcome", list(reverse_prev_outcome_mapping.values()))




# Convert categorical features to numerical representations
job_label = job_mapping[job]
marital_label = marital_mapping[marital]
education_label = education_mapping[education_qual]
call_type_label = call_type_mapping[call_type]
mon_label = mon_mapping[mon]
prev_outcome_label = prev_outcome_mapping[prev_outcome]

# Create input data as a list of lists
input_data = [[age, job_label, marital_label, education_label, call_type_label, day, mon_label, dur_label, num_calls, prev_outcome_label]]

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
scaled_input_data = scaler.fit_transform(input_data)

# Example: Make predictions using the loaded model and user input
# Preprocess the input data according to the model's requirements
# ...
if st.button("Get Prediction"):
    prediction_prob = loaded_model.predict_proba(scaled_input_data)
    prediction = loaded_model.predict(scaled_input_data)

    # Display prediction probability and prediction in bold
    st.markdown("**Prediction Probability:**")
    st.write(prediction_prob)
    st.markdown("**Prediction:**")
    st.write(f"**{prediction[0]}**")