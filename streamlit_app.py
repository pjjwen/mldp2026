import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HeartCheck AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

#layout styling
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f6f8fb;
    }

    label,
    p,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    [data-testid="stWidgetLabel"] {
        color: #111827 !important;
        font-weight: 600;
    }

    input {
        color: white !important;
    }

    div[data-baseweb="select"] {
        color: white !important;
    }  

        .main .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #8b1e3f, #c93f62);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(139, 30, 63, 0.18);
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.95;
            margin-bottom: 0;
        }

        .info-card {
            background-color: white;
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid #e6e9ef;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
            margin-bottom: 1rem;
        }

        .result-positive {
            background-color: #fff1f3;
            border-left: 6px solid #c93f62;
            padding: 1.3rem;
            border-radius: 12px;
            margin-top: 1rem;
        }

        .result-negative {
            background-color: #edf9f1;
            border-left: 6px solid #2e8b57;
            padding: 1.3rem;
            border-radius: 12px;
            margin-top: 1rem;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 650;
            min-height: 3rem;
        }

        [data-testid="stMetric"] {
            background-color: white;
            padding: 1rem;
            border: 1px solid #e6e9ef;
            border-radius: 14px;
        }

        .small-note {
            color: #667085;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#sidebar styling
with st.sidebar:
    st.title("HeartCheck AI")

    st.markdown(
        """
        This application uses a trained **Random Forest model** to estimate
        whether a patient may have heart disease.
        """
    )

    st.divider()

    st.subheader("How to use")

    st.markdown(
        """
        1. Enter the patient's details.
        2. Check that the information is correct.
        3. Select **Analyse heart disease risk**.
        4. Review the prediction and probability.
        """
    )

    st.divider()

    st.warning(
        "Educational demonstration only. This application does not replace "
        "professional medical assessment."
    )

#header
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Heart Disease Risk Assessment</div>
        <p class="hero-subtitle">
            Enter patient information to receive an AI-assisted prediction
            based on a trained Random Forest model.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_saved_files():
    model = joblib.load("rf_model.pkl")
    num_imputer = joblib.load("numeric_imputer.pkl")
    cat_imputer = joblib.load("categorical_imputer.pkl")
    model_columns = joblib.load("columns.pkl")

    return model, num_imputer, cat_imputer, model_columns


model, num_imputer, cat_imputer, model_columns = load_saved_files()


numeric_columns = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]

categorical_columns = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
    "age_group"
]

st.subheader("Patient information")
st.caption("Enter the patient's clinical details below.")

with st.form("prediction_form"):

    st.markdown("### Personal and clinical details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50,
            help="Patient's age in years."
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        dataset = st.selectbox(
            "Hospital dataset",
            [
                "Cleveland",
                "Hungary",
                "Switzerland",
                "VA Long Beach"
            ],
            help="The hospital group used in the original dataset."
        )

    with col2:
        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120
        )

        chol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=700,
            value=200
        )

        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dL?",
            [False, True],
            format_func=lambda value: "Yes" if value else "No"
        )

    with col3:
        cp = st.selectbox(
            "Chest pain type",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ]
        )

        restecg = st.selectbox(
            "Resting ECG result",
            [
                "normal",
                "st-t abnormality",
                "lv hypertrophy"
            ]
        )

        thal = st.selectbox(
            "Thalassemia result",
            [
                "normal",
                "fixed defect",
                "reversable defect"
            ]
        )

    st.divider()

    st.markdown("### Exercise-related details")

    col4, col5, col6 = st.columns(3)

    with col4:
        thalch = st.number_input(
            "Maximum heart rate achieved",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise-induced angina?",
            [False, True],
            format_func=lambda value: "Yes" if value else "No"
        )

    with col5:
        oldpeak = st.number_input(
            "ST depression",
            min_value=-5.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest."
        )

        slope = st.selectbox(
            "Peak exercise ST slope",
            [
                "upsloping",
                "flat",
                "downsloping"
            ]
        )

    with col6:
        ca = st.number_input(
            "Number of major vessels",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

    confirm = st.checkbox(
        "I confirm that the entered information has been checked."
    )

    submitted = st.form_submit_button(
        "Analyse heart disease risk",
        type="primary"
    )


    if submitted:
        validation_errors = []

        if not confirm:
            validation_errors.append(
                "Please confirm that the entered information has been checked."
                )

        if chol == 0:
            validation_errors.append(
            "Cholesterol cannot be zero. Please enter a valid measurement."
                )

        if trestbps < 70:
            validation_errors.append(
                "Resting blood pressure appears unusually low."
                )


        if validation_errors:
            st.error("Please correct the following information:")

            for error in validation_errors:
                st.warning(error)

    else:
        try:
            # EVERYTHING that can cause an error goes here

            if age < 40:
                age_group = "Below 40"
            elif age < 50:
                age_group = "40-49"
            elif age < 60:
                age_group = "50-59"
            elif age < 70:
                age_group = "60-69"
            else:
                age_group = "70 and above"


            input_data = pd.DataFrame({
                "age": [age],
                "sex": [sex],
                "dataset": [dataset],
                "cp": [cp],
                "trestbps": [trestbps],
                "chol": [chol],
                "fbs": [fbs],
                "restecg": [restecg],
                "thalch": [thalch],
                "exang": [exang],
                "oldpeak": [oldpeak],
                "slope": [slope],
                "ca": [ca],
                "thal": [thal],
                "age_group": [age_group]
            })


            input_data[numeric_columns] = num_imputer.transform(
                input_data[numeric_columns]
            )

            input_data[categorical_columns] = cat_imputer.transform(
                input_data[categorical_columns]
            )

            input_data = pd.get_dummies(
                input_data,
                columns=categorical_columns,
                dtype=int
            )

            input_data = input_data.reindex(
                columns=model_columns,
                fill_value=0
            )


            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]


            # Result display
            st.subheader("Prediction result")

            if prediction == 1:
                st.error(
                    "The model predicts that the patient may have heart disease."
                )
            else:
                st.success(
                    "The model predicts that the patient is unlikely to have heart disease."
                )

            st.metric(
                "Predicted probability of heart disease",
                f"{probability * 100:.1f}%"
            )


        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the prediction."
            )

            with st.expander("Technical details"):
                st.code(str(error))