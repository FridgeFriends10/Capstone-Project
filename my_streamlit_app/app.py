import streamlit as st, pandas as pd, pickle

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Page background + all labels/titles in white */
    .stApp {
        background-color: #40E0D0;
    }
    h1, .stMarkdown, label {
        color: #ffffff !important;
    }

    /* Input‑box text (number & text fields) */
    .stTextInput input, .stNumberInput input {
        color: #888 !important;
    }

    /* Selectbox selected value + placeholder */
    .stSelectbox .css-1uccc91-singleValue,
    .stSelectbox .css-14el2xx-placeholder {
        color: #888 !important;
    }

    /* Dropdown menu items */
    .stSelectbox .css-26l3qy-menu div {
        color: #888 !important;
    }

    /* Slider value */
    .stSlider > div > input {
        color: #888 !important;
    }

    /* Predict button */
    div.stButton > button {
        background-color: #ff0000 !important;
        color: #ffffff !important;
        border-radius: 4px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)



# --- LOAD MODELS ---
with open("models.pkl","rb") as f:
    m = pickle.load(f)
clf, reg, le_comp, encoders = m["clf"], m["reg"], m["le_comp"], m["label_encoders"]

# --- PREDICTION FUNCTION ---
def predict_outcomes(age, gender, operation, method, surgery_type, asa_score):
    df = pd.DataFrame({
        'Age':[age], 'Gender':[gender],
        'Operation':[operation], 'Method':[method],
        'Surgery type':[surgery_type], 'ASA Score':[asa_score]
    })
    for c, le in encoders.items():
        df[c] = le.transform(df[c])
    probs = clf.predict_proba(df)[0]
    mapping = {i:lab for i,lab in enumerate(le_comp.classes_)}
    return {mapping[i]: probs[i] for i in range(len(probs))}, reg.predict(df)[0]

# --- STREAMLIT APP UI ---
st.title("Surgery Outcome Predictor")
age = st.number_input("Age", 0, 120, 45)
gender = st.selectbox("Gender", ["Male","Female"])
operation = st.selectbox("Operation", ["Cholecystectomy","Hemicolectomy"])
method = st.selectbox("Method", ["Laparoscopic","Open"])
surgery_type = st.selectbox("Surgery Type", ["Planned","Urgent"])
asa_score = st.slider("ASA Score", 1, 5, 2)

if st.button("Predict"):
    probs, los = predict_outcomes(age, gender, operation, method, surgery_type, asa_score)
    st.subheader("Predicted Complication Probabilities:")
    for comp, prob in probs.items():
        st.write(f"{comp}: {prob*100:.1f}%")
    st.subheader("Estimated Hospital Stay:")
    st.write(f"{los:.1f} days")
