import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import streamlit as st  # <-- Streamlit ko import kiya

# --- Streamlit Page ki Config ---
st.set_page_config(page_title="ENERGY THEFT DETECTOR", layout="wide")
st.title("💡 Electric Energy Theft Anomaly Detector ")

# --- Functions (Streamlit ke liye modified) ---

# @st.cache_data Streamlit ko batata hai ki function ko dobara run mat karo
# agar input file change nahi hui hai. Isse app fast chalti hai!
@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Uploaded file ko load karta hai, empty columns aur missing rows hatata hai.
    """
    st.info("Data load aur clean ho raha hai...")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Data load karne mein error: {e}")
        return None

    original_shape = df.shape
    
    # Poore empty columns ko drop karo
    df = df.dropna(axis='columns', how='all')
    
    # Bachi hui missing values wali rows ko drop karo
    df = df.dropna()
    
    st.write(f"Original shape: {original_shape} -> Cleaned shape: {df.shape}")
    return df

@st.cache_data
def feature_engineer(df):
    """
    Timestamp column se time-based features (hour, day, month) banata hai.
    """
    st.info("Time features banaye ja rahe hain (hour, day, month)...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month
    
    return df

def build_preprocessing_pipeline(df):
    """
    Numerical/Categorical features pehchanta hai aur preprocessing pipeline banata hai.
    """
    features_to_drop = ['USER ID', 'Timestamp']
    
    numerical_features = df.drop(columns=features_to_drop).select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.drop(columns=features_to_drop).select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, features_to_drop

# --- Main App Function ---
def main():
    
    # --- 1. Sidebar (jahaan settings hongi) ---
    with st.sidebar:
        st.header("(Settings)")
        
        # --- YEH HAI AAPKA "UPLOAD" BUTTON ---
        uploaded_file = st.file_uploader("Apni .csv file yahaan upload karein", type=["csv"])
        
        # --- Model parameter ke liye slider ---
        st.subheader("Model Parameters")
        CONTAMINATION_RATE = st.slider(
            "Contamination Rate (Kitne % anomalies ho sakti hain):",
            min_value=0.001, 
            max_value=0.1, 
            value=0.01,  # Default value
            step=0.005,
            format="%.3f"
        )
        st.caption("Value jitni high hogi, utni zyaada anomalies detect hongi.")

    # --- 2. Main Logic: Sirf tabhi run hoga jab file upload hogi ---
    if uploaded_file is not None:
        
        df = load_and_clean_data(uploaded_file)
        if df is None:
            return

        df_engineered = feature_engineer(df.copy())
        
        preprocessor, features_to_drop = build_preprocessing_pipeline(df_engineered)
        
        X = df_engineered.drop(columns=features_to_drop)
        
        # --- Preprocessing ---
        with st.spinner("Data ko process kiya ja raha hai..."):
            X_processed = preprocessor.fit_transform(X)
        
        # --- Model Training ---
        with st.spinner(f"Model train ho raha hai (Contamination={CONTAMINATION_RATE})..."):
            model = IsolationForest(
                contamination=CONTAMINATION_RATE, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_processed)
        
        # --- Prediction & Analysis ---
        st.info("Anomalies dhoondi ja rahi hain...")
        predictions = model.predict(X_processed)
        anomaly_scores = model.decision_function(X_processed)
        
        df_engineered['anomaly_prediction'] = predictions
        df_engineered['anomaly_score'] = anomaly_scores
        
        # --- 3. Results Dikhana ---
        st.subheader("Anomaly Detection ke Results")
        
        num_anomalies = (df_engineered['anomaly_prediction'] == -1).sum()
        percent_anomalies = num_anomalies / len(df_engineered) * 100
        
        st.success(f"Detection poora hua! **{num_anomalies}** anomalies mili hain.")
        st.write(f"Yeh aapke data ka ~**{percent_anomalies:.2f}%** hai.")
        
        # Anomaly score ke hisaab se sort karna
        df_anomalies = df_engineered.sort_values(by='anomaly_score').reset_index(drop=True)
        
        columns_to_show = [
            'Timestamp', 
            'USER ID', 
            'Power_Consumption_kWh', 
            'anomaly_score',
            'Voltage_V', 
            'Current_A', 
            'Power_Factor',
            'hour'
        ]
        
        # Top 10 results dikhana
        st.subheader("Top 10 Sabse Zyada Suspicious Records (Potential Theft)")
        st.dataframe(df_anomalies[columns_to_show].head(10))
        
        # Saari anomalies dikhana
        st.subheader("Saare Detected Anomaly Records")
        st.dataframe(df_anomalies[df_anomalies['anomaly_prediction'] == -1][columns_to_show])

    else:
        st.warning("Analysis shuru karne ke liye, Ek CSV file upload karein.")

if __name__ == "__main__":

    main()
