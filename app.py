import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import streamlit as st

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="ENERGY THEFT DETECTOR", layout="wide")
st.title("💡 Electric Energy Theft Anomaly Detector")

# --- Functions (Modified for Streamlit) ---

# @st.cache_data tells Streamlit not to re-run the function
# if the input file hasn’t changed. This improves app performance.
@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Loads the uploaded file, removes empty columns and missing rows.
    """
    st.info("Loading and cleaning data...")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        return None

    original_shape = df.shape

    # Drop completely empty columns
    df = df.dropna(axis='columns', how='all')

    # Drop rows with remaining missing values
    df = df.dropna()

    st.write(f"Original shape: {original_shape} -> Cleaned shape: {df.shape}")
    return df


@st.cache_data
def feature_engineer(df):
    """
    Generates time-based features (hour, day, month) from the Timestamp column.
    """
    st.info("Creating time-based features (hour, day, month)...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month

    return df


def build_preprocessing_pipeline(df):
    """
    Identifies numerical and categorical features and creates a preprocessing pipeline.
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


# --- Main Application Function ---
def main():

    # --- 1. Sidebar (for user inputs and settings) ---
    with st.sidebar:
        st.header("(Settings)")

        # --- File Upload Button ---
        uploaded_file = st.file_uploader("Upload your .csv file here", type=["csv"])

        # --- Model Parameter Slider ---
        st.subheader("Model Parameters")
        CONTAMINATION_RATE = st.slider(
            "Contamination Rate (Estimated percentage of anomalies):",
            min_value=0.001,
            max_value=0.1,
            value=0.01,  # Default value
            step=0.005,
            format="%.3f"
        )
        st.caption("Higher values will detect more anomalies.")

    # --- 2. Main Logic: Runs only if a file is uploaded ---
    if uploaded_file is not None:

        df = load_and_clean_data(uploaded_file)
        if df is None:
            return

        df_engineered = feature_engineer(df.copy())

        preprocessor, features_to_drop = build_preprocessing_pipeline(df_engineered)

        X = df_engineered.drop(columns=features_to_drop)

        # --- Data Preprocessing ---
        with st.spinner("Processing data..."):
            X_processed = preprocessor.fit_transform(X)

        # --- Model Training ---
        with st.spinner(f"Training Isolation Forest model (Contamination={CONTAMINATION_RATE})..."):
            model = IsolationForest(
                contamination=CONTAMINATION_RATE,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_processed)

        # --- Prediction & Analysis ---
        st.info("Detecting anomalies...")
        predictions = model.predict(X_processed)
        anomaly_scores = model.decision_function(X_processed)

        df_engineered['anomaly_prediction'] = predictions
        df_engineered['anomaly_score'] = anomaly_scores

        # --- 3. Display Results ---
        st.subheader("Anomaly Detection Results")

        num_anomalies = (df_engineered['anomaly_prediction'] == -1).sum()
        percent_anomalies = num_anomalies / len(df_engineered) * 100

        st.success(f"Detection complete! **{num_anomalies}** anomalies found.")
        st.write(f"This represents approximately **{percent_anomalies:.2f}%** of your dataset.")

        # Sort by anomaly score
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

        # Display Top 10 Most Suspicious Records
        st.subheader("Top 10 Most Suspicious Records (Potential Energy Theft)")
        st.dataframe(df_anomalies[columns_to_show].head(10))

        # Display All Detected Anomalies
        st.subheader("All Detected Anomaly Records")
        st.dataframe(df_anomalies[df_anomalies['anomaly_prediction'] == -1][columns_to_show])

    else:
        st.warning("Please upload a CSV file to begin the analysis.")


if __name__ == "__main__":
    main()
