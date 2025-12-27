# ============================================
# CREDIT CARD FRAUD DETECTION SYSTEM
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
import tempfile
import sys

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 10px;
    }
    
    /* Cards */
    .card {
        background-color: #FFFFFF;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Model Cards */
    .model-card {
        border-left: 6px solid;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 0 10px 10px 0;
        background: #F8FAFC;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .cnn-card { 
        border-color: #EF4444;
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
    }
    
    .rf-card { 
        border-color: #10B981;
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    }
    
    .lr-card { 
        border-color: #3B82F6;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'cnn_history' not in st.session_state:
        st.session_state.cnn_history = None
    if 'fraud_detector' not in st.session_state:
        st.session_state.fraud_detector = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None

# ============================================
# MACHINE LEARNING MODEL CLASS
# ============================================
class FraudDetectionModels:
    """Credit Card Fraud Detection Machine Learning Models"""
    
    def __init__(self):
        """Initialize models and scalers"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.is_cnn_available = True
        
    # ============================================
    # DATA PROCESSING METHODS
    # ============================================
    
    def create_synthetic_data(self, n_samples=5000, fraud_percentage=1.0):
        """
        Create synthetic credit card transaction data for demonstration
        
        Parameters:
        -----------
        n_samples : int
            Number of transactions to generate
        fraud_percentage : float
            Percentage of fraudulent transactions (0-100)
            
        Returns:
        --------
        pd.DataFrame: Synthetic transaction data
        """
        np.random.seed(42)
        n_features = 30
        
        # Generate base features (V1-V28)
        X = np.random.randn(n_samples, n_features)
        
        # Calculate fraud indices
        n_fraud = int(n_samples * (fraud_percentage / 100))
        fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
        
        # Create fraud patterns
        for idx in fraud_indices:
            # Make first few features unusually high for fraud
            X[idx, :8] += np.random.uniform(2.5, 5.0, 8)
            # Make some features unusually low
            X[idx, 8:16] -= np.random.uniform(1.5, 3.5, 8)
        
        # Create target variable
        y = np.zeros(n_samples)
        y[fraud_indices] = 1
        
        # Add realistic noise
        X += np.random.randn(*X.shape) * 0.15
        
        # Create feature names
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        df['Class'] = y
        
        # Make Amount column realistic (log-normal distribution)
        df['Amount'] = np.exp(np.random.randn(n_samples) * 0.6 + 3.5)
        df.loc[fraud_indices, 'Amount'] *= np.random.uniform(3, 8, n_fraud)
        
        # Add time feature
        df['Time'] = np.random.uniform(0, 172800, n_samples)  # 48 hours in seconds
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the input dataframe for model training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with 'Class' column as target
            
        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        from sklearn.model_selection import train_test_split
        
        try:
            # Ensure 'Class' column exists
            if 'Class' not in df.columns:
                if 'class' in df.columns:
                    df.rename(columns={'class': 'Class'}, inplace=True)
                elif 'target' in df.columns:
                    df.rename(columns={'target': 'Class'}, inplace=True)
                else:
                    st.error("❌ Target column 'Class' not found in data")
                    return None, None, None, None, None
            
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class'].astype(int)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            feature_names = X.columns.tolist()
            
            # Split data (80-20 split)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=0.2, 
                random_state=42, 
                stratify=y
            )
            
            # Handle class imbalance with SMOTE
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.success("✅ Applied SMOTE for class balancing")
            except Exception as e:
                st.warning(f"⚠️ SMOTE failed: {str(e)}. Using original data.")
            
            return X_train, X_test, y_train, y_test, feature_names
            
        except Exception as e:
            st.error(f"❌ Error in data preprocessing: {str(e)}")
            return None, None, None, None, None
    
    # ============================================
    # MODEL TRAINING METHODS
    # ============================================
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Logistic Regression model"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
            
            st.info("📊 Training Logistic Regression...")
            
            # Train model
            lr_model = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.1,
                solver='liblinear',
                penalty='l2'
            )
            
            lr_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = lr_model.predict(X_test)
            y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Store model and results
            self.models['Logistic Regression'] = lr_model
            
            metrics = {
                'model': 'Logistic Regression',
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report
            }
            
            st.success("✅ Logistic Regression trained successfully!")
            return metrics
            
        except Exception as e:
            st.error(f"❌ Error training Logistic Regression: {str(e)}")
            return self._create_dummy_metrics('Logistic Regression')
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
            
            st.info("🌳 Training Random Forest...")
            
            # Train model (reduced parameters for speed)
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Store model and results
            self.models['Random Forest'] = rf_model
            
            metrics = {
                'model': 'Random Forest',
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report
            }
            
            st.success("✅ Random Forest trained successfully!")
            return metrics
            
        except Exception as e:
            st.error(f"❌ Error training Random Forest: {str(e)}")
            return self._create_dummy_metrics('Random Forest')
    
    def build_cnn_model(self, input_shape):
        """Build a 1D CNN model for tabular data"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Reshape
            from tensorflow.keras.optimizers import Adam
            
            # Check TensorFlow availability
            try:
                tf_version = tf.__version__
                st.info(f"🧠 TensorFlow {tf_version} detected")
            except:
                self.is_cnn_available = False
                return None
            
            # Build CNN model
            model = Sequential([
                # Reshape for Conv1D
                Reshape((input_shape[0], 1), input_shape=input_shape),
                
                # First convolutional block
                Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                # Second convolutional block
                Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                # Flatten and dense layers
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
            
            return model
            
        except Exception as e:
            st.warning(f"⚠️ CNN not available: {str(e)}")
            self.is_cnn_available = False
            return None
    
    def train_cnn(self, X_train, y_train, X_test, y_test):
        """Train and evaluate CNN model"""
        try:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
            
            st.info("🤖 Training Convolutional Neural Network...")
            
            # Check if CNN can be built
            if not self.is_cnn_available:
                st.warning("⚠️ CNN training skipped (TensorFlow not available)")
                return self._create_dummy_metrics('CNN'), None
            
            # Reshape data for CNN
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build model
            cnn_model = self.build_cnn_model((X_train.shape[1],))
            
            if cnn_model is None:
                st.warning("⚠️ CNN model could not be built")
                return self._create_dummy_metrics('CNN'), None
            
            # Train with progress updates
            import tensorflow as tf
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Custom training progress callback
            class TrainingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / 15
                    progress_bar.progress(progress)
                    status_text.text(f"📈 Epoch {epoch + 1}/15 - Loss: {logs['loss']:.4f}")
            
            # Train model
            history = cnn_model.fit(
                X_train_cnn, y_train,
                validation_split=0.2,
                epochs=15,
                batch_size=64,
                callbacks=callbacks + [TrainingCallback()],
                class_weight={0: 1., 1: 15.},
                verbose=0
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Make predictions
            y_pred_proba = cnn_model.predict(X_test_cnn, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Store model and results
            self.models['CNN'] = cnn_model
            
            metrics = {
                'model': 'CNN',
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report
            }
            
            st.success("✅ CNN trained successfully!")
            return metrics, history
            
        except Exception as e:
            st.error(f"❌ Error training CNN: {str(e)}")
            return self._create_dummy_metrics('CNN'), None
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _create_dummy_metrics(self, model_name):
        """Create placeholder metrics for failed models"""
        return {
            'model': model_name,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.5,
            'classification_report': {
                '0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                'accuracy': 0.0
            }
        }
    
    def predict_transaction(self, transaction_features):
        """
        Predict fraud probability for a single transaction
        
        Parameters:
        -----------
        transaction_features : np.array
            Array of transaction features
            
        Returns:
        --------
        float: Fraud probability (0-1)
        """
        try:
            # Scale features
            scaled_features = self.scaler.transform(transaction_features.reshape(1, -1))
            
            # Try CNN first if available
            if 'CNN' in self.models and self.models['CNN'] is not None:
                try:
                    # Reshape for CNN
                    cnn_features = scaled_features.reshape(1, scaled_features.shape[1], 1)
                    prediction = self.models['CNN'].predict(cnn_features, verbose=0)[0][0]
                    return float(prediction)
                except:
                    pass
            
            # Fallback to Random Forest
            if 'Random Forest' in self.models:
                prediction = self.models['Random Forest'].predict_proba(scaled_features)[0][1]
                return float(prediction)
            
            # Final fallback to Logistic Regression
            if 'Logistic Regression' in self.models:
                prediction = self.models['Logistic Regression'].predict_proba(scaled_features)[0][1]
                return float(prediction)
            
            return 0.5  # Default if no model available
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return 0.5

# ============================================
# STREAMLIT UI COMPONENTS
# ============================================

def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">💰 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #6B7280; font-size: 1.2rem; margin-bottom: 2rem;'>
            Advanced Machine Learning & Deep Learning Models for Real-time Fraud Detection
        </div>
        """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar controls"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: white;'>🔐 Fraud Detection</h2>
            <p style='color: rgba(255,255,255,0.8);'>Secure Transaction Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Configuration
        st.markdown("### 📁 **Data Configuration**")
        data_source = st.radio(
            "Select Data Source:",
            ["📊 Use Sample Data", "📤 Upload CSV File"],
            key="data_source"
        )
        
        if data_source == "📤 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your credit card transaction data (CSV format)",
                type=['csv'],
                help="Upload a CSV file with transaction data. Must include a 'Class' column (0=Legitimate, 1=Fraud)"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.current_data = df
                    st.success(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Show data statistics
                        if 'Class' in df.columns:
                            fraud_count = df['Class'].sum()
                            total_count = len(df)
                            fraud_percentage = (fraud_count / total_count) * 100
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Total Transactions", f"{total_count:,}")
                            col2.metric("Fraud Cases", f"{fraud_count:,}")
                            st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        else:
            # Sample data configuration
            col1, col2 = st.columns(2)
            with col1:
                n_samples = st.number_input(
                    "Sample Size",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000
                )
            with col2:
                fraud_rate = st.number_input(
                    "Fraud Rate (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f"
                )
            
            if st.button("🎲 Generate Sample Data", type="secondary", use_container_width=True):
                with st.spinner("Generating synthetic transaction data..."):
                    detector = FraudDetectionModels()
                    df = detector.create_synthetic_data(n_samples=n_samples, fraud_percentage=fraud_rate)
                    st.session_state.current_data = df
                    st.success(f"✅ Generated {n_samples} transactions with {fraud_rate}% fraud rate")
        
        st.markdown("---")
        
        # Model Training Configuration
        st.markdown("### 🤖 **Model Training**")
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='color: white; margin: 0;'>Select models to train:</p>
        </div>
        """, unsafe_allow_html=True)
        
        model_options = st.multiselect(
            "Choose ML Models:",
            ["Logistic Regression", "Random Forest", "Convolutional Neural Network (CNN)"],
            default=["Logistic Regression", "Random Forest", "Convolutional Neural Network (CNN)"],
            label_visibility="collapsed"
        )
        
        # Training button
        if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
            if st.session_state.current_data is not None:
                st.session_state.training_in_progress = True
            else:
                st.error("⚠️ Please load or generate data first!")
        
        st.markdown("---")
        
        # Real-time Prediction
        st.markdown("### 🔍 **Real-time Testing**")
        st.info("After training models, use the 'Fraud Detection' tab to test transactions.")
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🔄 Reset Application", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Footer
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: rgba(255,255,255,0.6); font-size: 0.9rem;'>
            <p>© 2024 Fraud Detection System</p>
            <p>v1.0.0</p>
        </div>
        """, unsafe_allow_html=True)

def display_dashboard():
    """Display main dashboard"""
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 **Please train models from the sidebar to see the dashboard**")
        return
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_count = len(st.session_state.results) if st.session_state.results else 0
        st.markdown(f'''
        <div class="metric-card">
            <h3>Trained Models</h3>
            <h2>{model_count}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.current_data is not None and 'Class' in st.session_state.current_data.columns:
            fraud_rate = st.session_state.current_data['Class'].mean() * 100
            st.markdown(f'''
            <div class="metric-card">
                <h3>Data Fraud Rate</h3>
                <h2>{fraud_rate:.2f}%</h2>
            </div>
            ''', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.results:
            best_model = max(st.session_state.results, key=lambda x: x.get('f1_score', 0))
            st.markdown(f'''
            <div class="metric-card">
                <h3>Best Model (F1)</h3>
                <h2>{best_model['model'][:12]}</h2>
            </div>
            ''', unsafe_allow_html=True)
    
    with col4:
        if st.session_state.current_data is not None:
            total_samples = len(st.session_state.current_data)
            st.markdown(f'''
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{total_samples:,}</h2>
            </div>
            ''', unsafe_allow_html=True)
    
    # Model Performance Comparison
    if st.session_state.results:
        st.markdown("### 📈 Model Performance Comparison")
        
        # Prepare data for visualization
        model_names = [r['model'] for r in st.session_state.results]
        accuracies = [r['accuracy'] for r in st.session_state.results]
        f1_scores = [r['f1_score'] for r in st.session_state.results]
        roc_aucs = [r['roc_auc'] for r in st.session_state.results]
        
        # Create comparison chart
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('📊 Accuracy', '🎯 F1-Score', '📈 ROC-AUC'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy',
                  marker_color='#3B82F6', text=[f'{x:.3f}' for x in accuracies],
                  textposition='auto'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name='F1-Score',
                  marker_color='#10B981', text=[f'{x:.3f}' for x in f1_scores],
                  textposition='auto'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=roc_aucs, name='ROC-AUC',
                  marker_color='#EF4444', text=[f'{x:.3f}' for x in roc_aucs],
                  textposition='auto'),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### 📋 Detailed Performance Metrics")
        
        detailed_metrics = []
        for result in st.session_state.results:
            report = result['classification_report']
            if '1' in report:
                detailed_metrics.append({
                    'Model': result['model'],
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'ROC-AUC': f"{result['roc_auc']:.4f}",
                    'Precision': f"{report['1']['precision']:.4f}",
                    'Recall': f"{report['1']['recall']:.4f}",
                    'Fraud Support': int(report['1']['support'])
                })
        
        # Display as styled dataframe
        if detailed_metrics:
            metrics_df = pd.DataFrame(detailed_metrics)
            st.dataframe(
                metrics_df.style
                .background_gradient(subset=['Accuracy', 'F1-Score', 'ROC-AUC'], cmap='Blues')
                .background_gradient(subset=['Precision', 'Recall'], cmap='Greens')
                .format({
                    'Accuracy': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'ROC-AUC': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}'
                }),
                use_container_width=True
            )

def display_model_details():
    """Display detailed model information"""
    st.markdown('<h2 class="sub-header">🤖 Model Architectures & Details</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 **Please train models first to see detailed information**")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CNN Details
        st.markdown("### 🧠 Convolutional Neural Network")
        st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
        
        st.markdown("""
        **Architecture:**
        - **Input Layer**: Reshape → (features, 1)
        - **Conv Block 1**: Conv1D(64 filters) → BatchNorm → MaxPool → Dropout(0.3)
        - **Conv Block 2**: Conv1D(128 filters) → BatchNorm → MaxPool → Dropout(0.3)
        - **Dense Layers**: Flatten → Dense(128) → Dropout(0.4) → Dense(64) → Dropout(0.3)
        - **Output Layer**: Dense(1, sigmoid)
        
        **Training:**
        - Optimizer: Adam (lr=0.001)
        - Loss: Binary Crossentropy
        - Metrics: Accuracy, Precision, Recall, AUC
        - Epochs: 15 with Early Stopping
        - Batch Size: 64
        
        **Advantages:**
        - Excellent for sequential/pattern data
        - Automatic feature extraction
        - High accuracy with enough data
        """)
        
        # Display CNN training history if available
        if st.session_state.cnn_history:
            history = st.session_state.cnn_history.history
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#EF4444', width=3)
            ))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='#3B82F6', width=3)
                ))
            
            fig.update_layout(
                title='📈 CNN Training History',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Random Forest Details
        st.markdown("### 🌳 Random Forest")
        st.markdown('<div class="model-card rf-card">', unsafe_allow_html=True)
        
        st.markdown("""
        **Configuration:**
        - Number of Trees: 100
        - Max Depth: 12
        - Min Samples Split: 5
        - Min Samples Leaf: 2
        - Class Weight: Balanced
        
        **Features:**
        - Ensemble of decision trees
        - Bootstrap aggregation
        - Feature importance ranking
        - Robust to outliers
        
        **Advantages:**
        - Handles non-linear relationships
        - Feature importance scores
        - Less prone to overfitting
        - Works well with imbalanced data
        """)
        
        # Feature importance visualization (if available)
        if 'Random Forest' in st.session_state.fraud_detector.models:
            try:
                model = st.session_state.fraud_detector.models['Random Forest']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-10:]  # Top 10 features
                    
                    fig = px.bar(
                        x=importances[indices],
                        y=[st.session_state.feature_names[i] for i in indices],
                        orientation='h',
                        title='🔝 Top 10 Important Features',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=importances[indices],
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=300,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logistic Regression Details
        st.markdown("### 📊 Logistic Regression")
        st.markdown('<div class="model-card lr-card">', unsafe_allow_html=True)
        
        st.markdown("""
        **Configuration:**
        - Regularization: L2 (C=0.1)
        - Max Iterations: 1000
        - Class Weight: Balanced
        - Solver: liblinear
        
        **Features:**
        - Linear classification
        - Probability outputs
        - L2 regularization
        - Fast training/inference
        
        **Advantages:**
        - Fast training and prediction
        - Good baseline model
        - Interpretable coefficients
        - Low computational cost
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_fraud_detection():
    """Display real-time fraud detection interface"""
    st.markdown('<h2 class="sub-header">🔍 Real-time Fraud Detection</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.info("👈 **Please train models first to enable fraud detection**")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Test Transaction Parameters")
        
        with st.form("transaction_test_form"):
            # Create two columns for better layout
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 💰 Transaction Details")
                amount = st.number_input(
                    "Transaction Amount ($)",
                    min_value=0.01,
                    max_value=100000.0,
                    value=150.0,
                    step=50.0,
                    format="%.2f",
                    help="Enter the transaction amount in USD"
                )
                
                time_of_day = st.slider(
                    "Time of Day (seconds)",
                    min_value=0,
                    max_value=86400,
                    value=43200,
                    step=3600,
                    help="Transaction time in seconds since midnight"
                )
            
            with col_b:
                st.markdown("#### 📊 Feature Values")
                v1 = st.slider("V1 (Feature 1)", -15.0, 15.0, 0.0, 0.5)
                v2 = st.slider("V2 (Feature 2)", -15.0, 15.0, 0.0, 0.5)
                v3 = st.slider("V3 (Feature 3)", -15.0, 15.0, 0.0, 0.5)
                v4 = st.slider("V4 (Feature 4)", -15.0, 15.0, 0.0, 0.5)
            
            # Model selection
            st.markdown("#### 🤖 Select Prediction Model")
            available_models = [r['model'] for r in st.session_state.results] if st.session_state.results else []
            selected_model = st.selectbox(
                "Choose model for prediction:",
                available_models,
                help="Select which trained model to use for prediction"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "🔍 Analyze Transaction for Fraud",
                type="primary",
                use_container_width=True
            )
        
        # Process form submission
        if submitted and st.session_state.fraud_detector:
            try:
                # Create transaction vector
                n_features = len(st.session_state.feature_names) if st.session_state.feature_names else 30
                transaction = np.zeros(n_features)
                
                # Set feature values
                transaction[0] = time_of_day  # Time
                if n_features > 1: transaction[1] = v1
                if n_features > 2: transaction[2] = v2
                if n_features > 3: transaction[3] = v3
                if n_features > 4: transaction[4] = v4
                
                # Set amount (find Amount column if exists)
                if st.session_state.feature_names and 'Amount' in st.session_state.feature_names:
                    amount_idx = st.session_state.feature_names.index('Amount')
                    transaction[amount_idx] = amount
                elif n_features > 29:
                    transaction[29] = amount
                
                # Store test data
                st.session_state.test_data = {
                    'features': transaction,
                    'amount': amount,
                    'selected_model': selected_model
                }
                
                # Make prediction
                fraud_probability = st.session_state.fraud_detector.predict_transaction(transaction)
                
                # Store result
                st.session_state.prediction_result = {
                    'probability': fraud_probability,
                    'is_fraud': fraud_probability > 0.5,
                    'model': selected_model,
                    'amount': amount,
                    'features': transaction[:5]  # Store first 5 features for display
                }
                
                # Success message
                st.success("✅ Transaction analyzed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error analyzing transaction: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Fraud Detection Results")
        
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            # Create fraud probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability'] * 100,
                title={
                    'text': "Fraud Probability",
                    'font': {'size': 24, 'color': '#1E3A8A'}
                },
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "darkblue"
                    },
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'green'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display decision and details
            if result['is_fraud']:
                st.error("""
                ## 🚨 **FRAUD DETECTED!**
                
                **Decision:** ⛔ **BLOCK TRANSACTION**
                
                **Immediate Actions Required:**
                1. Block transaction immediately
                2. Notify cardholder via SMS/Email
                3. Flag account for manual review
                4. Request additional authentication
                """)
            else:
                st.success("""
                ## ✅ **LEGITIMATE TRANSACTION**
                
                **Decision:** ✓ **APPROVE TRANSACTION**
                
                **Status:** Safe to proceed with normal processing
                
                **Recommendation:** Continue monitoring similar patterns
                """)
            
            # Transaction details card
            st.markdown("""
            <div class="card">
                <h4>📋 Transaction Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Amount", f"${result['amount']:,.2f}")
                st.metric("Confidence", f"{result['probability']*100:.1f}%")
            with col_b:
                st.metric("Model Used", result['model'])
                st.metric("Risk Level", "HIGH" if result['is_fraud'] else "LOW")
        
        else:
            # Placeholder when no prediction made yet
            st.info("""
            ## 🔍 **Ready for Analysis**
            
            **Instructions:**
            1. Enter transaction details in the left panel
            2. Select prediction model
            3. Click "Analyze Transaction"
            
            **The system will provide:**
            - Fraud probability (0-100%)
            - Risk assessment
            - Recommended action
            - Transaction details
            """)
    
    # Test with predefined scenarios
    st.markdown("---")
    st.markdown("### 🧪 **Test Predefined Scenarios**")
    
    col1, col2, col3 = st.columns(3)
    
    test_scenarios = [
        {
            "name": "Normal Purchase",
            "description": "Typical online shopping",
            "amount": 89.99,
            "v1": -1.2,
            "v2": 0.8,
            "v3": -0.5,
            "v4": 0.3,
            "color": "green"
        },
        {
            "name": "Suspicious Activity",
            "description": "Unusual large transaction",
            "amount": 2499.99,
            "v1": 4.5,
            "v2": -3.2,
            "v3": 5.1,
            "v4": -2.8,
            "color": "orange"
        },
        {
            "name": "High Risk Fraud",
            "description": "Clear fraud pattern",
            "amount": 5499.99,
            "v1": 9.8,
            "v2": -7.5,
            "v3": 11.2,
            "v4": -8.9,
            "color": "red"
        }
    ]
    
    for idx, scenario in enumerate(test_scenarios):
        with [col1, col2, col3][idx]:
            if st.button(
                f"Test: {scenario['name']}",
                key=f"scenario_{idx}",
                use_container_width=True,
                type="secondary" if scenario['color'] == 'green' else "primary"
            ):
                # Simulate form submission
                with st.spinner(f"Testing {scenario['name']}..."):
                    try:
                        n_features = len(st.session_state.feature_names) if st.session_state.feature_names else 30
                        transaction = np.zeros(n_features)
                        
                        # Set scenario values
                        transaction[0] = np.random.uniform(0, 86400)
                        if n_features > 1: transaction[1] = scenario['v1']
                        if n_features > 2: transaction[2] = scenario['v2']
                        if n_features > 3: transaction[3] = scenario['v3']
                        if n_features > 4: transaction[4] = scenario['v4']
                        
                        # Set amount
                        if st.session_state.feature_names and 'Amount' in st.session_state.feature_names:
                            amount_idx = st.session_state.feature_names.index('Amount')
                            transaction[amount_idx] = scenario['amount']
                        elif n_features > 29:
                            transaction[29] = scenario['amount']
                        
                        # Make prediction
                        fraud_prob = st.session_state.fraud_detector.predict_transaction(transaction)
                        
                        # Show result
                        if fraud_prob > 0.7:
                            st.error(f"**{scenario['name']}**\n\nFraud Risk: **{fraud_prob*100:.1f}%**\n\n⚠️ **HIGH RISK**")
                        elif fraud_prob > 0.3:
                            st.warning(f"**{scenario['name']}**\n\nFraud Risk: **{fraud_prob*100:.1f}%**\n\n⚠️ **MEDIUM RISK**")
                        else:
                            st.success(f"**{scenario['name']}**\n\nFraud Risk: **{fraud_prob*100:.1f}%**\n\n✅ **LOW RISK**")
                            
                    except Exception as e:
                        st.error(f"Error testing scenario: {str(e)}")

def display_data_analysis():
    """Display data analysis and visualizations"""
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    if st.session_state.current_data is None:
        st.info("👈 **Please load or generate data from the sidebar**")
        return
    
    df = st.session_state.current_data
    
    # Data Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 **Class Distribution**")
        
        # Find target column
        target_col = None
        for col in ['Class', 'class', 'target', 'is_fraud']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            class_counts = df[target_col].value_counts()
            labels = ['Legitimate', 'Fraud'] if len(class_counts) == 2 else ['Class 0', 'Class 1']
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=class_counts.values,
                hole=0.4,
                marker_colors=['#10B981', '#EF4444'],
                textinfo='label+percent',
                textposition='inside'
            )])
            
            fig.update_layout(
                height=350,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            fraud_count = df[target_col].sum()
            total_count = len(df)
            fraud_percentage = (fraud_count / total_count) * 100
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Legitimate", f"{total_count - fraud_count:,}")
            col_b.metric("Fraud", f"{fraud_count:,}")
            col_c.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
    
    with col2:
        st.markdown("### 💰 **Transaction Amount Analysis**")
        
        if 'Amount' in df.columns:
            # Create histogram
            fig = px.histogram(
                df,
                x='Amount',
                nbins=50,
                title="Transaction Amount Distribution",
                color_discrete_sequence=['#3B82F6'],
                opacity=0.8
            )
            
            fig.update_layout(
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Amount ($)",
                yaxis_title="Frequency",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Amount statistics
            amount_stats = df['Amount'].describe()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Mean", f"${amount_stats['mean']:,.2f}")
            col_b.metric("Std Dev", f"${amount_stats['std']:,.2f}")
            col_c.metric("Max", f"${amount_stats['max']:,.2f}")
        
        elif target_col:
            # If Amount not available, show feature correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 5:
                corr_matrix = df[numeric_cols[:5]].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    height=350,
                    title="Feature Correlation Matrix",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Data Statistics
    st.markdown("### 📋 **Dataset Statistics**")
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Number of Features", f"{df.shape[1]}")
    
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    with col4:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")
    
    # Data preview
    with st.expander("🔍 **View Raw Data**"):
        st.dataframe(df, use_container_width=True)
        
        # Data summary
        st.markdown("**Data Types:**")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df, use_container_width=True)

def train_models():
    """Train selected models with progress tracking"""
    if not st.session_state.current_data:
        st.error("❌ No data available for training")
        return
    
    if st.session_state.training_in_progress:
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Data Preparation
            status_text.text("📦 Step 1/5: Preparing data...")
            progress_bar.progress(10)
            
            # Initialize detector
            detector = FraudDetectionModels()
            
            # Save data to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                st.session_state.current_data.to_csv(tmp.name, index=False)
                temp_path = tmp.name
            
            # Step 2: Data Preprocessing
            status_text.text("🔧 Step 2/5: Preprocessing data...")
            progress_bar.progress(25)
            
            # Preprocess data
            X_train, X_test, y_train, y_test, feature_names = detector.preprocess_data(
                st.session_state.current_data
            )
            
            if X_train is None:
                st.error("❌ Data preprocessing failed")
                return
            
            # Step 3: Model Training
            status_text.text("🤖 Step 3/5: Training models...")
            progress_bar.progress(40)
            
            results = []
            model_options = ["Logistic Regression", "Random Forest", "Convolutional Neural Network (CNN)"]
            
            # Train Logistic Regression
            if "Logistic Regression" in model_options:
                status_text.text("📊 Training Logistic Regression...")
                lr_metrics = detector.train_logistic_regression(X_train, y_train, X_test, y_test)
                if lr_metrics:
                    results.append(lr_metrics)
                progress_bar.progress(55)
            
            # Train Random Forest
            if "Random Forest" in model_options:
                status_text.text("🌳 Training Random Forest...")
                rf_metrics = detector.train_random_forest(X_train, y_train, X_test, y_test)
                if rf_metrics:
                    results.append(rf_metrics)
                progress_bar.progress(70)
            
            # Train CNN
            cnn_history = None
            if "Convolutional Neural Network (CNN)" in model_options:
                status_text.text("🧠 Training CNN...")
                cnn_metrics, history = detector.train_cnn(X_train, y_train, X_test, y_test)
                if cnn_metrics:
                    results.append(cnn_metrics)
                    cnn_history = history
                progress_bar.progress(85)
            
            # Step 4: Store Results
            status_text.text("💾 Step 4/5: Storing results...")
            progress_bar.progress(95)
            
            # Store in session state
            st.session_state.fraud_detector = detector
            st.session_state.results = results
            st.session_state.models_trained = True
            st.session_state.feature_names = feature_names
            st.session_state.cnn_history = cnn_history
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Step 5: Completion
            status_text.text("✅ Step 5/5: Training completed!")
            progress_bar.progress(100)
            
            # Show success message
            st.balloons()
            st.success("""
            ## 🎉 Training Completed Successfully!
            
            **All selected models have been trained and evaluated.**
            
            **Next Steps:**
            1. Check the **Dashboard** tab for performance metrics
            2. View **Model Details** for architecture information
            3. Use **Fraud Detection** to test new transactions
            4. Explore **Data Analysis** for insights
            """)
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
        
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            st.session_state.training_in_progress = False

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application function"""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Train models if requested
    if st.session_state.training_in_progress:
        train_models()
    
    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🤖 Model Details", 
        "🔍 Fraud Detection",
        "📈 Data Analysis"
    ])
    
    # Display tab content
    with tab1:
        display_dashboard()
    
    with tab2:
        display_model_details()
    
    with tab3:
        display_fraud_detection()
    
    with tab4:
        display_data_analysis()

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
