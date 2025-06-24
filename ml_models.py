import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import streamlit as st

class AadhaarMLModels:
    def __init__(self, data):
        self.data = data

        # Prepare data and encode target
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)

    def prepare_data(self):
        """Prepare data for machine learning"""
        # Drop identifiers, target, and anything derived from target
        columns_to_drop = [
            'Date', 'Pincode', 'Total_Updates', 'Update_Volume_Category',
            'Year', 'Month', 'Pct_5_17', 'Pct_17+', 'Updates_Per_Pincode', 'Pincode_Count'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
        X = self.data.drop(columns=columns_to_drop, errors='ignore')

        # Keep only numeric features (State, District codes)
        X = X.select_dtypes(include=[np.number]).copy()

        # Target
        y = self.data['Update_Volume_Category'] if 'Update_Volume_Category' in self.data.columns else None

        # Encode target if needed
        if y is not None and (pd.api.types.is_categorical_dtype(y) or y.dtype == 'object' or not np.issubdtype(y.dtype, np.integer)):
            y = y.astype(str)
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = None

        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def scale_data(self, X):
        """Scale the data"""
        return self.scaler.transform(X)
    
    def decision_tree(self):
        """Decision Tree Classifier"""
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def random_forest(self):
        """Random Forest Classifier"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def gradient_boosting(self):
        """Gradient Boosting Classifier"""
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def svm(self):
        """Support Vector Machine"""
        X_train_scaled = self.scale_data(self.X_train)
        X_test_scaled = self.scale_data(self.X_test)
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train_scaled, self.y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def stochastic_gradient_descent(self):
        """Stochastic Gradient Descent Classifier"""
        X_train_scaled = self.scale_data(self.X_train)
        X_test_scaled = self.scale_data(self.X_test)
        
        model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, self.y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def xgboost(self):
        """XGBoost Classifier"""
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return model, accuracy, report
    
    def deep_learning(self):
        """Deep Neural Network"""
        X_train_scaled = self.scale_data(self.X_train)
        X_test_scaled = self.scale_data(self.X_test)
        
        # Convert labels to binary if needed
        if len(np.unique(self.y_train)) > 2:
            # Use label encoder to get the integer value for 'High'
            if self.label_encoder is not None:
                high_label = self.label_encoder.transform(['High'])[0]
                y_train = (self.y_train == high_label).astype(int)
                y_test = (self.y_test == high_label).astype(int)
            else:
                # fallback: use the max value as 'High'
                high_label = np.max(self.y_train)
                y_train = (self.y_train == high_label).astype(int)
                y_test = (self.y_test == high_label).astype(int)
        else:
            y_train = self.y_train
            y_test = self.y_test
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        history = model.fit(X_train_scaled, 
                          y_train,
                          epochs=20,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0)
        
        _, accuracy = model.evaluate(X_test_scaled, 
                                   y_test,
                                   verbose=0)
        
        return model, accuracy, history
    
    def run_all_models(self):
        """Run all models and return results"""
        results = {}
        
        # Decision Tree
        dt_model, dt_acc, dt_report = self.decision_tree()
        results['Decision Tree'] = {'accuracy': dt_acc, 'report': dt_report, 'model': dt_model}
        
        # Random Forest
        rf_model, rf_acc, rf_report = self.random_forest()
        results['Random Forest'] = {'accuracy': rf_acc, 'report': rf_report, 'model': rf_model}
        
        # Gradient Boosting
        gb_model, gb_acc, gb_report = self.gradient_boosting()
        results['Gradient Boosting'] = {'accuracy': gb_acc, 'report': gb_report, 'model': gb_model}
        
        # SVM
        svm_model, svm_acc, svm_report = self.svm()
        results['SVM'] = {'accuracy': svm_acc, 'report': svm_report, 'model': svm_model}
        
        # SGD
        sgd_model, sgd_acc, sgd_report = self.stochastic_gradient_descent()
        results['Stochastic Gradient Descent'] = {'accuracy': sgd_acc, 'report': sgd_report, 'model': sgd_model}
        
        # XGBoost
        xgb_model, xgb_acc, xgb_report = self.xgboost()
        results['XGBoost'] = {'accuracy': xgb_acc, 'report': xgb_report, 'model': xgb_model}
        
        # Deep Learning
        dl_model, dl_acc, dl_history = self.deep_learning()
        results['Deep Learning'] = {'accuracy': dl_acc, 'history': dl_history, 'model': dl_model}
        
        return results

    def display_warning(self):
        """Display a warning about the machine learning task"""
        st.warning(
            "Warning: The current machine learning task is not meaningful because the target variable is highly dependent on the features (State/District). "
            "For a realistic ML demo, use features that are not directly or indirectly derived from the target, or use external demographic/geographic data."
        )