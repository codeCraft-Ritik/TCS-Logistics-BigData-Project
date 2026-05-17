"""
TCS Internship Project - Machine Learning Models
================================================
Predictive models for logistics optimization

Author: Data Analytics Team
Date: April 2024
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
import warnings

# Set UTF-8 encoding for Windows systems
os.environ['PYTHONIOENCODING'] = 'utf-8'

warnings.filterwarnings('ignore')


class LogisticsMLModels:
    """Machine Learning Models for Logistics Prediction"""
    
    def __init__(self, data_processor):
        """Initialize with processed data"""
        self.processor = data_processor
        self.models = {}
        self.scalers = {}
        self.results = {}
    
    def prepare_fuel_consumption_data(self):
        """Prepare data for fuel consumption prediction"""
        print("\nPreparing fuel consumption prediction data...")
        
        trip_data = self.processor.processed_data['trip_analysis'].copy()
        
        # Select relevant features
        features = ['actual_distance_miles', 'actual_duration_hours', 'idle_time_hours']
        X = trip_data[features].copy()
        y = trip_data['fuel_gallons_used'].copy()
        
        # Remove outliers
        X = X[(X > 0).all(axis=1)]
        y = y[X.index]
        
        print(f"  - Training samples: {len(X)}")
        return X, y
    
    def predict_fuel_consumption(self):
        """Predict fuel consumption for trips"""
        print("\n" + "="*60)
        print("MODEL 1: FUEL CONSUMPTION PREDICTION")
        print("="*60)
        
        X, y = self.prepare_fuel_consumption_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['fuel_consumption'] = scaler
        
        # Train models
        print("\nTraining Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        gb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = gb_model.predict(X_test_scaled)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Model Performance:")
        print(f"  - RMSE: {rmse:.2f} gallons")
        print(f"  - R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        self.models['fuel_consumption'] = gb_model
        self.results['fuel_consumption'] = {
            'rmse': rmse, 'r2': r2, 'feature_importance': feature_importance,
            'test_predictions': y_pred, 'actual': y_test
        }
        
        return gb_model
    
    def prepare_delivery_delay_data(self):
        """Prepare data for delivery delay prediction"""
        print("\nPreparing delivery delay prediction data...")
        
        trip_data = self.processor.processed_data['trip_analysis'].copy()
        delivery_perf = self.processor.processed_data.get('delivery_performance')
        
        if delivery_perf is None:
            print("⚠ Delivery performance data not available")
            return None, None
        
        # Merge delay information
        delay_data = trip_data.merge(delivery_perf[['trip_id', 'on_time_flag']], on='trip_id', how='left')
        
        # Select features
        features = ['actual_distance_miles', 'actual_duration_hours', 'idle_time_hours', 
                   'average_mpg', 'fuel_gallons_used']
        X = delay_data[features].dropna()
        y = delay_data.loc[X.index, 'on_time_flag']
        
        # Convert continuous on_time_flag to binary (on-time if >= 0.5)
        y = (y >= 0.5).astype(int)
        
        print(f"  - Training samples: {len(X)}")
        return X, y
    
    def predict_delivery_delays(self):
        """Predict on-time delivery probability"""
        print("\n" + "="*60)
        print("MODEL 2: DELIVERY DELAY PREDICTION")
        print("="*60)
        
        result = self.prepare_delivery_delay_data()
        if result[0] is None:
            return None
        
        X, y = result
        
        if len(X) < 10:
            print("⚠ Insufficient data for model training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['delivery_delay'] = scaler
        
        # Train classifier
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n✓ Model Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        self.models['delivery_delay'] = rf_model
        self.results['delivery_delay'] = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'feature_importance': feature_importance, 'test_predictions': y_pred
        }
        
        return rf_model
    
    def prepare_route_profitability_data(self):
        """Prepare data for route profitability prediction"""
        print("\nPreparing route profitability prediction data...")
        
        route_metrics = self.processor.processed_data.get('route_metrics')
        
        if route_metrics is None:
            print("⚠ Route metrics not available")
            return None, None
        
        # Select features
        features = ['total_distance', 'trips_count', 'avg_mpg', 'cost_per_mile']
        X = route_metrics[features].dropna()
        y = route_metrics.loc[X.index, 'profit_margin']
        
        print(f"  - Training samples: {len(X)}")
        return X, y
    
    def predict_route_profitability(self):
        """Predict route profit margin"""
        print("\n" + "="*60)
        print("MODEL 3: ROUTE PROFITABILITY PREDICTION")
        print("="*60)
        
        result = self.prepare_route_profitability_data()
        if result[0] is None:
            return None
        
        X, y = result
        
        if len(X) < 10:
            print("⚠ Insufficient data for model training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['route_profitability'] = scaler
        
        # Train model
        print("\nTraining Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        gb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = gb_model.predict(X_test_scaled)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Model Performance:")
        print(f"  - RMSE: {rmse:.2f}%")
        print(f"  - R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        self.models['route_profitability'] = gb_model
        self.results['route_profitability'] = {
            'rmse': rmse, 'r2': r2, 'feature_importance': feature_importance,
            'test_predictions': y_pred, 'actual': y_test
        }
        
        return gb_model
    
    def prepare_driver_safety_data(self):
        """Prepare data for driver safety risk prediction"""
        print("\nPreparing driver safety risk prediction data...")
        
        driver_metrics = self.processor.processed_data.get('driver_metrics')
        safety_summary = self.processor.processed_data.get('safety_summary')
        
        if driver_metrics is None or safety_summary is None:
            print("⚠ Required data not available")
            return None, None
        
        # Merge safety data with driver metrics
        safety_data = driver_metrics.merge(safety_summary, on='driver_id', how='left', suffixes=('_driver', '_safety'))
        safety_data['total_incidents'] = safety_data['total_incidents'].fillna(0)
        safety_data['has_incidents'] = (safety_data['total_incidents'] > 0).astype(int)
        
        # Select features
        features = ['avg_mpg', 'idle_percentage', 'miles_per_hour', 'revenue_per_mile']
        X = safety_data[features].dropna()
        y = safety_data.loc[X.index, 'has_incidents']
        
        print(f"  - Training samples: {len(X)}")
        return X, y
    
    def predict_driver_safety_risk(self):
        """Predict driver safety risk"""
        print("\n" + "="*60)
        print("MODEL 4: DRIVER SAFETY RISK PREDICTION")
        print("="*60)
        
        result = self.prepare_driver_safety_data()
        if result[0] is None:
            return None
        
        X, y = result
        
        if len(X) < 10 or len(y.unique()) < 2:
            print("⚠ Insufficient data for model training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['driver_safety'] = scaler
        
        # Train classifier
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n✓ Model Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        self.models['driver_safety'] = rf_model
        self.results['driver_safety'] = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'feature_importance': feature_importance, 'test_predictions': y_pred
        }
        
        return rf_model
    
    def train_all_models(self):
        """Train all predictive models"""
        print("\n" + "="*80)
        print("STARTING MACHINE LEARNING MODEL TRAINING".center(80))
        print("="*80)
        
        self.predict_fuel_consumption()
        self.predict_delivery_delays()
        self.predict_route_profitability()
        self.predict_driver_safety_risk()
        
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETED".center(80))
        print("="*80)
        
        return self.models
    
    def generate_predictions(self, trip_data):
        """Generate predictions for new trip data"""
        predictions = {}
        
        if 'fuel_consumption' in self.models:
            features = trip_data[['actual_distance_miles', 'actual_duration_hours', 'idle_time_hours']]
            scaler = self.scalers['fuel_consumption']
            features_scaled = scaler.transform(features)
            predictions['fuel_consumption'] = self.models['fuel_consumption'].predict(features_scaled)
        
        return predictions
    
    def export_model_results(self, output_path='./'):
        """Export model results to CSV"""
        final_output_path = output_path + 'Output/'
        print(f"\nExporting model results to {final_output_path}...")
        
        for model_name, results in self.results.items():
            if 'feature_importance' in results:
                filename = f"{final_output_path}{model_name}_feature_importance.csv"
                results['feature_importance'].to_csv(filename, index=False)
                print(f"✓ Exported {filename}")


if __name__ == "__main__":
    # This would be used with the data processor
    print("This module is imported by analysis notebooks")
