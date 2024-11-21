import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

def transform_data():
    # Read from raw_data directory
    raw_data_path = 'raw_data'
    ngl_df = pd.read_csv(os.path.join(raw_data_path, 'NGL-2024_data.csv'))
    nit_price_df = pd.read_csv(os.path.join(raw_data_path, 'NIT-Price-2024_data.csv'))
    climate_df = pd.read_csv(os.path.join(raw_data_path, 'Climate-Calgary-2024_data.csv'))
    
    # Get Extrapolated columns from NGL data
    extrapolated_cols = [col for col in ngl_df.columns if 'Extrapolated' in col]
    ngl_filtered = ngl_df[['Date'] + extrapolated_cols]
    
    # Cleanup column names for NGL data
    ngl_filtered.columns = [col.replace('-Extrapolated', '').replace('*', '') for col in ngl_filtered.columns]
    
    # Get NIT price data
    nit_price = nit_price_df[['Date', '[Western Canada,GASPRICE,NONE,SPOT]']]
    nit_price.columns = ['Date', 'NIT_Price']
    
    # Clean up climate data - drop coordinates and keep relevant columns
    climate_columns_to_keep = ['LOCAL_DATE', 'MAX_TEMPERATURE', 'MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 
                             'MIN_REL_HUMIDITY', 'MAX_REL_HUMIDITY', 'TOTAL_PRECIPITATION',
                             'TOTAL_RAIN', 'TOTAL_SNOW', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS']
    climate_filtered = climate_df[climate_columns_to_keep]
    climate_filtered = climate_filtered.rename(columns={'LOCAL_DATE': 'Date'})
    
    # Merge all datasets on Date
    merged_df = pd.merge(ngl_filtered, nit_price, on='Date', how='inner')
    merged_df = pd.merge(merged_df, climate_filtered, on='Date', how='inner')
    
    # Convert Date to datetime
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # Sort by date
    merged_df = merged_df.sort_values('Date')
    
    # Create transform_data directory if it doesn't exist
    os.makedirs('transform_data', exist_ok=True)
    
    # Save to transform_data directory
    merged_df.to_csv('transform_data/merged_data.csv', index=False)
    
    return merged_df

def create_ml_model(df):
    # Drop non-numeric columns and Date
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Prepare features (X) and target (y)
    X = numeric_df.drop('NIT_Price', axis=1)
    y = numeric_df['NIT_Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, feature_importance

if __name__ == "__main__":
    df = transform_data()
    model, scaler, importance = create_ml_model(df)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))