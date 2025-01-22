# filename: stock_randomforect_ml.py
# Description: This script extracts stock data from Yahoo Finance, trains a Random Forest model to predict the next day's stock movement, and saves the data to a SQLite database.
# Dependencies: yfinance, pandas, sqlalchemy, sklearn
# Usage: python stock_randomforect_ml.py
# For running against multiple stocks, the Accuracy was 0.48, Precision was 0.48, and Recall was 0.56, which is not very good. The model is not very accurate in predicting the stock movement.
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


def extract_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def transform_data(stock_data):
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
    stock_data.dropna(inplace=True)
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Target']]

def load_data(data, table_name, engine):
    # Rename columns before saving to the database
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data.to_sql(table_name, engine, if_exists='replace', index=True)



# Function to make predictions for the next day
def predict_next_day(model, last_data):
    prediction = model.predict(last_data.reshape(1, -1))
    return "Up" if prediction[0] == 1 else "Down"


# ETL Pipeline
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

# Extract
raw_data = extract_stock_data(ticker, start_date, end_date)

# Transform
transformed_data = transform_data(raw_data)

# Load
engine = create_engine('sqlite:///stock_data.db')
load_data(transformed_data, 'stock_data', engine)

# Load data from the database
data = pd.read_sql('stock_data', engine)
# Rename columns if they are still in tuple format
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

print(data.columns)
print(data.head())

# Prepare features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
X = data[features]
y = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Get the last available data point
last_data = X.iloc[-1].values

# Predict the next day's movement
next_day_prediction = predict_next_day(model, last_data)
print(f"Prediction for next day: {next_day_prediction}")
