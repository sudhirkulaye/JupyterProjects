# !pip install pymysql
# !pip install sqlalchemy
# !pip install sqlalchemy pymysql
# !pip install --upgrade sqlalchemy
#!pip show sqlalchemy

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Database connection details
host = 'localhost'
user = 'username'
password = 'password'
database = 'database'

# Model file path
model_path = 'xgboost_marketcap_model.pkl'

# Flag to decide if training should be done from scratch
train_from_scratch = False

# Create a database connection
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

# Logging function
def log(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Step 1: Load Data from Database
log("Loading data from the database...")
query = """
    SELECT * FROM daily_data_s 
    WHERE name IN (
        SELECT ticker FROM stock_universe a 
        WHERE is_nse500 = 1 OR is_bse500 = 1
    )
    ORDER BY name, date DESC;
"""
df = pd.read_sql(query, engine)
log("Data loaded successfully.")

# Step 2: Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Step 3: Create the target variable (1-year ahead market cap)
log("Creating 1-year ahead market cap target...")
def get_one_year_ahead_market_cap(df):
    # Ensure data is sorted by name and date
    df = df.sort_values(by=['name', 'date'])
    
    # Create a new column for the target date (exactly 1 year ahead)
    df['target_date'] = df['date'] + pd.DateOffset(years=1)
    
    # Self-join to match each row with its corresponding target date
    df_with_target = df.merge(
        df[['name', 'date', 'market_cap']],
        left_on=['name', 'target_date'],
        right_on=['name', 'date'],
        suffixes=('', '_target'),
        how='left'
    )
    
    # Rename the merged market cap column to 'target_market_cap'
    df_with_target.rename(columns={'market_cap_target': 'target_market_cap'}, inplace=True)
    
    # Drop the duplicate 'date' column from the merged result
    df_with_target.drop(columns=['date_target'], inplace=True)
    
    # Drop rows where 'target_market_cap' is missing
    df_with_target = df_with_target.dropna(subset=['target_market_cap'])
    
    return df_with_target

df = get_one_year_ahead_market_cap(df)
log("Target variable created.")

# Verify for TCS on 2023-01-03
# log("Verifying target market cap for TCS on 2023-01-03...")
# verification_row = df[(df['name'] == 'TCS') & (df['date'] == '2023-01-03')]

# if not verification_row.empty:
#     market_cap = verification_row['market_cap'].values[0]
#     target_market_cap = verification_row['target_market_cap'].values[0]
#     print(f"Name: TCS")
#     print(f"Date: 2023-01-03")
#     print(f"Market Cap: {market_cap}")
#     print(f"Target Market Cap (1 year ahead): {target_market_cap}")
# else:
#     log("No data found for TCS on 2023-01-03.")

# Step 4: Feature Engineering (excluding non-numeric and target columns)
log("Preparing features and target...")
features = df.drop(columns=['name', 'rank', 'last_result_date','market_cap', 'sector', 'industry', 'sub_industry', 'target_date'])
features_sorted = features.sort_values(by='date')
target_sorted = features_sorted['target_market_cap']
features_sorted = features_sorted.drop(columns=['target_market_cap', 'date'])

# Step 5: Train-test split while preserving time order
log("Splitting data into train and test sets while preserving time order...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_sorted, target_sorted, test_size=0.3, shuffle=False)
    
log("Data split completed.")

# Check if the model already exists and the flag for training
if os.path.exists(model_path) and not train_from_scratch:
    log("Loading existing model...")
    best_model = joblib.load(model_path)
    log("Model loaded successfully.")
else:
    # Step 6: Define XGBoost model and GridSearchCV parameters
    log("Training the model with GridSearchCV...")
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5],
    #     'learning_rate': [0.01, 0.1],
    #     'subsample': [0.8],
    #     'colsample_bytree': [0.8]
    # }
    # param_grid = {
    #     'n_estimators': [100, 200, 300, 500],
    #     'max_depth': [3, 5, 7, 9],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'subsample': [0.6, 0.8, 1.0],
    #     'colsample_bytree': [0.6, 0.8, 1.0]
    # }
    # Best Tree
    param_grid = {
        'n_estimators': [500],
        'max_depth': [7],
        'learning_rate': [0.05],
        'subsample': [1],
        'colsample_bytree': [0.6]
    }

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)

    # Measure training time
    start_time = datetime.now()
    grid_search.fit(X_train, y_train)
    end_time = datetime.now()
    log(f"Model training completed in {end_time - start_time}.")

    # Best parameters from GridSearchCV
    log(f"Best Parameters: {grid_search.best_params_}")

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_path)
    log("Best model saved for future use.")

# Step 7: Evaluate the model on the test set
log("Evaluating the model on the test set...")
start_time = datetime.now()
y_pred = best_model.predict(X_test)
end_time = datetime.now()
log(f"Model testing completed in {end_time - start_time}.")

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

log(f"Test RMSE: {rmse:.2f}")
log(f"Test R²: {r2:.2f}")

# Optional: Feature Importance
importances = best_model.feature_importances_
feature_names = X_train.columns  # Ensure feature_names match the training data columns

# Ensure lengths match
if len(importances) != len(feature_names):
    log("Mismatch between feature importances and feature names. Adjusting...")
    min_length = min(len(importances), len(feature_names))
    importances = importances[:min_length]
    feature_names = feature_names[:min_length]

# Create DataFrame and display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

log("Feature Importances:")
print(feature_importance_df)

def get_latest_data_and_predict(ticker, date, engine, model):
    log(f"Fetching data for ticker: {ticker} on date: {date}")

    # SQL query to fetch the data for the specified ticker and date
    query = f"""
        SELECT * FROM daily_data_s
        WHERE name = '{ticker}'
        AND date = '{date}';
    """

    # Fetch the data from the database
    latest_data = pd.read_sql(query, engine)

    if latest_data.empty:
        log(f"No data found for ticker: {ticker} on date: {date}")
        return

    # Convert date column to datetime
    latest_data['date'] = pd.to_datetime(latest_data['date'])

    # Extract relevant fields
    latest_row = latest_data.iloc[0]
    latest_date = latest_row['date']
    current_market_cap = latest_row['market_cap']

    log(f"Latest available date for {ticker}: {latest_date.strftime('%Y-%m-%d')}")

    # Prepare the test data by dropping unnecessary columns
    test_data = latest_row.drop(labels=['date', 'name', 'rank', 'last_result_date', 'market_cap', 'sector', 'industry', 'sub_industry'])

    # Convert to DataFrame with a single row
    test_data_df = pd.DataFrame([test_data])

    # Loading existing model...
    log("Loading existing model...")
    best_model = joblib.load(model_path)
    log("Model loaded successfully.")

    # Predict the target market cap
    predicted_market_cap = model.predict(test_data_df)[0]

    # Calculate expected percentage change
    expected_change = ((predicted_market_cap / current_market_cap) - 1) * 100

    # Print the results
    print(f"\nPrediction Results for {ticker}")
    print(f"Latest Available Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Current Market Cap: {current_market_cap:,.2f}")
    print(f"Predicted Target Market Cap (1 year ahead): {predicted_market_cap:,.2f}")
    print(f"Expected % Change Over the Year: {expected_change:.2f}%")

# Example call for TARSONS with a specific date
ticker = 'TCS'
date = '2025-01-20'  # Replace with the desired date in 'YYYY-MM-DD' format
get_latest_data_and_predict(ticker, date, engine, best_model)

