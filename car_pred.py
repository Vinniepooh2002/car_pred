# --- Step 1: Import all our libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import warnings

# Ignore any simple warnings
warnings.filterwarnings('ignore')

print("--- Script Started. Libraries imported. ---")

# --- Step 2: Load the dataset ---
try:
    df = pd.read_csv('car data.csv')
    print("--- Data loaded successfully. ---")
except FileNotFoundError:
    print("!!! ERROR: 'car data.csv' not found. !!!")
    print("Please make sure the .csv file is in the same folder as your predict.py script.")
    exit() # Stop the script if the file isn't found

# --- Step 3: Clean and Prepare the Data ---
# (Using all the correct column names we found)

# 1. Create 'car_age'
current_year = 2025
# We use a try-except block in case 'Year' isn't in the file
try:
    df['car_age'] = current_year - df['Year']
    # Drop old columns
    df.drop(columns=['Car_Name', 'Year'], inplace=True)
    print("--- 'car_age' created and old columns dropped. ---")
except KeyError:
    print("--- 'Year' or 'Car_Name' not found. Assuming data is already partly clean. ---")


# 2. Convert text to numbers
#    Using the *exact* names from your file: 'Fuel_Type', 'Selling_type', 'Transmission'
try:
    df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
    print("--- Data cleaning complete. All columns are numbers. ---")
except Exception as e:
    print(f"!!! ERROR during data cleaning: {e} !!!")
    exit()

# --- Step 4: Split the Data (Train/Test) ---
# X = all columns *except* the answer ('Selling_Price')
X = df.drop('Selling_Price', axis=1)

# y = *only* the answer column
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- Data split into training and testing sets. ---")

# --- Step 5: Train Model 1 (Linear Regression) ---
print("--- Training Model 1: Linear Regression... ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_predictions)

# --- Step 6: Train Model 2 (Random Forest) ---
print("--- Training Model 2: Random Forest... ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)

# --- Step 7: Train Model 3 (XGBoost) ---
print("--- Training Model 3: XGBoost... ---")
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\n--- MODEL COMPARISON ---")
print(f"Linear Regression R-squared: {lr_r2:.2f}")
print(f"Random Forest R-squared: {rf_r2:.2f}")
print(f"XGBoost R-squared: {xgb_r2:.2f}")

# --- Step 8: Show Feature Importance (from our best model) ---
print("\n--- Showing Feature Importance from Random Forest... ---")
# Get the "importance scores" from the model
importances = rf_model.feature_importances_

# Match the scores with their column names
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort them from most important to least important
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot them on a bar chart
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Which Features Are Most Important to the Model?")
plt.gca().invert_yaxis()  # Show most important at the top
plt.tight_layout() # Make sure labels fit

# This command will make the chart pop up in a new window
print("--- Displaying plot. Close the plot window to finish. ---")
plt.show()

print("--- Script Finished. ---")