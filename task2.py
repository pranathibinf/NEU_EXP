import numpy as np
import pandas as pd
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import statsmodels.api as sm


def download_dataset(url):
    response = requests.get(url)
    data = pd.read_csv(BytesIO(response.content), compression='gzip')
    return data

# Scaling functions
def x_scale(x, p=7.5):
    return 1/p * np.log(1 + x * (np.exp(p) - 1))

def y_scale(y):
    return np.log(1 + y) if y >= 0 else -np.log(1 - y)

# Download dataset
url = "http://129.10.224.71/~apaul/data/tests/dataset.csv.gz"
dataset = download_dataset(url)

# Apply scaling
dataset['x1'] = dataset['x1'].apply(x_scale)
dataset['y1'] = dataset['y1'].apply(y_scale)

# Define X and y
X = dataset[['x1', 'x2', 'x3', 'x4']]
y = dataset['y1']

# Add a constant to the features for the intercept
X_with_const = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X_with_const).fit()

# Print out the summary which includes p-values among other statistics
print(model.summary())

# Splitting dataset: 70% training (for complex model improvement), 15% validation, 15% test for balanced evaluation.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Hyper parameters chosen depending on how well they are contributing to the better performance of the model.
params = {
    'max_depth': 9,  # Chosen for model complexity and to avoid overfitting, optimal for dataset structure.
    'eta': 0.1,  # Learning rate set to ensure steady convergence.
    'min_child_weight': 10,  # Helps prevent overfitting by controlling the decision-making process.
    'gamma': 0,  # Minimal loss reduction required for further partition, set to 0 for baseline model complexity.
    'colsample_bytree': 1.0,  # Use all features for each tree to maximize model learning potential.
    'objective': 'reg:squarederror',  # Objective for regression tasks.
    'eval_metric': 'rmse'  # Evaluation metric to focus on minimizing prediction error.
}
num_round = 100 # Number of boosting rounds

evallist = [(dval, 'eval'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

# Evaluating the model with test data
dtest = xgb.DMatrix(X_test, label=y_test)
y_pred = bst.predict(dtest)

# Calculate RMSE, MAE, and R-squared
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")
print(f"Test R-squared: {test_r2}")
