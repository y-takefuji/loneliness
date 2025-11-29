import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_excel('DataBase.xlsx')

# Drop unnecessary columns
df = df.drop(['SWLS_Interpretation', 'UCLA_M', 'AIS_Interpretation', 'ID'], axis=1)

# Handle the 'Age' variable that contains ranges like '21-23' or 'Above 26'
def extract_age(age_str):
    age_str = str(age_str)
    if age_str.startswith('Above'):
        # Extract the number from 'Above X'
        return float(age_str.split(' ')[1])
    elif '-' in age_str:
        # Take the first number in ranges like '21-23'
        return float(age_str.split('-')[0])
    else:
        # For single values
        return float(age_str)

if 'Age' in df.columns:
    df['Age'] = df['Age'].apply(extract_age)

# Define target variable
y = df['UCLA_100']
X = df.drop(['UCLA_100'], axis=1)

# Function to get top N features using Random Forest
def get_top_features_rf(X, y, n=5):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:n]
    return X.columns[indices].tolist()

# Function to get top N features using XGBoost
def get_top_features_xgb(X, y, n=5):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:n]
    return X.columns[indices].tolist()

# Function to get top N features using Highly Variable Gene Selection (variance-based)
def get_top_features_hvgs(X, y, n=5):
    # Calculate feature variances
    variances = X.var().sort_values(ascending=False)
    return variances.index.tolist()[:n]

# Function to get top N features using Spearman correlation
def get_top_features_spearman(X, y, n=5):
    corr = pd.DataFrame()
    for col in X.columns:
        corr.loc[col, 'correlation'] = abs(X[col].corr(y, method='spearman'))
    corr = corr.sort_values('correlation', ascending=False)
    return corr.index.tolist()[:n]

# Function to perform cross-validation
def perform_cv(X, y, features, method):
    if method in ['Random Forest', 'HVGS', 'Spearman']:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif method == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    
    X_selected = X[features]
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
    return np.mean(scores)

# Dictionary to store results
results = {
    'Method': [],
    'CV5 Accuracy': [],
    'CV4 Accuracy': [],
    'Top5 Feature Rankings': [],
    'Top4 Feature Rankings': [],
    'Removed Feature': []  # Track which feature was removed
}

# Store the actual feature lists for verification later
top4_features_lists = {}

# 1. Random Forest
print("\nRandom Forest:")
rf_top5 = get_top_features_rf(X, y, n=5)
print(f"Top 5 features: {rf_top5}")
removed_feature = rf_top5[0]
print(f"Removing feature: {removed_feature}")
X_reduced = X.drop(removed_feature, axis=1)
rf_top4 = get_top_features_rf(X_reduced, y, n=4)
print(f"Top 4 features after removing {removed_feature}: {rf_top4}")
top4_features_lists['Random Forest'] = rf_top4  # Save for verification
rf_cv5 = perform_cv(X, y, rf_top5, 'Random Forest')
rf_cv4 = perform_cv(X, y, rf_top4, 'Random Forest')

results['Method'].append('Random Forest')
results['CV5 Accuracy'].append(rf_cv5)
results['CV4 Accuracy'].append(rf_cv4)
results['Top5 Feature Rankings'].append(', '.join(rf_top5))
results['Top4 Feature Rankings'].append(', '.join(rf_top4))
results['Removed Feature'].append(removed_feature)

# 2. XGBoost
print("\nXGBoost:")
xgb_top5 = get_top_features_xgb(X, y, n=5)
print(f"Top 5 features: {xgb_top5}")
removed_feature = xgb_top5[0]
print(f"Removing feature: {removed_feature}")
X_reduced = X.drop(removed_feature, axis=1)
xgb_top4 = get_top_features_xgb(X_reduced, y, n=4)
print(f"Top 4 features after removing {removed_feature}: {xgb_top4}")
top4_features_lists['XGBoost'] = xgb_top4  # Save for verification
xgb_cv5 = perform_cv(X, y, xgb_top5, 'XGBoost')
xgb_cv4 = perform_cv(X, y, xgb_top4, 'XGBoost')

results['Method'].append('XGBoost')
results['CV5 Accuracy'].append(xgb_cv5)
results['CV4 Accuracy'].append(xgb_cv4)
results['Top5 Feature Rankings'].append(', '.join(xgb_top5))
results['Top4 Feature Rankings'].append(', '.join(xgb_top4))
results['Removed Feature'].append(removed_feature)

# 3. Highly Variable Gene Selection
print("\nHVGS:")
hvgs_top5 = get_top_features_hvgs(X, y, n=5)
print(f"Top 5 features: {hvgs_top5}")
removed_feature = hvgs_top5[0]
print(f"Removing feature: {removed_feature}")
X_reduced = X.drop(removed_feature, axis=1)
hvgs_top4 = get_top_features_hvgs(X_reduced, y, n=4)
print(f"Top 4 features after removing {removed_feature}: {hvgs_top4}")
top4_features_lists['HVGS'] = hvgs_top4  # Save for verification
hvgs_cv5 = perform_cv(X, y, hvgs_top5, 'HVGS')
hvgs_cv4 = perform_cv(X, y, hvgs_top4, 'HVGS')

results['Method'].append('HVGS')
results['CV5 Accuracy'].append(hvgs_cv5)
results['CV4 Accuracy'].append(hvgs_cv4)
results['Top5 Feature Rankings'].append(', '.join(hvgs_top5))
results['Top4 Feature Rankings'].append(', '.join(hvgs_top4))
results['Removed Feature'].append(removed_feature)

# 4. Spearman Correlation
print("\nSpearman Correlation:")
spearman_top5 = get_top_features_spearman(X, y, n=5)
print(f"Top 5 features: {spearman_top5}")
removed_feature = spearman_top5[0]
print(f"Removing feature: {removed_feature}")
X_reduced = X.drop(removed_feature, axis=1)
spearman_top4 = get_top_features_spearman(X_reduced, y, n=4)
print(f"Top 4 features after removing {removed_feature}: {spearman_top4}")
top4_features_lists['Spearman Correlation'] = spearman_top4  # Save for verification
spearman_cv5 = perform_cv(X, y, spearman_top5, 'Spearman')
spearman_cv4 = perform_cv(X, y, spearman_top4, 'Spearman')

results['Method'].append('Spearman Correlation')
results['CV5 Accuracy'].append(spearman_cv5)
results['CV4 Accuracy'].append(spearman_cv4)
results['Top5 Feature Rankings'].append(', '.join(spearman_top5))
results['Top4 Feature Rankings'].append(', '.join(spearman_top4))
results['Removed Feature'].append(removed_feature)

# Format accuracy to 4 decimal places
for key in ['CV5 Accuracy', 'CV4 Accuracy']:
    results[key] = [f"{x:.4f}" for x in results[key]]

# Create result dataframe
result_df = pd.DataFrame(results)

# Verify that the Top4 Feature Rankings column has 4 features for each method
print("\nVerifying Top4 Feature Rankings:")
for method, features in top4_features_lists.items():
    print(f"{method}: {len(features)} features - {features}")
    # Double check the corresponding entry in the result_df
    idx = result_df[result_df['Method'] == method].index[0]
    features_in_df = result_df.loc[idx, 'Top4 Feature Rankings'].split(', ')
    print(f"  In results df: {len(features_in_df)} features - {features_in_df}")
    # If there's a mismatch, fix it
    if len(features) != len(features_in_df) or set(features) != set(features_in_df):
        print(f"  Mismatch detected! Fixing...")
        result_df.loc[idx, 'Top4 Feature Rankings'] = ', '.join(features)

# Save to CSV
result_df.to_csv('result.csv', index=False)

print("\nAnalysis completed and results saved to result.csv")
