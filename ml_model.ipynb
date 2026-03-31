import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.colab import files
import os

# Ensure files are present; if not, prompt for upload
required_files = ['movies.csv', 'users.csv', 'watch_history.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f'Missing files: {missing_files}. Please upload them.')
    uploaded = files.upload()

# Load datasets
df_movies = pd.read_csv('movies.csv')
df_users = pd.read_csv('users.csv')
df_watch = pd.read_csv('watch_history.csv')

# Merge datasets
df = df_watch.merge(df_users, on='user_id', how='left').merge(df_movies, on='movie_id', how='left')

# Preprocessing
df.ffill(inplace=True)

le = LabelEncoder()
categorical_cols = ['genre_primary', 'device_type', 'subscription_plan', 'location_country', 'action', 'quality']
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Create target columns
df['churn'] = np.where((df['watch_duration_minutes'] < 30) & (df['progress_percentage'] < 50), 1, 0)
df['ad_click'] = np.where((df['user_rating'] >= 4) & (df['watch_duration_minutes'] > 60), 1, 0)

# Define columns to drop (leakage + identifiers)
leakage_cols = ['watch_duration_minutes', 'progress_percentage', 'user_rating']
identifier_cols = ['session_id', 'user_id', 'movie_id', 'watch_date']
drop_cols_base = leakage_cols + identifier_cols

# Create refined feature sets
X_churn_refined = df.drop(columns=drop_cols_base + ['churn', 'ad_click'], errors='ignore')
y_churn = df['churn']

X_ads_refined = df.drop(columns=drop_cols_base + ['churn', 'ad_click'], errors='ignore')
y_ads = df['ad_click']

print(f'Refined Churn features shape: {X_churn_refined.shape}')
print(f'Refined Ads features shape: {X_ads_refined.shape}')
print('Refined feature sets created successfully.')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Helper to ensure only numeric data is passed to the model
def get_numeric_only(df):
    return df.select_dtypes(include=[np.number])

# 1. Train-Test Split for Churn
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    get_numeric_only(X_churn_refined), y_churn, test_size=0.2, random_state=42
)

# 2. Train Churn Model
churn_model_refined = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model_refined.fit(Xc_train, yc_train)
print('Refined Churn model trained.')

# 3. Train-Test Split for Ad-Click
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    get_numeric_only(X_ads_refined), y_ads, test_size=0.2, random_state=42
)

# 4. Train Ad-Click Model
ad_model_refined = RandomForestClassifier(n_estimators=100, random_state=42)
ad_model_refined.fit(Xa_train, ya_train)
print('Refined Ad-Click model trained.')
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Evaluate Churn Model
yc_pred = churn_model_refined.predict(Xc_test)
print("--- Refined Churn Model Classification Report ---")
print(classification_report(yc_test, yc_pred))

# Feature Importance for Churn
churn_importances = churn_model_refined.feature_importances_
churn_features = Xc_train.columns
churn_indices = np.argsort(churn_importances)[-10:]  # Top 10

plt.figure(figsize=(10, 5))
plt.barh(range(len(churn_indices)), churn_importances[churn_indices], color='salmon')
plt.yticks(range(len(churn_indices)), [churn_features[i] for i in churn_indices])
plt.xlabel('Relative Importance')
plt.title('Top 10 Feature Importances: Refined Churn Model')
plt.show()

# 2. Evaluate Ad-Click Model
ya_pred = ad_model_refined.predict(Xa_test)
print("\n--- Refined Ad-Click Model Classification Report ---")
print(classification_report(ya_test, ya_pred))

# Feature Importance for Ad-Click
ad_importances = ad_model_refined.feature_importances_
ad_features = Xa_train.columns
ad_indices = np.argsort(ad_importances)[-10:]  # Top 10

plt.figure(figsize=(10, 5))
plt.barh(range(len(ad_indices)), ad_importances[ad_indices], color='skyblue')
plt.yticks(range(len(ad_indices)), [ad_features[i] for i in ad_indices])
plt.xlabel('Relative Importance')
plt.title('Top 10 Feature Importances: Refined Ad-Click Model')
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# 1. Verify files exist and load them
required_files = ['movies.csv', 'users.csv', 'watch_history.csv']
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f'Required file {f} is missing.')

df_movies = pd.read_csv('movies.csv')
df_users = pd.read_csv('users.csv')
df_watch = pd.read_csv('watch_history.csv')

# 2. Merge DataFrames
# Merge watch history with users on user_id, then with movies on movie_id
df_merged = df_watch.merge(df_users, on='user_id', how='left').merge(df_movies, on='movie_id', how='left')

# 3. Handle missing values
# Using forward fill as a general strategy for this dataset
df_merged.ffill(inplace=True)
df_merged.bfill(inplace=True) # Catch any leading NaNs

# 4. Identify and Encode Categorical Columns
le = LabelEncoder()
categorical_cols = ['genre_primary', 'device_type', 'subscription_plan', 'location_country', 'action', 'quality']

for col in categorical_cols:
    if col in df_merged.columns:
        df_merged[col] = le.fit_transform(df_merged[col].astype(str))

# 5. Confirm readiness
null_count = df_merged.isnull().sum().sum()
print(f'Total null values remaining: {null_count}')
print(f'Merged DataFrame shape: {df_merged.shape}')
print('Categorical columns encoded. Preview:')
print(df_merged[categorical_cols].head())
import pandas as pd
import numpy as np

# 1. Define target variables based on logic
df_merged['churn'] = np.where((df_merged['watch_duration_minutes'] < 30) & (df_merged['progress_percentage'] < 50), 1, 0)
df_merged['ad_click'] = np.where((df_merged['user_rating'] >= 4) & (df_merged['watch_duration_minutes'] > 60), 1, 0)

# 2. Create engineered features (aggregates) to avoid direct leakage
# Calculate average rating and session count per user
user_behavior = df_merged.groupby('user_id').agg({
    'user_rating': 'mean',
    'session_id': 'count'
}).rename(columns={'user_rating': 'avg_user_rating_hist', 'session_id': 'session_count_hist'})

df_merged = df_merged.merge(user_behavior, on='user_id', how='left')

# 3. Extract time-based features
df_merged['watch_date'] = pd.to_datetime(df_merged['watch_date'])
df_merged['hour_of_day'] = df_merged['watch_date'].dt.hour
df_merged['day_of_week'] = df_merged['watch_date'].dt.dayofweek

# 4. Define columns to exclude (Leakage + Identifiers)
leakage_columns = ['watch_duration_minutes', 'progress_percentage', 'user_rating']
identifier_columns = ['session_id', 'user_id', 'movie_id', 'watch_date']
target_columns = ['churn', 'ad_click']

drop_cols = leakage_columns + identifier_columns + target_columns

# 5. Prepare final feature matrices (numeric only)
X_churn = df_merged.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
y_churn = df_merged['churn']

X_ads = df_merged.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
y_ads = df_merged['ad_click']

print(f'X_churn shape: {X_churn.shape}')
print(f'X_ads shape: {X_ads.shape}')
print('Feature engineering complete. Data leakage columns and identifiers removed.')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Train-Test Split for Churn
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42
)

# 2. Train Churn Model
churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model.fit(Xc_train, yc_train)
print('RandomForest model for Churn prediction trained successfully.')

# 3. Train-Test Split for Ad-Click
Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_ads, y_ads, test_size=0.2, random_state=42
)

# 4. Train Ad-Click Model
ad_model = RandomForestClassifier(n_estimators=100, random_state=42)
ad_model.fit(Xa_train, ya_train)
print('RandomForest model for Ad-Click prediction trained successfully.')
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. Evaluate and Visualize Churn Model
yc_pred = churn_model.predict(Xc_test)
print('--- Refined Churn Model Classification Report ---')
print(classification_report(yc_test, yc_pred))

churn_importances = churn_model.feature_importances_
churn_features = Xc_train.columns
churn_indices = np.argsort(churn_importances)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(churn_indices)), churn_importances[churn_indices], color='salmon', label='Feature Importance')
plt.yticks(range(len(churn_indices)), [churn_features[i] for i in churn_indices])
plt.xlabel('Relative Importance Value')
plt.ylabel('Feature Name')
plt.title('Top 10 Feature Importances: Refined Churn Model')
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Evaluate and Visualize Ad-Click Model
ya_pred = ad_model.predict(Xa_test)
print('\n--- Refined Ad-Click Model Classification Report ---')
print(classification_report(ya_test, ya_pred))

ad_importances = ad_model.feature_importances_
ad_features = Xa_train.columns
ad_indices = np.argsort(ad_importances)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(ad_indices)), ad_importances[ad_indices], color='skyblue', label='Feature Importance')
plt.yticks(range(len(ad_indices)), [ad_features[i] for i in ad_indices])
plt.xlabel('Relative Importance Value')
plt.ylabel('Feature Name')
plt.title('Top 10 Feature Importances: Refined Ad-Click Model')
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
!pip install -U imbalanced-learn
import imblearn
print(f'imbalanced-learn version: {imblearn.__version__}')
from imblearn.over_sampling import SMOTE

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the churn training data
Xc_train_resampled, yc_train_resampled = smote.fit_resample(Xc_train, yc_train)

# Confirm the new class distribution
print('Original class distribution in yc_train:')
print(yc_train.value_counts())

print('\nResampled class distribution in yc_train_resampled:')
print(yc_train_resampled.value_counts())

print(f'\nShape of resampled features: {Xc_train_resampled.shape}')
from sklearn.ensemble import RandomForestClassifier

# 1. Instantiate the model
churn_model_balanced = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Fit the model on balanced data
churn_model_balanced.fit(Xc_train_resampled, yc_train_resampled)

# 3. Confirmation
print('Churn model (balanced) trained successfully using resampled data.')
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Predict using the balanced model on the original test set
yc_pred_balanced = churn_model_balanced.predict(Xc_test)

# 2. Print Classification Report
print('--- Balanced Churn Model Classification Report (Original Test Set) ---')
print(classification_report(yc_test, yc_pred_balanced))

# 3. Visualize Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    yc_test,
    yc_pred_balanced,
    display_labels=['No Churn', 'Churn'],
    cmap='YlGnBu',
    ax=ax
)
plt.title('Confusion Matrix: Balanced Churn Model')
plt.grid(False)
plt.show()
