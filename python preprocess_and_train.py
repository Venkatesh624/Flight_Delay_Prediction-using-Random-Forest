import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Inspect the dataset
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing Values:\n", df.isnull().sum())
print("Category Unique Values:\n", df['Category'].unique())
print("Category Value Counts:\n", df['Category'].value_counts())

# Step 1: Feature Engineering
# Convert time columns to datetime and extract features
time_cols = ['Scheduled Departure', 'Scheduled Arrival']
for col in time_cols:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.dayofweek
        except Exception as e:
            print(f"Error processing {col}: {e}")

# Step 2: Encode Categorical Variables
# One-hot encode nominal variables
nominal_cols = ['From', 'To', 'Airline', 'weather__hourly__weatherDesc__value']
if all(col in df.columns for col in nominal_cols):
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Label encode Status (assuming it's ordinal)
if 'Status' in df.columns:
    le_status = LabelEncoder()
    df['Status'] = le_status.fit_transform(df['Status'])
    print("Status Encoded Classes:", le_status.classes_)

# Encode target (Category)
try:
    le_category = LabelEncoder()
    df['Category'] = le_category.fit_transform(df['Category'])
    print("Category Encoded Classes:", le_category.classes_)
except Exception as e:
    print(f"Error encoding Category: {e}")
    raise

# Step 3: Define Features and Target
target = 'Category'
if target not in df.columns:
    raise ValueError("Target column 'Category' not found.")

# Drop irrelevant or redundant columns
drop_cols = ['Used Date', 'Scheduled Departure', 'Departure',
             'Scheduled Arrival', 'Arrival', 'SDEP', 'DEP', 'SARR', 'ARR']
drop_cols = [col for col in drop_cols if col in df.columns]
X = df.drop(columns=[target] + drop_cols)
y = df[target]

# Step 4: Scale Numerical Features
numerical_cols = ['Departure Delay', 'Arrival Delay', 'Distance', 'Passenger Load Factor',
                  'Airline Rating', 'Airport Rating', 'Market Share', 'OTP Index',
                  'weather__hourly__windspeedKmph', 'weather__hourly__precipMM',
                  'weather__hourly__humidity', 'weather__hourly__visibility',
                  'weather__hourly__pressure', 'weather__hourly__cloudcover']
numerical_cols = [col for col in numerical_cols if col in X.columns]
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
# Use placeholder names for classes if original labels are numerical
target_names = [str(cls) for cls in le_category.classes_]  # Fallback to stringified numbers
try:
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
except Exception as e:
    print("Error in classification_report:", e)
    print("y_test unique values:", np.unique(y_test))
    print("y_pred unique values:", np.unique(y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to 'confusion_matrix.png'")

# Cross-Validation to check for overfitting
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved to 'feature_importance.png'")

# Step 8: Save Preprocessed Data and Model
df.to_csv('Preprocessed_Dataset.csv', index=False)
joblib.dump(model, 'flight_delay_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_category, 'label_encoder_category.pkl')

print("Preprocessed data saved to 'Preprocessed_Dataset.csv'")
print("Model saved to 'flight_delay_model.pkl'")
print("Scaler saved to 'scaler.pkl'")
print("Label encoder for Category saved to 'label_encoder_category.pkl'")