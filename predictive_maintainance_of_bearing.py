import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==============================================================================
# STEP 1: FEATURE EXTRACTION (RAW DATA TO DATAFRAME)
# ==============================================================================
datasets = [
    {
        "path": r"C:\Users\Atharva\Desktop\maching learning\second project\dataset\2nd_test\2nd_test", 
        "failed_bearing_col": 0, 
        "test_name": "Test2_B1"
    },
    {
        "path": r"C:\Users\Atharva\Desktop\maching learning\second project\dataset\3rd_test\4th_test\txt", 
        "failed_bearing_col": 2, 
        "test_name": "Test3_B3"
    }
]

combined_feature_list = []

print("--- STEP 1: Starting Feature Extraction ---")
for dataset in datasets:
    folder_path = dataset["path"]
    bearing_col = dataset["failed_bearing_col"]
    tag = dataset["test_name"]
    
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}. Skipping...")
        continue
        
    file_names = sorted(os.listdir(folder_path))
    print(f"Processing {len(file_names)} files from {tag}...")
    
    for index, filename in enumerate(file_names):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        try:
            df_raw = pd.read_csv(file_path, sep='\t', header=None)
            signal = df_raw[bearing_col].values
            
            # Extract statistical metrics
            rms = np.sqrt(np.mean(signal**2))
            kurt = kurtosis(signal)
            peak_to_peak = np.max(signal) - np.min(signal)
            skewness = skew(signal)
            std_dev = np.std(signal)
            
            combined_feature_list.append({
                'Test_ID': tag,
                'Operating_Hours': index * (10 / 60),  # 10 minute logging intervals
                'RMS': rms,
                'Kurtosis': kurt,
                'Peak_to_Peak': peak_to_peak,
                'Skewness': skewness,
                'Std_Dev': std_dev
            })
            
        except Exception as e:
            pass

# Convert list to a structured DataFrame
df_features = pd.DataFrame(combined_feature_list)
print(f"Feature extraction finished. Total rows generated: {len(df_features)}")

# ==============================================================================
# STEP 2: DATA LABELING (CREATING TARGET LABELS)
# ==============================================================================
print("\n--- STEP 2: Labeling Health Status ---")

def label_health_status(group):
    max_hours = group['Operating_Hours'].max()
    hours_to_failure = max_hours - group['Operating_Hours']
    
    # 2 = Critical (last 5 hours) | 1 = Warning (5-25 hours left) | 0 = Normal (>25 hours left)
    conditions = [
        (hours_to_failure <= 5),
        (hours_to_failure > 5) & (hours_to_failure <= 25),
        (hours_to_failure > 25)
    ]
    choices = [2, 1, 0]
    
    group['Health_Status'] = np.select(conditions, choices, default=0)
    return group

# Group by Test_ID to label each dataset lifecycle independently
df_features = df_features.groupby('Test_ID', group_keys=False).apply(label_health_status)

print("Label Distribution:")
print(df_features['Health_Status'].value_counts())

# ==============================================================================
# STEP 3: MODEL TRAINING & EVALUATION
# ==============================================================================
print("\n--- STEP 3: Training Machine Learning Model ---")

features = ['RMS', 'Kurtosis', 'Peak_to_Peak', 'Skewness', 'Std_Dev']
X = df_features[features]
y = df_features['Health_Status']

# Stratified split to keep clean proportions of labels across train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Performance summary
y_pred = model.predict(X_test)
print(f"Model Training Complete. Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Warning (1)', 'Critical (2)']))

# ==============================================================================
# STEP 4: MODEL EXPORT
# ==============================================================================
model_filename = 'bearing_trained_model.pkl'
joblib.dump(model, model_filename)
print(f"\n--- SUCCESS: Core pipeline complete! Brain saved as '{model_filename}' ---")