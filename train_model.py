import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read data with proper index column handling
df = pd.read_csv("german_credit_data.csv", index_col=0)

# Map Risk to 0/1
df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})

# Fix missing values
for col in ['Saving accounts', 'Checking account']:
    df[col] = df[col].replace("NA", pd.NA)
    df[col] = df[col].fillna(df[col].mode()[0])

# Define categorical columns
cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
df[cat_cols] = df[cat_cols].astype("category")

# Prepare features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get categorical indices for verification
cat_indices = [X.columns.get_loc(col) for col in cat_cols]
print("🐞 Categorical feature indices:", cat_indices)  # Should be [1, 3, 4, 5, 8]

# Train model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    verbose=0,
    cat_features=cat_indices  # Use indices instead of names for safety
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Model trained with accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "catboost_model.pkl")
print("💾 Model saved successfully!")

joblib.dump(model, "train_model.pkl")