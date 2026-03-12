import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/student_scores_dataset.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "student_score_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# -------------------------------
# Features and target
# -------------------------------
X = df.drop("final_score", axis=1)
y = df["final_score"]

# -------------------------------
# Train test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# ML Pipeline
# -------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# -------------------------------
# Train model
# -------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print("Model MAE:", mae)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(pipeline, MODEL_PATH)


print("Model saved to:", MODEL_PATH)

# -------------------------------
# Example JSON prediction test
# -------------------------------
sample_json = {
    "instances": [
        [5, 85, 70, 7]
    ]
}

sample_df = pd.DataFrame(
    sample_json["instances"],
    columns=X.columns
)

prediction = pipeline.predict(sample_df)

print("Sample Prediction:", prediction)