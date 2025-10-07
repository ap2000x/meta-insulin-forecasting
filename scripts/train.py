import pandas as pd
from utils.data_loader import load_azt1d_data, get_X_y
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def main():
    # Load and preprocess data
    print("Loading AZT1D dataset...")
    df = load_azt1d_data(data_dir="data/AZT1D_raw")
    X, y = get_X_y(df)

    # Split data (subject-agnostic for now)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train simple Ridge regression model
    print("Training Ridge regression model...")
    model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Evaluation ---")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R^2: {r2_score(y_test, y_pred):.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/baseline_ridge.joblib")
    print("Model saved to models/baseline_ridge.joblib")


if __name__ == "__main__":
    main()
