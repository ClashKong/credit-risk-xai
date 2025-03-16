import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier   
from sklearn.metrics import accuracy_score, f1_score
import joblib

df = pd.read_csv("data/processed_data.csv")

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}
for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": float(accuracy), "F1-Score": float(f1)}
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
    print(f"{name} - Acucuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

pd.DataFrame(results).to_csv("models/model_results.csv")
print("\n Model Training abgeschlossen! Ergebnisse gespeichert unter 'models/model_results.csv'")
