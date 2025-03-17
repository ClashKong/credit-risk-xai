import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Daten laden
df = pd.read_csv("data/processed_data.csv")
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ **Random Forest Hyperparameter-Tuning**
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 2️⃣ **XGBoost Hyperparameter-Tuning**
xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, 10]
}
xgb = XGBClassifier(eval_metric='logloss')
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

# Ergebnisse speichern
results = {
    "Random Forest": {"Best Params": rf_grid.best_params_, "F1-Score": f1_score(y_test, best_rf.predict(X_test))},
    "XGBoost": {"Best Params": xgb_grid.best_params_, "F1-Score": f1_score(y_test, best_xgb.predict(X_test))}
}
pd.DataFrame(results).to_csv("models/hyperparameter_results.csv")

print("✅ Hyperparameter-Tuning abgeschlossen! Ergebnisse gespeichert in 'models/hyperparameter_results.csv'")
