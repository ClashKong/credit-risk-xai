import pandas as pd     
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/cleaned_data.csv")
print(df.columns)

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

df["Debt_to_Income_Ratio"] = df["DebtRatio"] / (df["MonthlyIncome"] + 1)

scaler = StandardScaler()
numeric_cols = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "age", "MonthlyIncome"]
df[numeric_cols] = df[numeric_cols].astype(float)
df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) 

df.to_csv("data/processed_data.csv", index=False)
print(" Feature Engineering abgeschlossen! DAten gespeichert unter 'data/processed_data.csv'")
