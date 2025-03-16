import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
df = pd.read_csv("data/cs-training.csv")

# Erste 5 Zeilen anzeigen
print("\n🔍 Erste 5 Zeilen des Datensatzes:")
print(df.head())

# Spaltennamen & fehlende Werte prüfen
print("\n📊 Überblick über fehlende Werte:")
print(df.isnull().sum())

# Fehlende Werte mit Median ersetzen
df.fillna(df.median(), inplace=True)

# Datentypen überprüfen
print("\n📌 Datentypen:")
print(df.dtypes)

# Grundlegende Statistik anzeigen
print("\n📈 Basisstatistiken:")
print(df.describe())

# Histogramm für numerische Spalten
plt.figure(figsize=(10,6))
df.hist(bins=30, figsize=(12,8), edgecolor='black')
plt.tight_layout()
plt.show()

# Bereinigte Daten speichern
df.to_csv("data/cleaned_data.csv", index=False)
print("✅ Preprocessing abgeschlossen! Daten gespeichert unter 'data/cleaned_data.csv'")
