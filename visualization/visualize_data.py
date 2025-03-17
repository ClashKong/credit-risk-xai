import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Lade die Daten
file_path = "data/processed_data.csv"
df = pd.read_csv(file_path)

# 1️⃣ **Feature-Korrelation (Heatmap)**
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Feature Korrelation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# 2️⃣ **PCA-Reduktion (2D-Visualisierung)**
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], alpha=0.5)
plt.title("PCA Reduktion auf 2D")
plt.xlabel("Hauptkomponente 1")
plt.ylabel("Hauptkomponente 2")
plt.savefig("plots/pca_scatter.png")
plt.show()

print("📊 Visualisierungen erfolgreich erstellt und gespeichert!")
