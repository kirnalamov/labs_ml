from sklearn.datasets import load_svmlight_file
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


file_path = "dataset/wine.data"



columns = [
    "Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315", "Proline"
]

# Загружаем датасет
df = pd.read_csv(file_path, header=None, names=columns)


X = df.iloc[:, 1:] 
y = df.iloc[:, 0]   



# Нормализация (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)



# Применяем PCA без ограничения на число компонент
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Получаем собственные векторы (главные компоненты)
eigenvectors = pca.components_

# Получаем собственные значения (доля объясненной дисперсии)
eigenvalues = pca.explained_variance_
eigen_df = pd.DataFrame(
    data=eigenvectors,
    columns=X.columns,
    index=[f"PC{i + 1}" for i in range(len(eigenvalues))]
)
eigen_df["Eigenvalue"] = eigenvalues  # Добавляем столбец с собственными значениями

print("Таблица собственных значений и векторов:")
print(eigen_df)

# Применяем PCA с 3 главными компонентами
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Визуализация в 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', edgecolors='k')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA projection of Wine dataset")
plt.colorbar(sc, label="Wine Class")
plt.show()