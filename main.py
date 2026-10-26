# ==============================================================================
# MINI-PROJETO AVALIATIVO - MÓDULO 2 (SEMANA 05)
# TÍTULO: Desenvolvimento de IA para Análise Preditiva e Classificação Multiclasse
# DATASET: MNIST_784 (70.000 imagens 28x28)
# ==============================================================================

# --- IMPORTAÇÃO DE BIBLIOTECAS PRINCIPAIS ---
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Modelos para a Fase 3
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Bibliotecas para processamento de imagem própria (Fase 5.3)
from PIL import Image, ImageOps
import cv2

# Configurações visuais do Matplotlib e Seaborn
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_theme(style="whitegrid")

# ==============================================================================
# FASE 1: CARREGAMENTO E ANÁLISE EXPLORATÓRIA DE IMAGENS (EDA)
# ==============================================================================
"""
OBJETIVO:
- Importar o dataset MNIST via OpenML.
- Exibir a dimensionalidade das matrizes X e y.
- Comprovar o balanceamento das 10 classes (dígitos de 0 a 9).
- Gerar uma grade visual 2x5 exibindo amostras originais.
- Documentar a estrutura vetorial (28x28 = 784 features) e escala de pixels (0 a 255).
"""

print("--- FASE 1: Carregamento do Dataset MNIST ---")
# Carregando a base de dados MNIST_784
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

X, y = mnist.data, mnist.target.astype(int)

print(f"Dimensão da matriz de características (X): {X.shape}")
print(f"Dimensão do vetor de rótulos (y): {y.shape}")

# Verificação do balanceamento de classes
df_y = pd.Series(y).value_counts().sort_index().reset_index()
df_y.columns = ['Dígito', 'Quantidade']
print("\nDistribuição das classes:")
print(df_y)

# Plotagem do Histograma de Balanceamento das Classes
plt.figure(figsize=(8, 4))
sns.barplot(data=df_y, x='Dígito', y='Quantidade', hue='Dígito', palette='viridis', legend=False)
plt.title("Distribuição das Classes no Dataset MNIST", fontsize=14, fontweight='bold')
plt.xlabel("Dígito (Classe)")
plt.ylabel("Frequência de Imagens")
plt.show()

# Grade Visual 2x5 com exemplos de cada dígito
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Amostras Representativas do Dataset MNIST (28x28 pixels)", fontsize=14, fontweight='bold')

for i in range(10):
    idx = np.where(y == i)[0]  # Obtém os índices das ocorrências do dígito i
    first_img_idx = idx[0]     # Seleciona o primeiro índice encontrado
    img = X[first_img_idx].reshape(28, 28)
    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Rótulo: {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()

"""
EXPLICAÇÃO TÉCNICA DA ESTRUTURA DOS DADOS (FASE 1):
- Representação Vetorial: Cada imagem de 28x28 pixels bidimensional é planificada (flattened) em um vetor unidimensional de 784 atributos (features).
- Escala de Intensidade: Os valores dos pixels variam de 0 (fundo preto/ausência de traço) a 255 (traço branco com brilho máximo).
- Balanceamento: A distribuição é equilibrada, contendo aproximadamente 7.000 amostras por dígito (~10% do total por classe).
"""

