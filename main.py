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

# ==============================================================================
# FASE 2: PIPELINE DE PRÉ-PROCESSAMENTO E DIVISÃO DOS DADOS
# ==============================================================================
"""
OBJETIVO:
- Realizar a divisão estratificada (Stratified Split) em Treino, Validação e Teste (70% / 10% / 20%).
- Aplicar a normalização dos pixels para a escala [0.0, 1.0].
- Justificar tecnicamente a importância da normalização.
"""

print("\n--- FASE 2: Pré-processamento e Divisão Estratificada ---")

# 1. Divisão Inicial: 80% Treino+Validação e 20% Teste (Garantindo estratificação)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Divisão Secundária: Separar Treino (70% do total) e Validação (10% do total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)

print(f"Conjunto de Treino: {X_train.shape} amostras ({X_train.shape[0] / X.shape[0]:.0%})")
print(f"Conjunto de Validação: {X_val.shape} amostras ({X_val.shape[0]/X.shape[0]:.0%})")
print(f"Conjunto de Teste: {X_test.shape} amostras ({X_test.shape[0]/X.shape[0]:.0%})")


# 3. Normalização dos valores dos pixels para o intervalo [0.0, 1.0] e conversão para float32
X_train_norm = X_train.astype('float32') / 255.0
X_val_norm = X_val.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

"""
JUSTIFICATIVA TÉCNICA DA NORMALIZAÇÃO (FASE 2):
A normalização ajusta os valores das características para o intervalo [0.0, 1.0].
Isso impede que algoritmos baseados em distâncias métricas (como KNN ou SVM) ou baseados em otimização
por gradiente (como Redes Neurais/MLP) sofram com instabilidade numérica, além de acelerar substancialmente
a convergência do treinamento ao manter as derivadas parciais e taxas de aprendizado em magnitudes adequadas.
"""

# ==============================================================================
# FASE 3: IMPLEMENTAÇÃO E TREINAMENTO DE 3 MODELOS
# ==============================================================================
"""
OBJETIVO:
- Treinar 3 modelos distintos com ajuste e variação justificada de ao menos 2 hiperparâmetros cada:
  1. Random Forest (Métricas de Árvores de Decisão Ensembled)
  2. SVM - Support Vector Machine (Kernel RBF e Marge Magnitudes)
  3. Perceptron Multicamadas - MLP (Rede Neural Artificial com TensorFlow/Keras ou Scikit-Learn)
"""

print("\n--- FASE 3: Treinamento dos 3 Modelos ---")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,      # Hiperparâmetro 1: Número de árvores na floresta
        max_depth=15,          # Hiperparâmetro 2: Profundidade máxima de cada árvore
        random_state=42,
        n_jobs=-1
    ),
    "SVM (RBF)": SVC(
        C=5.0,                 # Hiperparâmetro 1: Fator de penalização de erros (Regularização)
        kernel='rbf',          # Hiperparâmetro 2: Função de Kernel para Mapeamento Não-Linear
        gamma='scale',
        random_state=42
    ),
    "Rede Neural MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64), # Hiperparâmetro 1: Arquitetura com 2 camadas ocultas
        activation='relu',            # Hiperparâmetro 2: Função de ativação não-linear
        max_iter=30,
        random_state=42
    )
}

trained_models = {}
execution_times = {}

for name, model in models.items():
    print(f"Treinando modelo: {name}...")
    start_time = time.time()
    model.fit(X_train_norm, y_train)
    elapsed_time = time.time() - start_time

    trained_models[name] = model
    execution_times[name] = elapsed_time
    print(f"-> {name} treinado em {elapsed_time:.2f} segundos.")

# ==============================================================================
# FASE 4: AVALIAÇÃO COMPARATIVA DE DESEMPENHO
# ==============================================================================
"""
OBJETIVO:
- Avaliar cada modelo no conjunto de teste independente.
- Gerar Matriz de Confusão 10x10 com Mapa de Calor para cada modelo.
- Criar Tabela Comparativa Consolidada (Acurácia, Precisão, Recall, F1-Score).
- Redigir Conclusão Técnica abordando erros comuns e custo computacional.
"""

print("\n--- FASE 4: Avaliação Comparativa ---")

results_list = []

# Loop para cálculo das métricas e exibição das matrizes de confusão
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_norm)

    # Métricas Globais Ponderadas
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    results_list.append({
        "Modelo": name,
        "Acurácia Global": round(acc, 4),
        "Precisão (Ponderada)": round(prec, 4),
        "Recall (Ponderado)": round(rec, 4),
        "F1-Score (Ponderado)": round(f1, 4),
        "Tempo Treino (s)": round(execution_times[name], 2)
    })

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f"Matriz de Confusão: {name}", fontweight='bold')
    axes[i].set_xlabel("Rótulo Predito")
    axes[i].set_ylabel("Rótulo Verdadeiro")

plt.tight_layout()
plt.show()

# Tabela Comparativa Consolidada
df_results = pd.DataFrame(results_list)
print("\n--- TABELA COMPARATIVA CONSOLIDADA ---")
print(df_results.to_string(index=False))

"""
CONCLUSÃO TÉCNICA (FASE 4):
1. Maior Taxa de Confusão: As principais confusões ocorrem entre os dígitos '4' vs '9' e '3' vs '5', devido ao formato morfológico e curvatura semelhantes nos traços manuscritos.
2. Desempenho dos Modelos: O modelo SVM e a Rede Neural MLP apresentam superioridade na captura de fronteiras de decisão complexas em relação à Random Forest.
3. Custo Computacional: A Rede Neural MLP e o SVM exigem um tempo de processamento maior no treinamento, porém justificam o investimento computacional ao atingir F1-Scores superiores no conjunto de teste.
"""

# ==============================================================================
# FASE 5.1 & 5.2: DESAFIOS A E B - ROBUSTEZ E GENERALIZAÇÃO EXTREMA (OOD)
# ==============================================================================
"""
OBJETIVO:
- Fase 5.1 (Desafio A): Ocultar totalmente duas classes (ex: dígitos 4 e 7) do treino.
- Fase 5.2 (Desafio B): Testar o modelo treinado apenas nas imagens das classes ocultadas.
- Analisar o comportamento do classificador diante de dados OOD (Out-of-Distribution) e debater o conceito de Falsa Certeza (Overconfidence).
"""

print("\n--- FASE 5.1 & 5.2: Teste de Generalização Extrema (OOD) ---")

# Seleção das classes ocultadas
classes_ocultas = [4, 7]
print(f"Ocultando do conjunto de treinamento as classes: {classes_ocultas}")

# Filtrando o conjunto de treino para REMOVER as classes 4 e 7
mask_train = ~np.isin(y_train, classes_ocultas)
X_train_masked = X_train_norm[mask_train]
y_train_masked = y_train[mask_train]

# Treinando um modelo mascarado (Random Forest)
model_ood = RandomForestClassifier(n_estimators=100, random_state=42)
model_ood.fit(X_train_masked, y_train_masked)
print("Modelo mascarado treinado com sucesso sem ver os dígitos 4 e 7!")

# Filtrando o conjunto de teste para MANTER EXCLUSIVAMENTE as classes ocultadas
mask_test_ood = np.isin(y_test, classes_ocultas)
X_test_ood = X_test_norm[mask_test_ood]
y_test_ood = y_test[mask_test_ood]

# Inferência OOD
y_pred_ood = model_ood.predict(X_test_ood)

# Exibição das atribuições forçadas
df_ood_assign = pd.DataFrame({'Rótulo Real': y_test_ood, 'Classe Atribuída Pelo Modelo': y_pred_ood})
print("\nFrequência das classes atribuídas aos dígitos desconhecidos (4 e 7):")
print(df_ood_assign['Classe Atribuída Pelo Modelo'].value_counts())

# Plot da Matriz de Confusão OOD
cm_ood = confusion_matrix(y_test_ood, y_pred_ood, labels=list(range(10)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ood, annot=True, fmt='d', cmap='Oranges')
plt.title("Matriz de Confusão - Inferência OOD (Dígitos 4 e 7 Ocultos no Treino)", fontweight='bold')
plt.xlabel("Rótulo Atribuído pelo Modelo")
plt.ylabel("Rótulo Real (4 ou 7)")
plt.show()

"""
ANÁLISE DE OVERCONFIDENCE (FALSA CERTEZA):
Quando submetido a dados fora da distribuição conhecida (Out-of-Distribution), o modelo determinístico é forçado a classificar o elemento desconhecido em uma das classes previamente aprendidas. O dígito '4' é frequentemente confundido com '9' ou '1', enquanto o '7' é associado ao '1'. Modelos clássicos e redes neurais sofrem de "Falsa Certeza" (Overconfidence), emitindo probabilidades elevadas mesmo para dados totalmente fora da sua distribuição de treino.
"""

# ==============================================================================
# FASE 5.3: DESAFIO C - INFERÊNCIA COM IMAGENS MANUSCRITAS PRÓPRIAS
# ==============================================================================
"""
OBJETIVO:
- Criar pipeline de pré-processamento de imagens personalizadas (fotografadas ou desenhadas).
- Passos: Conversão para escala de cinza, inversão de cores, centralização de massa/bounding box, resize 28x28 e normalização.
- Predição com o melhor modelo e plotagem das probabilidades de saída.
"""
def preprocess_custom_image(image_path):
    # 1. Abre a imagem e converte para escala de cinza ('L')
    img = Image.open(image_path).convert('L')
    
    # 2. Redimensiona para o tamanho exato do MNIST (28x28)
    img = img.resize((28, 28))
    
    # 3. Converte para array do NumPy
    img_array = np.array(img, dtype='float32')
    
    # --- CORREÇÃO 1: Inverter as cores se o fundo for claro ---
    # Se a média dos pixels for maior que 127, significa que o fundo é predominantemente branco.
    if img_array.mean() > 127:
        img_array = 255.0 - img_array
        
    # --- CORREÇÃO 2: Limpar ruídos de fundo (Binarização/Limiarização) ---
    # Tudo que for cinza escuro vira preto absoluto (0), destacando o traço do número.
    img_array[img_array < 50] = 0.0
    
    # 4. Normaliza para o intervalo [0.0, 1.0] exigido pela rede
    img_array = img_array / 255.0
    
    # 5. Achata para o formato de vetor (1, 784) para o MLPClassifier
    img_vector = img_array.reshape(1, 784)
    
    return img_array, img_vector


print("\n--- FASE 5.3: Testando a imagem 'numero_dois.png' ---")

# 1. Aplica o pré-processamento na sua imagem
sample_img_array, sample_img_vector = preprocess_custom_image('numero_dois.png')

# CODIGO TEMPORARIO PARA VER A IMAGEM PROCESSADA
import matplotlib.pyplot as plt

# Adicione isso logo após carregar a imagem customizada
plt.imshow(sample_img_array, cmap='gray')
plt.title("O que a rede neural está vendo de verdade")
plt.show()

# FIM CODIGO TEMPORARIO

# 2. Faz a predição com o melhor modelo (SVM)
best_model = trained_models["SVM (RBF)"]

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(sample_img_vector)[0]
else:
    df_scores = best_model.decision_function(sample_img_vector)[0]
    probabilities = np.exp(df_scores) / np.sum(np.exp(df_scores))

pred_class = np.argmax(probabilities)

# 3. Exibe a imagem processada (28x28) e o gráfico de probabilidades
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(sample_img_array, cmap='gray')
ax1.set_title(f"Imagem Própria: numero_dois.png\nPredição da IA: {pred_class}", fontweight='bold')
ax1.axis('off')

bars = ax2.bar(range(10), probabilities, color='skyblue')
bars[pred_class].set_color('green')  # Destaque na classe predita
ax2.set_xticks(range(10))
ax2.set_xlabel("Dígitos (0 a 9)")
ax2.set_ylabel("Probabilidade / Confiança")
ax2.set_title("Distribuição de Probabilidades por Classe", fontweight='bold')
ax2.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

print(f"O modelo identificou a imagem como o número: {pred_class}")
