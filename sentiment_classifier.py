# Importação das bibliotecas necessárias

# Classe para gerar embeddings de textos
from FlagEmbedding import FlagModel

# Modelo de classificação binária usando Regressão Logística
from sklearn.linear_model import LogisticRegression

# Função para dividir dados em treino e teste (não utilizada neste código)
from sklearn.model_selection import train_test_split

# Função para calcular acurácia (não utilizada neste código)
from sklearn.metrics import accuracy_score

# ----- Conjunto de dados -----

# Lista de frases rotuladas manualmente
frases = [
    # Positivas
    "Eu adorei o filme, foi incrível!",
    "O produto é excelente, super recomendo!",
    "Estou muito feliz hoje",
    "O atendimento foi maravilhoso",
    "O livro é muito inspirador",
    "Essa música me deixa animado",
    "A comida estava deliciosa",
    "A viagem foi perfeita",
    "O serviço foi rápido e eficiente",
    "O carro é confortável e econômico",
    "A festa foi muito divertida",
    "O professor explica muito bem",
    "O celular é rápido e bonito",
    "O jogo é viciante e emocionante",
    "O hotel é limpo e aconchegante",
    "A série é envolvente",
    "O restaurante tem ótimo ambiente",
    "Minha experiência foi excelente",
    "Estou muito satisfeito com a compra",
    "Valeu muito a pena",
    "Eu acho tudo isso tão bom",
    "Que clima legal",
    
    # Negativas
    "O atendimento foi péssimo",
    "Não gostei da comida",
    "O filme foi uma perda de tempo",
    "O celular é lento e trava muito",
    "O quarto do hotel estava sujo",
    "A entrega atrasou bastante",
    "A música é irritante",
    "O produto veio com defeito",
    "O serviço foi muito ruim",
    "A viagem foi decepcionante",
    "O jogo é chato e repetitivo",
    "O professor não explica direito",
    "O carro é desconfortável",
    "A festa foi entediante",
    "O restaurante estava lotado e demorado",
    "Minha experiência foi terrível",
    "Não valeu o preço",
    "O aplicativo trava o tempo todo",
    "O livro é muito cansativo",
    "O filme é confuso e mal feito",
    "Situação bem chata essa",
    "Isso não é nada legal"
]

# Lista de rótulos correspondentes às frases (1 = positivo, 0 = negativo)
labels = [
    # Positivas
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # Negativas
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

# ----- Geração de embeddings -----

# Inicializa o modelo de embeddings BGE Small (HuggingFace)
# - Modelo público, não requer autenticação
# - Usa precisão float16 para melhor desempenho
modelo = FlagModel(
    'BAAI/bge-small-en-v1.5',
    use_fp16=True
)

# Converte as frases para vetores numéricos
embeddings = modelo.encode(frases)

# ----- Treinamento do modelo -----

# Cria o modelo de regressão logística
logistic_model = LogisticRegression()

# Treina o modelo com os embeddings e rótulos
logistic_model.fit(embeddings, labels)

# ----- Predição -----

# Solicita uma frase ao usuário
frase_test = input("\nInsira uma frase para prever o sentimento dela (Positivo ou Negativo): ")

# Converte a frase digitada para embedding
embedding_test = modelo.encode([frase_test])

# Realiza a previsão
result = logistic_model.predict(embedding_test)
prob = logistic_model.predict_proba(embedding_test)

# ----- Resultados -----

# Exibe o resultado da classificação
print("\nResultado:", "Positivo" if result[0] else "Negativo")

# Exibe as probabilidades de cada classe
print(f"\nProbabilidade de ser Negativo: {round(prob[0][0] * 100, 3)}%")
print(f"Probabilidade de ser Positivo: {round(prob[0][1] * 100, 3)}%\n")
