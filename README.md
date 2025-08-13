# 🧠 Classificador de Sentimentos com Embeddings e Regressão Logística  

Este projeto foi desenvolvido como prática da disciplina de **Inteligência Artificial**, aplicando **processamento de linguagem natural (NLP)** para classificar frases como **positivas** ou **negativas**.  

Utiliza embeddings para converter frases em representações vetoriais numéricas e um modelo de **Regressão Logística** para realizar a classificação.  

---

## ✨ Funcionalidades  

- Conversão de texto para **vetores de embeddings** utilizando o modelo **BGE Small (BAAI)**  
- Treinamento de um modelo de **Regressão Logística** para classificação binária (positivo/negativo)  
- Entrada de frases pelo usuário para prever o sentimento  
- Retorno da predição com **probabilidade associada** a cada classe  

---

## 💡 Conceitos aplicados  

- **Embeddings**: técnica para representar textos em vetores numéricos que preservam significado semântico  
- **Classificação supervisionada**: treinamento com dados rotulados (frases + rótulos)  
- **Regressão Logística**: modelo estatístico para classificação binária  
- **Pipeline de Machine Learning**: pré-processamento → treinamento → predição  
- **Interpretação de probabilidades**: uso de `predict_proba` para estimar confiança do modelo  

---

## 🛠 Ferramentas e bibliotecas utilizadas  

- **Python** (linguagem principal)  
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — para gerar embeddings  
- [Scikit-learn](https://scikit-learn.org/stable/) — para treino, teste e avaliação do modelo de Regressão Logística  

---

## 📷 Exemplo de uso  

```bash
Insira uma frase para prever o sentimento dela (Positivo ou Negativo): Gostei muito da viagem!

Resultado: Positivo

Probabilidade de ser Negativo: 12.356%
Probabilidade de ser Positivo: 87.644%
```
---

## 🚀 Como executar

1. Instalar as dependências
```bash
pip install FlagEmbedding scikit-learn
```
2. Executar o scriptr
```bash
python sentiment_classifier.py
```
---

## 📌 Melhorias futuras

- Ampliar dataset de treino para maior robustez do modelo
- Implementar suporte para múltiplos idiomas
- Criar interface web para interação com o classificador
- Aplicar técnicas de balanceamento de dados para melhorar a precisão

---

## 👤 Autor

**Enzo Cerneviva**  
Estudante de Ciência da Computação — FIAP  
**🌐 [LinkedIn](https://www.linkedin.com/in/enzocerneviva)**  
**💻 [GitHub](https://github.com/enzocerneviva)**

