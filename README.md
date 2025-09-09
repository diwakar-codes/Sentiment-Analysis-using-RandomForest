# Sentiment Analysis using Random Forest  

## Overview  
This project performs Sentiment Analysis on textual data (Amazon Alexa reviews) using Machine Learning.  
The initial implementation uses a Random Forest Classifier to classify sentiments as Positive or Negative.  

The goal is to build a scalable pipeline that can later integrate multiple models (e.g., Logistic Regression, Naive Bayes, SVM, XGBoost, and Deep Learning models like LSTMs/Transformers) for comparison and performance tuning.  

---

## Project Structure  
- `Sentiment-Analysis.ipynb` → Jupyter Notebook containing data preprocessing, feature extraction, model training, and evaluation.  
- `data/` → Dataset (Amazon Alexa reviews).  
- `README.md` → Project documentation.  

---

## Steps Implemented  
1. Data Preprocessing  
   - Cleaning reviews (lowercasing, removing stopwords and punctuation).  
   - Label encoding for sentiment.  

2. Feature Engineering  
   - Bag-of-Words / TF-IDF Vectorization.  

3. Model Training  
   - Implemented Random Forest Classifier.  

4. Evaluation  
   - Accuracy, Precision, Recall, and F1-score.  
   - Confusion Matrix visualization.  

---

## Future Enhancements  
- Add more models: Logistic Regression, Naive Bayes, SVM, XGBoost, LSTMs, BERT/Transformers.  
- Hyperparameter tuning (GridSearchCV, Optuna).  
- Use Word Embeddings (Word2Vec, GloVe, FastText).  
- Deploy model with a Flask/Django API or Streamlit dashboard.  
- Add performance comparison charts.  

---

## Tech Stack  
- Language: Python  
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, nltk  

---

