import pandas as pd
import re
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,f1_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score




df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]

df.columns=['label','message']
df['label']=df['label'].map({'ham':0,'spam':1})

def clean_text(text):
    text=text.lower()
    return re.sub(r"[^a-z0-9\s]",'',text)

df['message_clean']=df['message'].apply(clean_text)

count_vect = CountVectorizer(max_features=5000)
tfidf_vect = TfidfVectorizer(max_features=5000)


X_bow=count_vect.fit_transform(df['message_clean'])
X_tfidf=tfidf_vect.fit_transform(df['message_clean'])
y=df['label']

models={
    "LogReg":LogisticRegression(max_iter=1000),
    "RandomForest":RandomForestClassifier(),
    "MLP":MLPClassifier(max_iter=300)

}

def evaluate_models(X,y):
    results={}
    for name,model in models.items():
        accuracy=cross_val_score(model,X,y,cv=5,scoring='accuracy').mean()
        f1=cross_val_score(model,X,y,cv=5,scoring=make_scorer(f1_score)).mean()
        results[name] ={'accuracy':accuracy,'f1':f1}
    return results

results_bow=evaluate_models(X_bow,y)
results_tfidf=evaluate_models(X_tfidf,y)


feature_range=[500,1000,2000,3000,4000,5000]
acc_bow=[]
acc_tfidf=[]


for max_feature in feature_range:
       bow=CountVectorizer(max_features=max_feature).fit_transform(df['message_clean'])
       tfidf= TfidfVectorizer(max_features=max_feature).fit_transform(df['message_clean'])
       acc_bow.append(cross_val_score(LogisticRegression(max_iter=1000),bow,y,cv=5,scoring='accuracy').mean())
       acc_tfidf.append(cross_val_score(LogisticRegression(max_iter=1000),tfidf,y,cv=5,scoring='accuracy').mean())


acc_bow = [float(x) for x in acc_bow]
acc_tfidf = [float(x) for x in acc_tfidf]
print("max_features\tAccuracy_BoW\tAccuracy_TF-IDF")
for mf, bow, tfidf in zip(feature_range, acc_bow, acc_tfidf):
    print(f"{mf}\t\t{bow}\t\t{tfidf}")




# Partie 2 : Classification avec Sentence Transformers 

# 2. Instancier le modèle léger SBERT
print("Modèle SBERT - PARTIE 2")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Générer les embeddings pour chaque message nettoyé
embeddings = sbert_model.encode(df['message_clean'].tolist(), show_progress_bar=False)

# 4. Définir les modèles à tester
models_sbert = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "MLP Classifier": MLPClassifier(max_iter=300)
}

# 5. Fonction pour évaluer les modèles avec validation croisée
def evaluate_models_sbert(X, y):
    results = {}
    for name, model in models_sbert.items():
        print(f"Évaluation du modèle : {name}")
        accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        f1 = cross_val_score(model, X, y, cv=5, scoring=make_scorer(f1_score)).mean()
        results[name] = {'accuracy': accuracy, 'f1_score': f1}
    return results

# 6. Évaluer les performances sur les embeddings SBERT
print("Evaluation des performances avec validation croisee")
results_sbert = evaluate_models_sbert(embeddings, y)

# 7. Affichage des résultats comparés
print("\n=== Résultats comparés ===")
print("Bag of Words :", results_bow)
print("TF-IDF :", results_tfidf)
print("SBERT Embeddings :", results_sbert)




# Partie 3 

# 1. Définir les modèles de base
base_models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("RandomForest", RandomForestClassifier()),
    ("SVC", SVC(probability=True)) 
]

# 2. Générer les prédictions out-of-fold pour chaque modèle
Z = []  # matrice des predictions
for name, model in base_models:
    print(f"Prédictions croisées pour {name}")
    preds = cross_val_predict(model, X_tfidf, y, cv=5, method="predict_proba")
    Z.append(preds[:, 1].reshape(-1, 1)) 

# 3. Construction la matrice Z pour entraîner le méta-modèle
Z_stacked = np.hstack(Z)

# 4. Entraîner le méta-modèle (régression logistique)
meta_model = LogisticRegression()
meta_model.fit(Z_stacked, y)

# 5. Prédictions finales
y_pred = meta_model.predict(Z_stacked)

# 6. Évaluation
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\n=== Résultats Super Learner ===")
print("Accuracy :", accuracy)
print("F1-score :", f1)

# 7. Poids attribués par le méta-modèle
print("\nPoids des modèles de base :")
for (name, _), coef in zip(base_models, meta_model.coef_[0]):
    print(f"{name} : {coef:.4f}")