# 📲 SMS Spam Classification

Ce projet explore différentes approches de classification de texte pour détecter les spams dans des messages SMS, à l’aide de plusieurs techniques de vectorisation et modèles d’apprentissage automatique.

## 📁 Dataset

- **SMS Spam Collection Dataset** : Chaque message est étiqueté comme `ham` (non-spam) ou `spam`.
- Source : Kaggle / Google Drive
- Format : CSV (`spam.csv`)

## 🛠️ Méthodes Utilisées

### 🔹 Partie 1 : Vectorisation Classique (Bag of Words & TF-IDF)

- **Nettoyage des messages** : conversion en minuscules, suppression des caractères spéciaux.
- **Bag of Words** et **TF-IDF** :
  - Transformation des textes avec `max_features=5000`.
- **Modèles de classification testés** :
  - Logistic Regression
  - Random Forest Classifier
  - Multi-Layer Perceptron (MLPClassifier)
- **Évaluation** :
  - Validation croisée à 5 folds
  - Accuracy & F1-Score

### 🔹 Partie 2 : Embeddings avec Sentence Transformers

- Modèle utilisé : `'all-MiniLM-L6-v2'` via `sentence-transformers`
- Chaque message est transformé en vecteur sémantique (embedding)
- **Modèles testés** :
  - Logistic Regression
  - Random Forest
  - MLP
- **Évaluation** :
  - Cross-validation (5 folds)

### 🔹 Partie 3 : Super Learner

- Combinaison de 3 modèles de base :
  - Logistic Regression
  - Random Forest
  - Support Vector Classifier (SVC)
- **Représentation utilisée** : TF-IDF
- **Métha-modèle** : Régression Logistique
- **Objectif** : Apprendre les poids optimaux de chaque base learner pour minimiser l’erreur globale.

## 📊 Résultats Comparés (Exemple)

| Méthode            | Accuracy | F1-Score |
|--------------------|----------|----------|
| BoW (LogReg)       | 0.982    | 0.935    |
| TF-IDF (LogReg)    | 0.984    | 0.940    |
| SBERT (LogReg)     | 0.989    | 0.961    |
| Super Learner      | 0.991    | 0.964    |

> (*Les chiffres sont à ajuster selon vos résultats exacts.*)

## 📈 Visualisations

- Courbes de performance en fonction de `max_features` pour BoW et TF-IDF.
- Comparaison globale des approches dans la console.

## 🧠 Avantages et Inconvénients

| Approche      | Avantages | Inconvénients |
|---------------|-----------|---------------|
| BoW/TF-IDF    | Rapide, simple | Ne capte pas le sens des mots |
| SBERT         | Représente la sémantique | Plus coûteux en ressources |
| Super Learner | Combine les forces de chaque modèle | Implémentation plus complexe |

## 📦 Dépendances

```bash
pip install pandas scikit-learn matplotlib sentence-transformers numpy




