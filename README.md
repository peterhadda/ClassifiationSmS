📲 SMS Spam Classification
Ce projet explore différentes approches de classification de texte pour détecter les spams dans des messages SMS, à l’aide de plusieurs techniques de vectorisation et modèles d’apprentissage automatique.

📁 Dataset
SMS Spam Collection Dataset : Chaque message est étiqueté soit comme ham (non-spam), soit comme spam.

Source : Kaggle / Google Drive

Format : CSV (spam.csv)

🛠️ Méthodes utilisées
🔹 Partie 1 : Vectorisation Classique (Bag of Words & TF-IDF)
Nettoyage des textes : minuscules, suppression de caractères spéciaux.

Bag of Words et TF-IDF : transformation des messages en vecteurs numériques avec max_features=5000.

Modèles testés :

Régression Logistique

Random Forest

MLP Classifier

Évaluation : Validation croisée 5 folds, avec accuracy et F1-score.

🔹 Partie 2 : Embeddings SBERT
Utilisation du modèle pré-entraîné 'all-MiniLM-L6-v2' via sentence-transformers.

Chaque message est transformé en embedding sémantique.

Modèles utilisés : mêmes que Partie 1

Évaluation identique (cross-validation)

🔹 Partie 3 : Super Learner
Combinaison des prédictions de 3 modèles de base (LogReg, Random Forest, SVC) via un méta-modèle (Régression Logistique).

Utilisation de TF-IDF comme représentation de texte.

Apprentissage des poids des modèles de base pour améliorer les performances globales.

📊 Résultats (Exemples)
Méthode	Accuracy	F1-Score
BoW (LogReg)	0.982	0.935
TF-IDF (LogReg)	0.984	0.940
SBERT (LogReg)	0.989	0.961
Super Learner	0.991	0.964

(Les valeurs ci-dessus sont des exemples, à ajuster selon tes résultats exacts.)

📈 Visualisation
Graphique des performances en fonction de max_features pour BoW et TF-IDF.

Comparaison des approches dans la console.

🧠 Avantages des méthodes
BoW/TF-IDF : Simples, rapides, mais sensibles au vocabulaire.

SBERT : Capture la sémantique, plus robuste au sens du message.

Super Learner : Combine les forces des modèles pour une performance optimale.

🐍 Dépendances
bash
Copy
Edit
pip install pandas scikit-learn matplotlib sentence-transformers numpy
📂 Fichiers livrés
tp_classification_sms.py : Code principal avec les trois parties.

spam.csv : Dataset à télécharger séparément.

rapport.pdf ou .ipynb (à joindre selon les consignes).



