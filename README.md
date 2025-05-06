# ClassifiationSmS
# 📈 Analyse Statistique des Classes de Test - JFreeChart

Ce projet Python analyse les métriques de classes de test extraites du projet **JFreeChart** à partir d'un fichier CSV.

## 🔧 Fichier utilisé
- `jfreechart-test-stats.csv` : contient les colonnes suivantes :
  - `TLOC` : lignes de code dans les classes de test
  - `TASSERT` : nombre d'assertions dans les classes de test
  - `WMC` : complexité des classes (Weighted Methods per Class)

## 🧪 Objectifs

1. **Analyser la corrélation** entre les métriques `TLOC`, `TASSERT`, et `WMC`.
2. **Visualiser** les relations à l'aide de **scatter plots** et **régressions linéaires**.
3. **Comparer les groupes** de classes avec plus ou moins de 20 assertions.
4. Effectuer un **test t de Student** pour déterminer si les moyennes de `TLOC` et `WMC` diffèrent significativement entre les groupes.

## 📊 Résultats attendus

- **Corrélations** entre les métriques.
- **Visualisation graphique** des tendances linéaires.
- **Statistiques descriptives** (moyennes, écarts-types).
- **Résultats du test t** et **degrés de liberté**.

## 📦 Librairies utilisées

- `pandas` pour manipuler les données
- `numpy` pour les calculs mathématiques
- `matplotlib` pour les visualisations
- `math` pour le calcul du test t

## 🚀 Exécution

Assurez-vous d’avoir le fichier CSV dans le même répertoire que le script, puis exécutez :
```bash
python analyse_test_stats.py
