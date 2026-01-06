# Plan de Projet : Neural Bradley-Terry Rating appliqué à la Ligue 1


## 🎯 Objectifs du projet


1. **Implémentation** : Implémenter et comparer le modèle de Bradley-Terry classique avec sa version neuronale (NBTR)

2. **Evaluation** : Evaluer les deux approches (évaluation de modèles prédisant des outputs non observables)

3. **Application** : Démontrer la capacité de généralisation du NBTR sur de nouvelles équipes en s'intéressant aux équipes promues

---

## 📊 Données disponibles

### Dataset principal : Ligue 1 (2010-2025)
- **Format** : CSV
- **Granularité** : Une ligne = un match
- **Période** : Toutes les saisons de 2010-2011 à 2024-2025

### Colonnes disponibles :
**Informations de base :**
- Date du match
- Équipe domicile / Équipe extérieure
- Score mi-temps et score final
- Résultat du match (équipe gagnante)

**Statistiques de match (par équipe) :**
- Nombre de tirs
- Nombre de tirs cadrés
- Nombre de fautes
- Nombre de cartons jaunes/rouges
- Nombre de corners

**Ce qui manque :**
- ❌ Features "externes" sur les équipes (budget, valeur marchande, effectif)
- ❌ Autres données pour l'application aux équipes promues de L2

---

## 🗺️ Structure du projet

### PHASE 1 : Implémentation et Comparaison des deux modèles

**Objectif :** Implémenter les deux approches et les comparer rigoureusement

#### 1.1 Modèle Bradley-Terry Classique
- [x] Implémentation de base (`BT_classical.py`) ✅
- [ ] Application aux données Ligue 1
- [ ] Calcul des ratings par équipe
- [ ] Évaluation des performances

**Méthode :**
- MLE via gradient descent
- Input : matrice de résultats entre équipes
- Output : vecteur de ratings $θ_i$ pour chaque équipe connue

#### 1.2 Feature Engineering
- Création de features dérivées des statistiques de matchs

 ET/OU 

- Récupération d'un autre jeu de données pour avoir des infos sur les équipes (budget, effectif, etc...)

#### 1.3 NBTR - Implémentation

**Architecture du Rating Estimator E :**
```python
# A définir
```


#### 1.4 Protocole de Comparaison / Évaluation

**Split des données :**
- Train : saisons 2010-2011 à 2020-2021 (11 saisons)
- Validation : saisons 2021-2022 et 2022-2023 (2 saisons)
- Test : saisons 2023-2024 et 2024-2025 (2 saisons)

**Métriques d'évaluation :**

| Métrique | Description | Formule/Méthode |
|----------|-------------|-----------------|
| **Accuracy** | % prédictions correctes du vainqueur | (correct / total) × 100 |
| **Log-likelihood (?)** | Qualité probabiliste | Σ log P(résultat observé) |
| **Spearman correlation (?)** | Corrélation des rankings | Corrélation entre classement prédit et réel |
| **Brier Score (?)** | Calibration des probabilités | Moyenne de (p_prédit - y_réel)² |

**Analyses à produire :**
- [ ] Courbes de learning (train/val loss)
- [ ] Comparaison quantitative des métriques
- [ ] Visualisation des ratings appris (scatter plot BT vs NBTR)
- [ ] Analyse des erreurs : quels matchs sont mal prédits ?

---

### PHASE 2 : Application - Généralisation à de Nouvelles Équipes

**Objectif :** Démontrer la capacité de NBTR à prédire la force d'équipes absentes du training set

#### Option A : Équipes Promues de Ligue 2 (IDÉAL) 🎯

**Principe :**
- Entraîner NBTR sur équipes de Ligue 1 historiques
- Prédire le rating des équipes promues avant leur première saison en L1
- Comparer avec leur performance réelle

**Données nécessaires :**
Features qui existent AUSSI pour les équipes de Ligue 2 :
- ✅ Valeur marchande de l'effectif (Transfermarkt)
- ✅ Classement final en Ligue 2 l'année précédente
- ✅ Palmarès (nombre de titres L1/L2)
- ⚠️ Budget (si accessible)
- ⚠️ Âge moyen / Nombre d'internationaux (si accessible)

**Cas d'usage concrets :**
| Saison | Équipes promues | Utilisation |
|--------|----------------|-------------|
| 2024-25 | Auxerre, Angers, Saint-Étienne | Test set |
| 2023-24 | Le Havre, Metz | Test set |
| Avant | Autres équipes promues | Validation |

---

#### Option B : Simulation de Nouvelles Équipes

**Principe :**
- Split artificiel : retirer certaines équipes du train set
- Entraîner NBTR sans ces équipes
- Tester la prédiction sur les équipes retirées

**Protocole :**
```
Split spatial :
- Train : 80% équipes (ex: 16 équipes)
- Test : 20% équipes (ex: 4 équipes jamais vues)

Validation :
- Répéter avec plusieurs splits aléatoires
- Moyenner les performances
```

**Avantages :**
- ✅ Faisable avec données actuelles
- ✅ Pas de collecte supplémentaire

**Limites :**
- ⚠️ Moins naturel (on retire artificiellement des équipes connues)
- ⚠️ Ne répond pas à une vraie question sportive

**Analyses :**
- [ ] Accuracy de prédiction sur équipes test
- [ ] Corrélation ratings prédits vs ratings MLE réels
- [ ] Influence du nombre d'équipes dans le train set

---

## 📌 Notes et remarques

### Différences clés BT classique vs NBTR

| Aspect | BT Classique | NBTR |
|--------|--------------|------|
| **Input** | Matrice de résultats | Features + résultats |
| **Output** | Rating par équipe connue | Fonction E : features → rating |
| **Généralisation** | ❌ Impossible sur nouvelles équipes | ✅ Possible |
| **Interprétation** | Rating = force latente | Rating + importance des features |
| **Complexité** | Faible (MLE) | Élevée (NN) |

### Quand NBTR apporte de la valeur ?

**✅ NBTR est utile quand :**
- On a des features informatives sur les équipes
- On veut prédire la force d'équipes non observées
- On veut comprendre quels facteurs expliquent la force

**❌ NBTR n'apporte pas grand chose si :**
- On veut juste ranker des équipes connues (BT classique suffit)
- On n'a pas de features (ou seulement one-hot encoding)
- Dataset trop petit pour entraîner un NN

---