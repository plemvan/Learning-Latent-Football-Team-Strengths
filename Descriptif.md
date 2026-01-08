# Plan du projet : From Bradley-Terry to Neural Bradley-Terry


## 🎯 Objectifs du projet


1. **Implémentation** : Implémenter le modèle de Bradley-Terry classique et le modèle de Bradley-Terry neuronal (NBTR)


2. **Évaluation** : Évaluer et comparer les performances des deux modèles (probabilités de victoire, forces latentes, généralisation à de nouveaux matchs, généralisation à de nouvelles équipes)

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

**Statistiques d'équipes :**
- Valeur de l'effectif
- Average buts concédés
- Average buts marqués
- "Force" sur une saison

---

## 🗺️ Structure du projet

### PHASE 1 : Implémentation des deux modèles

**Objectif :** Implémenter les deux modèles

#### 1.1 Modèle Bradley-Terry Classique
- [x] Implémentation de base (`BT_classical.py`) ✅
- [ ] Extension de Davidson pour gérer les matchs nuls (paramètre $\nu$ : propension aux égalités)

**Méthode :**
- MLE via gradient descent
- Input : matrice de résultats entre équipes
- Output : vecteur de ratings $\theta_i$ pour chaque équipe connue

**⚠️ Important :** Considérer $\nu$ comme un hyperparamètre du modèle, on ne l'estime pas durant l'apprentissage. (Par contre on peut l'estimer avant pour le fixer à une valeur pas déconnante)
- Si on l'estime durant l'apprentissage des modèles, on aura un estimateur supplémentaire mais qu'on ne pourra pas interpréter avec les $\theta_i$ estimés.
- Donc autant le fixer à l'avance comme un hyper paramètre, et on expliquera dans le rapport qu'on l'a fixé parce que c'est pas une quantité qui nous intéresse pour l'évaluation mais juste un paramètre qui rend le modèle plus proche de la réalité.


#### 1.2 NBTR 

```python
# À compléter
```

**⚠️ Important :** Également considérer $\nu$ comme un hyperparamètre du modèle.

---

### PHASE 2 : Comparaison des performances des deux modèles (généralisation à de nouveaux matchs avec les mêmes équipes)

#### 2.1 Protocole d'évaluation / comparaison

**Objectif :** Comparer les modèles sur ce qu'ils savent tous deux faire

Les deux modèles ont la capacité d'apprendre les forces des équipes puis de prédire le résultat de nouveaux matchs entre ces mêmes équipes.

Les modèles nous renvoient $\hat\theta = \left( \hat\theta_i \right)_1^n$. A partir de cela, on dispose de deux quantités :
- Les probabilités de victoires estimées $\hat p_{ij}$ (liées aux matchs)
- Les forces latentes estimées $\hat\theta_i$ (liées aux équipes)

**Split des données :**

Comme le modèle de Bradley-Terry classique ne pas généraliser à de nouvelles équipes, le plus simple est d'utiliser une seule saison pour l'évaluation, et de faire un split des matchs (on peut répéter l'opération avec plusieurs saisons si on veut).
- Train : 27 premières (ou 24 si 18 équipes) journées du championnat. On entraine et on finetune/valide NBTR sur ces données, puis on réentraine NBTR et BT sur ces données pour estimer les forces $\hat\theta_i$.
- Test : 11 dernières (ou 10 si 18 équipes) journées du championnat. On compare les probas de victoire $\hat p_{ij}$ estimées avec les résultats des matchs.

**Métriques :**

Vu qu'on dispose des forces estimées et des probabilités de victoires estimées, on évalue les modèles sur deux niveaux distincts :
- **Niveau MATCH (proba de victoires) :** Prédiction des résultats des derniers matchs de la saison

    | Métrique | Description | Formule/Méthode |
    |----------|-------------|-----------------|
    |**log Loss**|Comparaison des log Loss des 2 modèles |Σ log P(résultat observé) sur le test|
    ||||


- **Niveau ÉQUIPE (forces latentes) :** Cohérence des forces latentes avec les résultats observés

    | Métrique | Description | Formule/Méthode |
    |----------|-------------|-----------------|
    |**Corrélation forces/winrates**|Comparaison forces estimées et winrates (pour chaque modèle) |Corrélation de Spearman|
    |**Corrélation forces BT/forces NBTR**|Comparaison des estimations des deux modèles|Corrélation de Spearman|

- **(Optionnel) Stabilité/Robustesse :** Étude de la variance sur plusieurs runs des modèles.
    
    On garde les mêmes données de test et on lance les tests des modèles plusieurs fois avec différentes seeds aléatoires et on regarde la variance des forces estimées et la variance de la log Loss pour chacun des deux modèles.



---

### PHASE 3 : Application - Généralisation à de nouvelles équipes

**Objectif :** Démontrer la capacité de NBTR à prédire la force d'équipes absentes du training set

**Idée :**  Utiliser les équipes fraîchement promues de Ligue 2 

```python
# À compléter
```

---