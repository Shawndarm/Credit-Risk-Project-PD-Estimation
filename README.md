# Credit-Risk-Project-PD-Estimation

Arborescence du projet
```
/
├── LICENSE
├── README.md
├── pyproject.toml
├── .python-version
├── uv.lock
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── update_discretization.py      # Script principal : applique boundaries, calcule stats, génère rapport
│   └── utils.py                      # Fonctions réutilisables (ChiMerge, chi2, stepwise, visualisations)
├── datasets/                         # Jeux de données (ex. credit_risk_dataset_post_prepocess.csv)
├── notebooks/                        # Jupyter notebooks d'exploration
├── docs/                             # Documentation additionnelle
└── result/                           # Sorties / artefacts (rapports, CSV, graphiques)
```

Ce dépôt contient le code et les artefacts d'un projet de credit scoring centré sur l'estimation de la Probabilité de Défaut (PD) via une régression logistique. Le pipeline principal comprend : préparation des données, discrétisation (ChiMerge / méthode mixte), sélection de variables (stepwise BIC + filtrage p-values) et génération d'un rapport texte décrivant la discrétisation finale.

---

## Vue d'ensemble du contenu

- `pyproject.toml` : dépendances et métadonnées (Python >= 3.12).
- `src/`
  - `update_discretization.py` : script principal pour appliquer des boundaries de discrétisation mises à jour, calculer statistiques par bin, test χ², et générer :
    - `credit_risk_dataset_discretized_final.csv`
    - `RAPPORT_DISCRETISATION_FINALE.txt`
  - `utils.py` : fonctions réutilisables :
    - Algorithme ChiMerge (`discretize_with_chimerge`)
    - Calculs Chi² entre bins
    - Tests d'indépendance (χ²) et matrice de Cramér's V
    - Visualisation des taux de défaut (Plotly)
    - Procédure stepwise BIC (forward + backward) puis filtrage par p-values (`stepwise_bic_forward_backward`)
- `datasets/` : dossier prévu pour les jeux de données (ex. `credit_risk_dataset_post_prepocess.csv`).
- `notebooks/` : cahiers d'exploration et analyses (Jupyter).
- `docs/` : documentation complémentaire (si présente).
- `result/` : sorties et artefacts (rapports, CSV, graphiques).
- `LICENSE` : licence du projet.
- `.python-version` : version Python recommandée.

---

## Principaux scripts & fonctions

1. Script de mise à jour de la discrétisation
   - Fichier : `src/update_discretization.py`
   - Entrées attendues :
     - `credit_risk_dataset_post_prepocess.csv` (dataset pré-traité)
     - `discretization_boundaries_final.json` (boundaries par variable sous forme JSON)
   - Sorties :
     - `credit_risk_dataset_discretized_final.csv` (dataset avec colonnes discrétisées)
     - `RAPPORT_DISCRETISATION_FINALE.txt` (rapport détaillé par variable)
   - Actions :
     - Applique les boundaries, calcule statistiques par bin, réalise test χ² d'indépendance, génère rapport texte et affiche un résumé en console.

2. Utilitaires (fonctions importables)
   - Fichier : `src/utils.py`
   - Fonctions importantes :
     - `discretize_with_chimerge(data, var_name, target_var, max_bins=5, significance_level=0.05)`  
       Discrétisation via ChiMerge, retourne dictionnaire contenant les boundaries, bins, score χ², p-value et statistiques par bin.
     - `calculate_chi2_between_bins(...)` : calcule χ² entre deux bins adjacents.
     - `chi2_independence_tests(df, alpha)` : teste l'indépendance χ² pour paires de variables discrétisées.
     - `cramers_v(x, y)` et `cramers_v_matrix(df)` : mesure d'association Cramér's V.
     - `stepwise_bic_forward_backward(X, y, cat_vars, ...)` : sélection stepwise par groupe de modalités avec critère BIC + filtrage p-values.

---

## Prérequis & installation

- Python >= 3.12 (voir `.python-version`)
- Dépendances déclarées dans `pyproject.toml`. Extraits principaux : `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy`, `plotly`, `seaborn`, `matplotlib`, `openpyxl`, `jinja2`, `nbformat`, `ruff`.

Exemples d'installation :

1. Créer un environnement virtuel
   - python -m venv .venv
   - source .venv/bin/activate  (macOS / Linux)
   - .venv\Scripts\activate     (Windows)

2. Installer les dépendances
   - Option 1 (pip via pyproject) :
     - pip install .
   - Option 2 (installer les paquets listés) :
     - pip install numpy pandas scikit-learn statsmodels scipy plotly seaborn matplotlib openpyxl jinja2 nbformat ruff

Remarque : si vous souhaitez importer `src` comme package depuis la racine sans installer, ajoutez `PYTHONPATH=src` ou exécutez les scripts avec `python src/update_discretization.py`.

---

## Exécution — cas d'usage courant

1. Préparer les fichiers d'entrée dans la racine du projet :
   - `credit_risk_dataset_post_prepocess.csv` (dataset)
   - `discretization_boundaries_final.json` (liste de dicts { "variable": ..., "boundaries": [...], "method": ... })

2. Lancer la mise à jour de la discrétisation :
   - python src/update_discretization.py

Résultats :
- `credit_risk_dataset_discretized_final.csv`
- `RAPPORT_DISCRETISATION_FINALE.txt`

Le script affiche aussi un résumé des changements et des statistiques (Chi², p-value, nombre de bins, fusions appliquées).

---

## Exemple d'utilisation des utilitaires (mini-guide)

Importer et utiliser ChiMerge depuis un REPL ou notebook :

```python
import pandas as pd
from src.utils import discretize_with_chimerge

df = pd.read_csv("datasets/credit_risk_dataset_post_prepocess.csv")
res = discretize_with_chimerge(df, var_name="loan_amnt", target_var="loan_status", max_bins=5)
print(res["boundaries"], res["chi2_score"])
```

Stepwise + filtrage p-values (exemple) :

```python
from src.utils import stepwise_bic_forward_backward
# X : DataFrame encodée (modalités en colonnes), y : Series cible binaire
selected_vars, fitted_result, table = stepwise_bic_forward_backward(X, y, cat_vars=list_of_categorical_variables)
```

Note : la fonction stepwise s'attend à un DataFrame `X` contenant l'encodage (par ex. one-hot) avec préfixe des colonnes `variable_modalite`.

---

## Bonnes pratiques & points d'attention

- Vérifier la qualité des expected counts avant d'interpréter un test χ² (le code vérifie la condition d'attente).
- Lors de la sélection stepwise, l'algorithme opère sur des groupes de modalités (chaque variable complète est ajoutée/retraitée).
- Le filtrage final retire les variables qui ont au moins une modalité non-significative (p > alpha).
- Pour reproductibilité, conservez les fichiers `discretization_boundaries_final.json` et le dataset d'origine.

---

## Fichiers générés (par update_discretization.py)
- `credit_risk_dataset_discretized_final.csv` — dataset final avec colonnes discrétisées.
- `RAPPORT_DISCRETISATION_FINALE.txt` — rapport complet (méthodes, chi², p-values, statistiques par bin, recommandations).
- `discretization_boundaries_final.json` — boundaries finales (si généré/édité manuellement).

---

## Développement & contributions

- Si vous souhaitez intégrer ce code dans un package, considérez :
  - Ajouter un `setup.cfg` / compléter `pyproject.toml` pour rendre `src` un package installable.
  - Fournir un `requirements.txt` pour installations rapides.
  - Ajouter des notebooks d'exemples dans `notebooks/` et de la documentation dans `docs/`.
- Tests unitaires recommandés pour :
  - Comportement de ChiMerge sur petits jeux de données.
  - Robustesse des tests χ² (cas de faibles effectifs).
  - Intégration stepwise (stabilité des résultats).

---

## Licence

Voir le fichier `LICENSE` à la racine du dépôt.

---

Si vous voulez, je peux :
- Mettre à jour automatiquement le README.md dans le dépôt avec ce contenu (préparer le commit),
- Générer un `requirements.txt` à partir de `pyproject.toml`,
- Ajouter un exemple de notebook montrant la discrétisation et la sélection étape par étape.
