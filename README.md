# Credit-Risk-Project-PD-Estimation

Ce projet développe un modèle de credit scoring fondé sur une régression logistique, avec discrétisation et sélection de variables par Stepwise, afin d’estimer la probabilité de défaut et produire une grille de score et une segmentation en classes de risque.


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


## Prérequis & installation

- Python >= 3.12 (voir `.python-version`)
- Dépendances déclarées dans `pyproject.toml`. Extraits principaux : `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `scipy`, `plotly`, `seaborn`, `matplotlib`, `openpyxl`, `jinja2`, `nbformat`, `ruff`.

##### Exemples d'installation

1. **Créer un environnement virtuel avec uv**
```bash
   uv venv
```

2. **Installer les dépendances**
   
   - **Option 1 (via pyproject.toml)** :
```bash
     uv pip install .
```
   
   - **Option 2 (installer les paquets listés)** :
```bash
     uv pip install numpy pandas scikit-learn statsmodels scipy plotly seaborn matplotlib openpyxl jinja2 nbformat ruff
```

## Licence

Voir le fichier `LICENSE` à la racine du dépôt.

---

Si vous voulez, je peux :
- Mettre à jour automatiquement le README.md dans le dépôt avec ce contenu (préparer le commit),
- Générer un `requirements.txt` à partir de `pyproject.toml`,
- Ajouter un exemple de notebook montrant la discrétisation et la sélection étape par étape.
