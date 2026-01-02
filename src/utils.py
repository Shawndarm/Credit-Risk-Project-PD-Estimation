# utils.py

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


#######################################################################################
######################################  Etape 1  ######################################
#######################################################################################

'''
def plot_default_rates(df, variables, n_cols=2):
    """
    Affiche les taux de défaut par modalité pour une liste de variables catégorielles.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant la variable 'loan_status' et les variables explicatives
    variables : list
        Liste de variables catégorielles à analyser
    n_cols : int, default=2
        Nombre de colonnes dans la grille de plots
    """

    n_vars = len(variables)
    n_rows = int(np.ceil(n_vars / n_cols))

    # Création d'une grille
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()   

    for ax, var in zip(axes, variables):
        
        # Statistiques
        stats = df.groupby(var).agg(
            effectif=('loan_status', 'count'),
            defaultrate=('loan_status', 'mean')
        )
        stats['pct_effectif'] = stats['effectif'] / len(df)

        # Barplot
        sns.barplot(
            x=stats.index, 
            y=stats['defaultrate'], 
            ax=ax
        )

        ax.set_title(f"Taux de défaut et effectifs : {var}")
        ax.set_ylabel("Taux de défaut")
        ax.set_xlabel(var)
        ax.set_ylim(0, 1)

        # Ajouter les pourcentages
        for i, (idx, row) in enumerate(stats.iterrows()):
            ax.text(
                i, row['defaultrate'] + 0.02,
                f"n={row['effectif']} ({row['pct_effectif']:.1%})",
                ha='center', fontsize=8
            )

    # Masquer les axes vides
    for ax in axes[n_vars:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
'''

#######################################################################################
######################################  Etape 2  ######################################
#######################################################################################


def calculate_chi2_between_bins(data, target, bin1_mask, bin2_mask):
    """
    Calcule le chi-carré entre deux bins adjacents.
    """
    # Créer la table de contingence
    bin1_defaults = data[bin1_mask][target].sum()
    bin1_non_defaults = bin1_mask.sum() - bin1_defaults
    
    bin2_defaults = data[bin2_mask][target].sum()
    bin2_non_defaults = bin2_mask.sum() - bin2_defaults
    
    # Table de contingence
    contingency = np.array([
        [bin1_defaults, bin1_non_defaults],
        [bin2_defaults, bin2_non_defaults]
    ])
    
    # Calculer le chi-carré
    if contingency.sum() > 0 and (contingency.sum(axis=0) > 0).all() and (contingency.sum(axis=1) > 0).all():
        chi2, _, _, _ = chi2_contingency(contingency)
        return chi2
    else:
        return np.inf


def discretize_with_chimerge(data, var_name, target_var, max_bins=5, significance_level=0.05):
    """
    Discrétise une variable en utilisant l'algorithme ChiMerge.
    
    Paramètres:
    -----------
    data : DataFrame
        Dataset
    var_name : str
        Variable à discrétiser
    target_var : str
        Variable cible
    max_bins : int
        Nombre maximum de bins souhaité
    significance_level : float
        Seuil de significativité pour arrêter les fusions
        
    Retourne:
    ---------
    dict : Résultats de la discrétisation
    """
    from scipy.stats import chi2 as chi2_dist
    
    # Trier les valeurs
    sorted_data = data[[var_name, target_var]].sort_values(var_name).reset_index(drop=True)
    
    # Initialisation: chaque valeur unique est un bin
    unique_values = sorted(data[var_name].unique())
    
    # Créer les bins initiaux (plus petits groupes possibles)
    n_initial_bins = min(50, len(unique_values))  # Limiter pour la performance
    try:
        _, initial_boundaries = pd.qcut(data[var_name], q=n_initial_bins, retbins=True, duplicates='drop')
    except:
        initial_boundaries = np.linspace(data[var_name].min(), data[var_name].max(), n_initial_bins + 1)
    
    boundaries = list(initial_boundaries[1:-1])  # Exclure min et max
    
    # Seuil chi-carré pour la significativité (ddl=1 pour 2 bins)
    chi2_threshold = chi2_dist.ppf(1 - significance_level, df=1)
    
    # Boucle de fusion
    iteration = 0
    while len(boundaries) >= max_bins - 1:
        iteration += 1
        
        # Créer les bins actuels
        current_bins = [-np.inf] + boundaries + [np.inf]
        
        # Calculer le chi-carré entre chaque paire de bins adjacents
        chi2_values = []
        
        for i in range(len(current_bins) - 2):
            # Masques pour les deux bins adjacents
            bin1_mask = (data[var_name] > current_bins[i]) & (data[var_name] <= current_bins[i+1])
            bin2_mask = (data[var_name] > current_bins[i+1]) & (data[var_name] <= current_bins[i+2])
            
            # Calculer le chi-carré
            chi2_val = calculate_chi2_between_bins(data, target_var, bin1_mask, bin2_mask)
            chi2_values.append((i, chi2_val))
        
        if not chi2_values:
            break
        
        # Trouver la paire avec le chi-carré le plus faible (plus similaires)
        min_chi2_idx, min_chi2_val = min(chi2_values, key=lambda x: x[1])
        
        # Arrêter si le chi-carré minimum dépasse le seuil
        if min_chi2_val > chi2_threshold and len(boundaries) <= max_bins:
            break
        
        # Fusionner les deux bins en supprimant la borne entre eux
        if min_chi2_idx < len(boundaries):
            boundaries.pop(min_chi2_idx)
        
        # Arrêter si on a atteint le nombre maximum de bins
        if len(boundaries) < max_bins - 1:
            break
    
    # Créer les bins finaux
    final_bins = [-np.inf] + boundaries + [np.inf]
    
    # Discrétiser
    discretized = pd.cut(data[var_name], bins=final_bins, labels=False)
    
    # Calculer le chi-carré final
    contingency_table = pd.crosstab(discretized, data[target_var])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Statistiques par bin
    bin_stats = []
    for bin_idx in range(len(final_bins) - 1):
        bin_mask = discretized == bin_idx
        bin_data = data[bin_mask]
        
        n_obs = np.sum(bin_mask)
        n_defaults = np.sum(bin_data[target_var])
        default_rate = n_defaults / n_obs if n_obs > 0 else 0
        
        lower = final_bins[bin_idx] if final_bins[bin_idx] != -np.inf else data[var_name].min()
        upper = final_bins[bin_idx + 1] if final_bins[bin_idx + 1] != np.inf else data[var_name].max()
        
        bin_stats.append({
            'bin': bin_idx,
            'lower_bound': lower,
            'upper_bound': upper,
            'n_observations': n_obs,
            'n_defaults': n_defaults,
            'default_rate': default_rate * 100,
            'percentage': (n_obs / len(data)) * 100
        })
    
    return {
        'variable': var_name,
        'method': 'ChiMerge',
        'boundaries': boundaries,
        'n_bins': len(final_bins) - 1,
        'chi2_score': chi2,
        'p_value': p_value,
        'discretized': discretized,
        'bin_statistics': pd.DataFrame(bin_stats),
        'iterations': iteration
    }
    print("✅ Fonction ChiMerge définie!")


#######################################################################################
#########################  2. Analyse de corrélations  ################################
#######################################################################################

# --- Fonction Chi2 d'indépendance pour toutes les paires de variables discrétisées ---
def chi2_independence_tests(df, alpha):
    """
    Tests d'indépendance du χ² pour toutes les paires de variables
    discrétisées d'un DataFrame.
    """
    results = []

    for i, var1 in enumerate(df.columns):
        for var2 in df.columns[i+1:]:

            contingency = pd.crosstab(df[var1], df[var2])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            # Vérification condition de validité
            expected_valid = (
                (expected >= 5).sum() / expected.size >= 0.8
                and (expected >= 1).all()
            )
            results.append({
                "Variable 1": var1,
                "Variable 2": var2,
                "Chi-2": chi2,
                "ddl": dof,
                "p_value": p_value,
                "dependance_significative": p_value < alpha,
                "chi2_valide": expected_valid
            })
    resultats = pd.DataFrame(results)
    # On affiche les paires de variables qui rejettent H0 au seuil alpha et pour lesquelles le test est valide
    return resultats[(resultats["p_value"] < alpha) & (resultats["chi2_valide"])].sort_values("Chi-2", ascending=False).reset_index(drop=True)

# --- Fonction Cramér's V pour deux variables catégorielles ---
# Fonction Cramér's V ---
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))
#  Matrice Cramér's V ---
def cramers_v_matrix(df):
    cols = df.columns
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            v = cramers_v(df[cols[i]], df[cols[j]])
            mat.iloc[i, j] = v
            mat.iloc[j, i] = v
    return mat


#######################################################################################
#########################  3. Selection des variables  ################################
#######################################################################################

def plot_default_rates(df, variables, n_cols=2):

    n_vars = len(variables)
    n_rows = int(np.ceil(n_vars / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=variables
    )
    for i, var in enumerate(variables):
        row = i // n_cols + 1
        col = i % n_cols + 1

        stats = (
            df.groupby(var)
            .agg(
                effectif=('loan_status', 'count'),
                defaultrate=('loan_status', 'mean')
            )
            .reset_index()
        )
        stats["pct_effectif"] = stats["effectif"] / len(df)
        text_labels = [
            f"{rate:.1%}<br>n={n} ({pct:.1%})"
            for rate, n, pct in zip(
                stats["defaultrate"],
                stats["effectif"],
                stats["pct_effectif"]
            )
        ]
        fig.add_trace(
            go.Bar(
                x=stats[var],
                y=stats["defaultrate"] * 100,
                text=text_labels,
                textposition="outside",
                marker=dict(color="navy")
            ),
            row=row,
            col=col
        )
        # Titre de l'axe Y uniquement pour la première colonne
        if col == 1:
            fig.update_yaxes(
                title_text="Taux de défaut (%)",
                range=[0, 100],
                ticksuffix="%",
                row=row,
                col=col
            )
        else:
            fig.update_yaxes(
                range=[0, 100],
                ticksuffix="%",
                row=row,
                col=col
            )
    fig.update_layout(
        height=350 * n_rows,
        width=1100,
        title="Taux de défaut par modalité",
        showlegend=False,
        template="simple_white"
    )
    fig.update_xaxes(type="category")
    fig.show()


### Sélection stepwise BIC + filtrage p-values ###
df = pd.DataFrame()  # <-- à remplacer par le DataFrame réel
def stepwise_bic_forward_backward(X, y, cat_vars, verbose=True, alpha=0.05, maxiter=100):
    """
    Stepwise (forward + backward) par groupe de modalités (variable complète) avec critère BIC.
    Ensuite, on retire les variables qui ont au moins une modalité non significative (p > alpha).

    """

    # utilitaires
    def make_exog_from_vars(var_list):
        """Construit la DataFrame exogène alignée sur y.index pour la liste de variables var_list."""
        if len(var_list) == 0:
            # DataFrame intercept-only aligné sur y.index
            X_tmp = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
            X_tmp = sm.add_constant(X_tmp, has_constant='add')
            # remove duplicate const if any
            X_tmp = X_tmp.loc[:, ~X_tmp.columns.duplicated()]
            return X_tmp
        else:
            cols = []
            for v in var_list:
                cols.extend([c for c in X.columns if c.startswith(v + "_")])
            # reindex pour forcer l'alignement avec y
            X_sel = X[cols].reindex(index=y.index)
            X_tmp = sm.add_constant(X_sel, has_constant='add')
            X_tmp = X_tmp.loc[:, ~X_tmp.columns.duplicated()]
            return X_tmp

    def fit_model_from_vars(var_list):
        """Retourne (result, bic) pour la liste de variables var_list (groupes complets)."""
        X_tmp = make_exog_from_vars(var_list)
        model = sm.Logit(y.reindex(X_tmp.index), X_tmp)
        res = model.fit(disp=False, maxiter=maxiter)
        return res, res.bic

    # 1) initial: modèle intercept seul
    selected = []
    result_null, current_bic = fit_model_from_vars(selected)
    if verbose:
        print(f"BIC intercept seul : {current_bic:.2f}")

    improving = True
    iteration = 0
    while improving:
        iteration += 1
        improving = False

        # ---- Forward step : essayer d'ajouter chaque variable non sélectionnée ----
        best_candidate = None
        best_bic_forward = current_bic

        candidates = [v for v in cat_vars if v not in selected]
        for cand in candidates:
            try:
                _, bic = fit_model_from_vars(selected + [cand])
            except Exception:
                # si le fit échoue pour ce candidat, on l'ignore
                continue
            if bic < best_bic_forward:
                best_bic_forward = bic
                best_candidate = cand

        # si on a trouvé un ajout qui améliore le BIC -> l'ajouter
        if best_candidate is not None:
            selected.append(best_candidate)
            current_bic = best_bic_forward
            improving = True
            if verbose:
                print(f"\nItération {iteration} - Forward : + ajout de '{best_candidate}' -> BIC = {current_bic:.2f}")

            # ---- Backward step : après ajout, tenter de retirer des variables ----
            removed = True
            while removed and len(selected) > 0:
                removed = False
                best_bic_backward = current_bic
                worst_var = None

                for cand_remove in selected:
                    try:
                        _, bic = fit_model_from_vars([v for v in selected if v != cand_remove])
                    except Exception:
                        continue
                    if bic < best_bic_backward:
                        best_bic_backward = bic
                        worst_var = cand_remove

                if worst_var is not None:
                    selected.remove(worst_var)
                    current_bic = best_bic_backward
                    removed = True
                    if verbose:
                        print(f"   Backward : - retrait de '{worst_var}' -> BIC = {current_bic:.2f}")

        else:
            # Aucun candidat apportant une amélioration du BIC
            if verbose:
                print("\nAucun ajout possible (forward) qui améliore le BIC. Fin des étapes stepwise BIC.\n")
            break

    if verbose:
        print(f"\nSélection finale par BIC (avant filtrage p-values) : {selected}")
        print(f"BIC final : {current_bic:.2f}\n")

    # 2) Filtrage : conserver seulement les variables dont TOUTES les modalités encodées sont significatives
    def fit_result_from_vars(var_list):
        """Return fitted result for var_list (raises on failure)."""
        X_tmp = make_exog_from_vars(var_list)
        return sm.Logit(y.reindex(X_tmp.index), X_tmp).fit(disp=False, maxiter=maxiter)

    # si aucune variable sélectionnée, on retourne l'intercept
    if len(selected) == 0:
        result_final = result_null
        final_selected = []
    else:
        final_selected = selected.copy()
        while True:
            result = fit_result_from_vars(final_selected)
            pvals = result.pvalues.to_dict()
            # pour chaque variable, vérifier ses modalités encodées
            vars_to_remove = []
            for v in final_selected:
                cols_v = [c for c in X.columns if c.startswith(v + "_")]
                # on vérifie uniquement les colonnes présentes dans le modèle (alignées)
                cols_in_model = [c for c in cols_v if c in pvals]
                non_sig = [c for c in cols_in_model if (pd.isna(pvals[c]) or pvals[c] > alpha)]
                if len(non_sig) > 0:
                    vars_to_remove.append((v, non_sig))

            if len(vars_to_remove) == 0:
                # toutes les modalités restantes sont significatives
                result_final = result
                break
            else:
                # on retire toutes les variables qui ont au moins une modalité non-significative
                removed_vars = [v for v, _ in vars_to_remove]
                for rv in removed_vars:
                    final_selected.remove(rv)
                if verbose:
                    print("Filtrage p-values : retrait des variables (au moins une modalité non-significative) ->", removed_vars)
                if len(final_selected) == 0:
                    # plus de variables -> garder l'intercept
                    result_final = result_null
                    break
                # sinon on refitte dans la boucle et re-vérifie

    if verbose:
        print(f"\nSélection finale après filtrage p-values (alpha={alpha}) : {final_selected}")
        if result_final is not None:
            print(f" • BIC modèle final : {getattr(result_final, 'bic', np.nan):.2f}")
            print(f" • Pseudo R² : {getattr(result_final, 'prsquared', np.nan):.4f}")
            print(f" • Nb paramètres : {len(getattr(result_final, 'params', []))}\n")

    # 3) Construire le tableau final modalité / coef / p-value / significativité
    rows = []
    # build map of coef/pval if we have a fitted result with parameters
    if result_final is not None and hasattr(result_final, "params"):
        coef_map = result_final.params.to_dict()
        pval_map = result_final.pvalues.to_dict()
    else:
        coef_map = {}
        pval_map = {}

    for var in final_selected:
        modalities = sorted(df[var].dropna().unique())
        for m in modalities:
            col_name = f"{var}_{m}"
            # --- IMPORTANT: afficher le vrai nom de la modalité (str(m)) dans la colonne "Modalité"
            if col_name in coef_map:
                coef = coef_map[col_name]
                pval = pval_map.get(col_name, np.nan)
                modalite_label = f"{var}_{m}"
            else:
                # modalité de référence (absente de l'encodage)
                coef = 0.0
                pval = "_"   # <-- remplacer NaN par "_" demandé
                modalite_label = f"{var}_{m} (Référence)"

            rows.append({
                "Variable": var,
                "Modalité": modalite_label,
                "Coefficient": coef,
                "p-value": pval
            })

    table_stepwise = pd.DataFrame(rows)

    # fonction d'étoiles (robuste si p n'est pas numérique)
    def signif_stars(p):
        try:
            pnum = float(p)
        except Exception:
            return "_"
        if pd.isna(pnum):
            return "_"
        if pnum <= 0.001:
            return "***"
        if pnum <= 0.01:
            return "**"
        if pnum <= 0.05:
            return "*"
        if pnum <= 0.1:
            return "."
        return "_"

    if not table_stepwise.empty:
        table_stepwise["Significativité"] = table_stepwise["p-value"].apply(signif_stars)

    return final_selected, result_final, table_stepwise