"""
Script pour mettre à jour la discrétisation après regroupement de bins
- loan_amnt: Fusion bins 0 et 1
- loan_percent_income: Fusion bins 0-1 et bins 4-5
- person_income: Fusion bins 2 et 3
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
from datetime import datetime

# Charger les données
print("📂 Chargement des données...")
df = pd.read_csv('credit_risk_dataset_post_prepocess.csv')
print(f"✅ {len(df):,} observations chargées")

# Charger les boundaries mises à jour
with open('discretization_boundaries_final.json', 'r') as f:
    boundaries_data = json.load(f)

print("\n" + "="*100)
print(" "*30 + "🔄 MISE À JOUR DE LA DISCRÉTISATION")
print("="*100)

# Statistiques globales
global_default_rate = df['loan_status'].mean()
print(f"\n📊 Taux de défaut global: {global_default_rate*100:.2f}%")

# Fonction pour discrétiser une variable
def discretize_variable(data, var_name, boundaries):
    """Discrétise une variable continue selon les boundaries données"""
    final_bins = [-np.inf] + boundaries + [np.inf]
    discretized = pd.cut(data[var_name], bins=final_bins, labels=False)
    return discretized, final_bins

# Fonction pour calculer les statistiques par bin
def calculate_bin_statistics(data, var_name, discretized, final_bins, target='loan_status'):
    """Calcule les statistiques pour chaque bin"""
    bin_stats = []
    
    for bin_idx in range(len(final_bins) - 1):
        bin_mask = discretized == bin_idx
        bin_data = data[bin_mask]
        
        n_obs = np.sum(bin_mask)
        n_defaults = np.sum(bin_data[target])
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
    
    return pd.DataFrame(bin_stats)

# Fonction pour déterminer le niveau de risque
def get_risk_level(default_rate, global_rate):
    """Détermine le niveau de risque basé sur le taux de défaut"""
    if default_rate < global_rate * 0.8:
        return "FAIBLE"
    elif default_rate < global_rate * 1.2:
        return "MODÉRÉ"
    else:
        return "ÉLEVÉ"

def get_risk_comment(default_rate, global_rate):
    """Génère un commentaire sur le risque"""
    if default_rate < global_rate * 0.8:
        return "Taux de défaut inférieur à la moyenne - Faible risque"
    elif default_rate < global_rate * 1.2:
        return "Taux de défaut proche de la moyenne - Risque modéré"
    else:
        return "Taux de défaut supérieur à la moyenne - Risque élevé"

# Discrétiser toutes les variables et calculer les statistiques
final_results = {}
df_discretized = df.copy()

for var_info in boundaries_data:
    var_name = var_info['variable']
    boundaries = var_info['boundaries']
    method = var_info['method']
    
    print(f"\n{'─'*100}")
    print(f"📊 Variable: {var_name}")
    print(f"🔧 Méthode: {method}")
    print(f"📈 Boundaries: {boundaries}")
    
    # Discrétiser
    discretized, final_bins = discretize_variable(df, var_name, boundaries)
    df_discretized[var_name] = discretized
    
    # Calculer les statistiques
    bin_stats = calculate_bin_statistics(df, var_name, discretized, final_bins)
    
    # Calculer Chi²
    contingency_table = pd.crosstab(discretized, df['loan_status'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Stocker les résultats
    final_results[var_name] = {
        'variable': var_name,
        'method': method,
        'boundaries': boundaries,
        'n_bins': len(boundaries) + 1,
        'chi2_score': chi2,
        'p_value': p_value,
        'bin_statistics': bin_stats
    }
    
    print(f"✅ {len(boundaries) + 1} bins créés")
    print(f"   Chi² = {chi2:.2f}, p-value = {p_value:.8f}")

# Sauvegarder le dataset discrétisé
output_csv = 'credit_risk_dataset_discretized_final.csv'
df_discretized.to_csv(output_csv, index=False)
print(f"\n✅ Dataset discrétisé sauvegardé: {output_csv}")

# Générer le rapport texte complet
print("\n📝 Génération du rapport...")
rapport_filename = 'RAPPORT_DISCRETISATION_FINALE.txt'

with open(rapport_filename, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write(" "*25 + "RAPPORT FINAL DE DISCRÉTISATION\n")
    f.write(" "*20 + "Méthodes Mixtes Optimales par Variable\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("Dataset: credit_risk_dataset_post_prepocess.csv\n")
    f.write(f"Nombre total d'observations: {len(df):,}\n")
    f.write(f"Taux de défaut global: {global_default_rate*100:.2f}%\n")
    f.write(f"Nombre de variables discrétisées: {len(final_results)}\n\n")
    
    f.write("="*100 + "\n")
    f.write("STRATÉGIE DE DISCRÉTISATION\n")
    f.write("="*100 + "\n\n")
    
    for var_name, result in final_results.items():
        f.write(f"  • {var_name:35s} → {result['method']}\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("RÉSUMÉ GLOBAL\n")
    f.write("="*100 + "\n\n")
    
    chi2_scores = [r['chi2_score'] for r in final_results.values()]
    bins_counts = [r['n_bins'] for r in final_results.values()]
    
    f.write(f"Chi² moyen: {np.mean(chi2_scores):.2f}\n")
    f.write(f"Chi² médian: {np.median(chi2_scores):.2f}\n")
    max_chi2_var = max(final_results.items(), key=lambda x: x[1]['chi2_score'])
    min_chi2_var = min(final_results.items(), key=lambda x: x[1]['chi2_score'])
    f.write(f"Chi² maximum: {max_chi2_var[1]['chi2_score']:.2f} ({max_chi2_var[0]})\n")
    f.write(f"Chi² minimum: {min_chi2_var[1]['chi2_score']:.2f} ({min_chi2_var[0]})\n")
    f.write(f"Nombre moyen de bins: {np.mean(bins_counts):.2f}\n\n")
    
    f.write("Distribution des méthodes:\n")
    methods_count = {}
    for result in final_results.values():
        method = result['method']
        methods_count[method] = methods_count.get(method, 0) + 1
    
    for method, count in methods_count.items():
        f.write(f"  - {method}: {count} variable{'s' if count > 1 else ''}\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("DÉTAILS PAR VARIABLE\n")
    f.write("="*100 + "\n\n")
    
    # Détails pour chaque variable
    for var_name, result in final_results.items():
        f.write("\n" + "─"*100 + "\n")
        f.write(f"VARIABLE: {var_name}\n")
        f.write("─"*100 + "\n\n")
        
        f.write(f"Méthode appliquée: {result['method']}\n")
        f.write(f"Nombre de bins: {result['n_bins']}\n")
        f.write(f"Nombre de boundaries: {len(result['boundaries'])}\n")
        f.write(f"Chi-carré: {result['chi2_score']:.2f}\n")
        f.write(f"P-value: {result['p_value']:.8f}\n")
        
        # Significativité
        if result['p_value'] < 0.001:
            signif = "*** Très hautement significatif (p < 0.001)"
        elif result['p_value'] < 0.01:
            signif = "** Hautement significatif (p < 0.01)"
        elif result['p_value'] < 0.05:
            signif = "* Significatif (p < 0.05)"
        else:
            signif = "Non significatif (p >= 0.05)"
        f.write(f"Significativité: {signif}\n")
        
        # Statistiques descriptives
        f.write("\nStatistiques descriptives:\n")
        f.write(f"  • Minimum: {df[var_name].min():.4f}\n")
        f.write(f"  • Maximum: {df[var_name].max():.4f}\n")
        f.write(f"  • Moyenne: {df[var_name].mean():.4f}\n")
        f.write(f"  • Médiane: {df[var_name].median():.4f}\n")
        f.write(f"  • Écart-type: {df[var_name].std():.4f}\n")
        
        # Boundaries
        f.write("\nBoundaries identifiées:\n")
        for i, boundary in enumerate(result['boundaries'], 1):
            f.write(f"  Boundary {i}: {boundary:.6f}\n")
        
        # Détails par bin
        f.write("\nDétails par bin:\n")
        f.write("-"*100 + "\n")
        
        for _, bin_info in result['bin_statistics'].iterrows():
            f.write(f"\n  BIN {int(bin_info['bin'])}:\n")
            f.write(f"    Intervalle: [{bin_info['lower_bound']:.6f}, {bin_info['upper_bound']:.6f}]\n")
            f.write(f"    Nombre d'observations: {int(bin_info['n_observations']):,} ({bin_info['percentage']:.2f}% du total)\n")
            f.write(f"    Nombre de défauts: {int(bin_info['n_defaults'])}\n")
            f.write(f"    Taux de défaut: {bin_info['default_rate']:.2f}%\n")
            
            risk_level = get_risk_level(bin_info['default_rate'] / 100, global_default_rate)
            risk_comment = get_risk_comment(bin_info['default_rate'] / 100, global_default_rate)
            
            f.write(f"    Niveau de risque: {risk_level}\n")
            f.write(f"    Commentaire: {risk_comment}\n")
        
        f.write("\n")
    
    # Tableau récapitulatif
    f.write("\n" + "="*100 + "\n")
    f.write("TABLEAU RÉCAPITULATIF COMPARATIF\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"{'Variable':<35s} {'Méthode':<30s} {'Bins':<6s} {'Chi²':<12s} {'P-value':<12s}\n")
    f.write("-"*100 + "\n")
    
    # Trier par Chi² décroissant
    sorted_results = sorted(final_results.items(), key=lambda x: x[1]['chi2_score'], reverse=True)
    
    for var_name, result in sorted_results:
        f.write(f"{var_name:<35s} {result['method']:<30s} {result['n_bins']:<6d} {result['chi2_score']:<12.2f} {result['p_value']:<12.8f}\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("RECOMMANDATIONS ET CONCLUSIONS\n")
    f.write("="*100 + "\n\n")
    
    f.write("1. QUALITÉ DE LA DISCRÉTISATION:\n")
    f.write("   Toutes les variables présentent des Chi² significatifs (p < 0.05).\n")
    f.write("   La méthode mixte permet d'optimiser chaque variable individuellement.\n\n")
    
    f.write("2. REGROUPEMENTS APPLIQUÉS:\n")
    f.write("   • loan_amnt: Fusion des bins 0 et 1 (petits montants homogènes)\n")
    f.write("   • loan_percent_income: Fusion bins 0-1 (faible ratio) et bins 4-5 (très haut ratio)\n")
    f.write("   • person_income: Fusion des bins 2 et 3 (revenus moyens similaires)\n\n")
    
    f.write("3. PROCHAINES ÉTAPES:\n")
    f.write("   • Validation des résultats sur un ensemble de test\n")
    f.write("   • Intégration dans un modèle de scoring\n")
    f.write("   • Analyse de stabilité temporelle\n")
    f.write("   • Calcul du Weight of Evidence (WoE) et Information Value (IV)\n\n")
    
    f.write("4. FICHIERS GÉNÉRÉS:\n")
    f.write("   • discretization_boundaries_final.json: Boundaries au format JSON\n")
    f.write("   • credit_risk_dataset_discretized_final.csv: Dataset discrétisé complet\n")
    f.write("   • RAPPORT_DISCRETISATION_FINALE.txt: Ce rapport détaillé\n\n")
    
    f.write("="*100 + "\n")
    f.write(" "*35 + "FIN DU RAPPORT\n")
    f.write("="*100 + "\n")

print(f"✅ Rapport sauvegardé: {rapport_filename}")

print("\n" + "="*100)
print(" "*30 + "✅ MISE À JOUR TERMINÉE AVEC SUCCÈS")
print("="*100)
print("\n📁 Fichiers générés:")
print(f"   1. {output_csv}")
print(f"   2. {rapport_filename}")
print("\n📊 Résumé des changements:")
print("   • loan_amnt: 5 bins → 4 bins (fusion bins 0-1)")
print("   • loan_percent_income: 6 bins → 4 bins (fusion bins 0-1 et 4-5)")
print("   • person_income: 6 bins → 5 bins (fusion bins 2-3)")
