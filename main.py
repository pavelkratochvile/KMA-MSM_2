from pathlib import Path
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA
import scikit_posthocs as sp
import pingouin as pg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class WineCultivarSet:
    def __init__(self, cultivar_id:int, filename:str):
        self.cultivar_id = cultivar_id
        self.alcohol = []
        self.malic_acid = []
        self.ash = []
        self.ash_alcalinity = []
        self.magnesium = []
        self.phenols_total = []
        self.flavanoids = []
        self.nonflavanoid_phenols = []
        self.proanthocyanins = []
        self.color_intensity = []
        self.hue = []
        self.OD280_OD315_of_diluted_wines = []
        self.proline = []

        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip(): 
                    continue

                line_data = line.strip().split(";")

                if int(line_data[0]) != self.cultivar_id:
                    continue
                
                self.alcohol.append(float(line_data[1]))
                self.malic_acid.append(float(line_data[2]))
                self.ash.append(float(line_data[3]))
                self.ash_alcalinity.append(float(line_data[4]))
                self.magnesium.append(float(line_data[5]))
                self.phenols_total.append(float(line_data[6]))
                self.flavanoids.append(float(line_data[7]))
                self.nonflavanoid_phenols.append(float(line_data[8]))
                self.proanthocyanins.append(float(line_data[9]))
                self.color_intensity.append(float(line_data[10]))
                self.hue.append(float(line_data[11]))
                self.OD280_OD315_of_diluted_wines.append(float(line_data[12]))
                self.proline.append(float(line_data[13]))

    def calculate_average(self, average_list):
        return sum(average_list) / len(average_list)

def calculate_average_of_all(average_list1, average_list2, average_list3):
    list_of_all = average_list1 + average_list2 + average_list3
    return sum(list_of_all) / len(list_of_all)

# Funkce pro jednotlivé výpisy výsledků testů a nadpisy kapitol

def print_main_chapter(chapter_title):
    """Vytvoří hlavní nadpis kapitoly (pro každý z bodů projektu)."""
    print("\n" + "#" * 85)
    print(f"#{chapter_title.center(83)}#")
    print("#" * 85 + "\n")

def print_sub_chapter(sub_title):
    """Vytvoří podnadpis uvnitř kapitoly."""
    print(f"\n--- [ {sub_title.upper()} ] " + "-" * (70 - len(sub_title)))

def print_results_one_sample_test(cultivar_name, variable_name, t_stat, p_val_t, w_stat, p_val_w, h0_text, is_normal, alpha=0.05):
    print(f"\n   🍷 Vzorek: {cultivar_name} | Proměnná: {variable_name}")
    print(f"   • H0: {h0_text}")
    
    # parametrický t-test (vyžaduje normalitu)
    t_result_str = f"t = {t_stat:.4f} | p = {p_val_t:.6f} ➔ {'ZAMÍTÁME H0' if p_val_t < alpha else 'NEZAMÍTÁME H0'}"
    if not is_normal:
        t_result_str += " ⚠️ [POZOR: Data nemají normální rozdělení, t-test není spolehlivý!]"
    print(f"   • [1] t-test    : {t_result_str}")
    
    # Neparametrický Wilcoxon (nevyžaduje normalitu, doporučený test)
    w_result_str = f"W = {w_stat:.4f} | p = {p_val_w:.6f} ➔ {'ZAMÍTÁME H0' if p_val_w < alpha else 'NEZAMÍTÁME H0'}"
    if not is_normal:
        w_result_str += " ✅ (Doporučeno: robustní vůči nenormalitě)"
    print(f"   • [2] Wilcoxon  : {w_result_str}")
    
    print("   " + "—" * 70)

def print_results_two_sample_test(cultivar1_name, cultivar2_name, variable_name, t_stat, p_val_t, w_stat, p_val_w, h0_text, is_sample_1_normal, is_sample_2_normal, alpha=0.05):
    print(f"\n   🍷 Vzorek: {cultivar1_name} vs {cultivar2_name} | Proměnná: {variable_name}")
    print(f"   • H0: {h0_text}")

    # Parametrický t-test (vyžaduje normalitu)
    t_result_str = f"t = {t_stat:.4f} | p = {p_val_t:.6f} ➔ {'ZAMÍTÁME H0' if p_val_t < alpha else 'NEZAMÍTÁME H0'}"
    if not is_sample_1_normal and not is_sample_2_normal:
        t_result_str += " ⚠️ [POZOR: Oba vzorky nemají normální rozdělení, t-test není spolehlivý!]"
    elif not is_sample_1_normal:
        t_result_str += f" ⚠️ [POZOR: Vzorek {cultivar1_name} nemá normální rozdělení, t-test není spolehlivý!]"
    elif not is_sample_2_normal:    
        t_result_str += f" ⚠️ [POZOR: Vzorek {cultivar2_name} nemá normální rozdělení, t-test není spolehlivý!]"

    print(f"   • [1] t-test    : {t_result_str}")
    
    # Neparametrický Wilcoxon (nevyžaduje normalitu, doporučený test)
    w_result_str = f"U = {w_stat:.4f} | p = {p_val_w:.6f} ➔ {'ZAMÍTÁME H0' if p_val_w < alpha else 'NEZAMÍTÁME H0'}"
    print(f"   • [2] Mann-Whitney : {w_result_str}")

def print_results_anova_kruskal_test(variable_name, t_stat, p_val_t, w_stat, p_val_w, is_sample_1_normal, is_sample_2_normal, is_sample_3_normal, h0_text, alpha=0.05):
    print(f"\n   🍷 Proměnná: {variable_name}")
    print(f"   • H0: {h0_text}")

    # Parametrický ANOVA test (vyžaduje normalitu)
    t_result_str = f"F = {t_stat:.4f} | p = {p_val_t:.6f} ➔ {'ZAMÍTÁME H0' if p_val_t < alpha else 'NEZAMÍTÁME H0'}"
    if not is_sample_1_normal and not is_sample_2_normal and not is_sample_3_normal:
        t_result_str += " ⚠️ [POZOR: Všechny vzorky nemají normální rozdělení, ANOVA není spolehlivá!]"
    elif not is_sample_1_normal:
        t_result_str += " ⚠️ [POZOR: Vzorek 1 nemá normální rozdělení, ANOVA není spolehlivá!]"
    elif not is_sample_2_normal:
        t_result_str += " ⚠️ [POZOR: Vzorek 2 nemá normální rozdělení, ANOVA není spolehlivá!]"
    elif not is_sample_3_normal:
        t_result_str += " ⚠️ [POZOR: Vzorek 3 nemá normální rozdělení, ANOVA není spolehlivá!]"

    print(f"   • [1] ANOVA     : {t_result_str}")

    # Neparametrický Kruskal-Wallis test (nevyžaduje normalitu, doporučený test)
    w_result_str = f"H = {w_stat:.4f} | p = {p_val_w:.6f} ➔ {'ZAMÍTÁME H0' if p_val_w < alpha else 'NEZAMÍTÁME H0'}"
    print(f"   • [2] Kruskal-W : {w_result_str}")

def print_result_normality_test(cultivar_name, variable_name, stat, p_value, h0_text, alpha=0.05):
    print(f"\n   🍷 Vzorek: {cultivar_name} | Proměnná: {variable_name}")
    print(f"   • H0: {h0_text}")
    print(f"   • Shapiro-Wilk : W = {stat:.4f} | p = {p_value:.6f} ➔ {'ZAMÍTÁME H0 (není normalita)' if p_value < alpha else 'NEZAMÍTÁME H0 (je normalita)'}")
    print("   " + "—" * 70)

def print_results_manova(manova_results, h0_text, alpha=0.05):
    """Vytáhne p-hodnotu z MANOVY (podle Pillaiovy stopy) a vypíše výsledek."""
    print(f"\n   🌍 Globální test: MANOVA")
    print(f"   • H0: {h0_text}")
    
    # manova výpis výsledku
    p_val = manova_results.results['Odrůda']['stat'].loc["Pillai's trace", "Pr > F"]
    
    result_str = f"p = {p_val:.6f} ➔ {'ZAMÍTÁME H0' if p_val < alpha else 'NEZAMÍTÁME H0'}"
    print(f"   • Pillai's trace p-hodnota : {result_str}")
    print("   " + "—" * 70)


def print_results_hotelling(cultivar1_name, cultivar2_name, test_results, h0_text, alpha=0.05):
    """Vytáhne p-hodnotu z Hotellingova párového testu (pingouin)."""
    print(f"\n   🍷 Vzorek: {cultivar1_name} vs {cultivar2_name}")
    print(f"   • H0: {h0_text}")
    
    # hotelling výpis výsledku
    p_val = test_results.loc['hotelling', 'pval']
    
    result_str = f"p = {p_val:.2e} ➔ {'ZAMÍTÁME H0' if p_val < alpha else 'NEZAMÍTÁME H0'}"
    print(f"   • Hotellingův T² test      : {result_str}")


def print_results_box_m(box_results, h0_text, alpha=0.05):
    """Vytáhne p-hodnotu z Boxova M-testu shody kovariančních matic (pingouin)."""
    print(f"\n   📦 Diagnostika: Boxův M-test (Shoda kovariančních matic)")
    print(f"   • H0: {h0_text}")
    
    p_val = box_results.loc['box', 'pval']
    
    result_str = f"p = {p_val:.6f} ➔ {'ZAMÍTÁME H0 (nejsou shodné)' if p_val < alpha else 'NEZAMÍTÁME H0 (jsou shodné)'}"
    print(f"   • Box's M p-hodnota        : {result_str}")
    print("   " + "—" * 70)


def print_results_multivariate_normality(cultivar_name, norm_results, h0_text, alpha=0.05):
    """Vytáhne p-hodnotu z Henze-Zirkler testu vícerozměrné normality (pingouin)."""
    print(f"\n   🔔 Vzorek: {cultivar_name} | Diagnostika: Vícerozměrná normalita")
    print(f"   • H0: {h0_text}")
    
    p_val = norm_results.pval
    
    result_str = f"p = {p_val:.6f} ➔ {'ZAMÍTÁME H0 (není normální)' if p_val < alpha else 'NEZAMÍTÁME H0 (je normální)'}"
    print(f"   • Henze-Zirkler p-hodnota  : {result_str}")

def print_results_posthoc_tukey(variable_name, tukey_results, alpha=0.05):
    """Rozebere tabulku z Tukeyho testu a úhledně ji vypíše."""
    print(f"\n   🔍 [Post-hoc analýza: {variable_name.upper()} - Tukeyho HSD test]")
    
    data = tukey_results.summary().data[1:]
    for row in data:
        g1, g2, meandiff, p_adj, lower, upper, reject = row
        result_str = f"p = {p_adj:.6f} ➔ {'ZAMÍTÁME H0' if p_adj < alpha else 'NEZAMÍTÁME H0'}"
        print(f"   • {g1} vs {g2}".ljust(30) + f": {result_str}")
    print("   " + "—" * 70)


def print_results_posthoc_dunn(variable_name, dunn_results, alpha=0.05):
    """Rozebere korelační matici z Dunnova testu a úhledně ji vypíše pro všechny páry."""
    print(f"\n   🔍 [Post-hoc analýza: {variable_name.upper()} - Dunnův test (Bonferroni)]")
    groups = dunn_results.columns.tolist()
    
    # posthoc pro všechny dvojice
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            p_adj = dunn_results.loc[g1, g2]
            
            result_str = f"p = {p_adj:.6f} ➔ {'ZAMÍTÁME H0' if p_adj < alpha else 'NEZAMÍTÁME H0'}"
            print(f"   • {g1} vs {g2}".ljust(30) + f": {result_str}")
    print("   " + "—" * 70)


if __name__ == "__main__":
    file_path = Path("wine-dataset.csv")
    if not file_path.exists(): 
        exit()

    wines_barolo = WineCultivarSet(1, file_path)
    wines_grignolino = WineCultivarSet(2, file_path)
    wines_barbera = WineCultivarSet(3, file_path)

    alpha = 0.05

    # =========================================================================
    # BOD 2: TESTY NORMALITY A JEDNOVÝBĚROVÉ TESTY O POLOZE
    # =========================================================================
    
    print_main_chapter("BOD 2: TESTY NORMALITY A JEDNOVÝBĚROVÉ TESTY O POLOZE")

    # testy normality (Shapiro-Wilk) pro jednotlivé proměnné a odrůdy
    print_sub_chapter("1.1 Výsledky testů normality (Shapiro-Wilk)")
    
    stat_borolo_alc_norm, p_borolo_alc_norm = stats.shapiro(wines_barolo.alcohol)
    stat_borolo_ma_norm, p_borolo_ma_norm = stats.shapiro(wines_barolo.malic_acid)
    stat_grig_alc_norm, p_grig_alc_norm = stats.shapiro(wines_grignolino.alcohol)
    stat_grig_ma_norm, p_grig_ma_norm = stats.shapiro(wines_grignolino.malic_acid)
    stat_barb_alc_norm, p_barb_alc_norm = stats.shapiro(wines_barbera.alcohol)
    stat_barb_ma_norm, p_barb_ma_norm = stats.shapiro(wines_barbera.malic_acid)

    borolo_alc_normal = p_borolo_alc_norm > alpha
    borolo_ma_normal = p_borolo_ma_norm > alpha
    grignolino_alc_normal = p_grig_alc_norm > alpha
    grignolino_ma_normal = p_grig_ma_norm > alpha
    barbera_alc_normal = p_barb_alc_norm > alpha
    barbera_ma_normal = p_barb_ma_norm > alpha

    h0_norm = "Data jsou normálně rozložena."
    h1_norm = "Data nejsou normálně rozložena."
    print_result_normality_test("Barolo", "Alkohol", stat_borolo_alc_norm, p_borolo_alc_norm, h0_norm, alpha)
    print_result_normality_test("Barolo", "Malická kyselina", stat_borolo_ma_norm, p_borolo_ma_norm, h0_norm, alpha)
    print_result_normality_test("Grignolino", "Alkohol", stat_grig_alc_norm, p_grig_alc_norm, h0_norm, alpha)
    print_result_normality_test("Grignolino", "Malická kyselina", stat_grig_ma_norm, p_grig_ma_norm, h0_norm, alpha)
    print_result_normality_test("Barbera", "Alkohol", stat_barb_alc_norm, p_barb_alc_norm, h0_norm, alpha)
    print_result_normality_test("Barbera", "Malická kyselina", stat_barb_ma_norm, p_barb_ma_norm, h0_norm, alpha)


    # Vizualizace Q-Q plotů pro jednotlivé odrůdy a proměnné (Alkohol a Malická kyselina)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    cultivars_data = [
        ("Barolo", wines_barolo.alcohol, wines_barolo.malic_acid),
        ("Grignolino", wines_grignolino.alcohol, wines_grignolino.malic_acid),
        ("Barbera", wines_barbera.alcohol, wines_barbera.malic_acid)
    ]

    for i, (name, alc, ma) in enumerate(cultivars_data):
        variables = [("Alkohol", alc), ("Malická kyselina", ma)]
        
        for j, (var_name, data) in enumerate(variables):
            ax = axes[i, j]
            
            # Q-Q plot
            osm, osr = stats.probplot(data, dist="norm", plot=None)[0]
            ax.scatter(osm, osr.data, color="#2c3e50", alpha=0.75, s=35, edgecolor="none")
            
            reg_result = stats.linregress(osm, osr.data)
            slope, intercept = reg_result.slope, reg_result.intercept
            line_x = np.array([min(osm), max(osm)])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color="#e74c3c", linewidth=2, linestyle="--", alpha=0.8)
            
            # Mini histogram v rohu grafu
            ax_inset = ax.inset_axes([0.12, 0.58, 0.30, 0.28])
            ax_inset.hist(data, bins=7, color="#3498db", alpha=0.5, edgecolor="none", density=True)
            mean_val = np.mean(data)
            ax_inset.axvline(mean_val, color="#e74c3c", linestyle="-", linewidth=1.5, label="Průměr")
            
            ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_inset.grid(False)
            sns.despine(ax=ax_inset, top=True, right=True, left=True, bottom=True)
            ax_inset.set_facecolor("white")
            ax_inset.patch.set_alpha(0.85)
            
            # Popisky hlavního grafu
            ax.set_title(f"{name} — {var_name}", fontsize=11, fontweight="bold", pad=15, color="#2c3e50")
            ax.set_xlabel("Teoretické kvantily", fontsize=9, color="#7f8c8d", labelpad=8)
            ax.set_ylabel("Naměřené hodnoty", fontsize=9, color="#7f8c8d", labelpad=8)
            
            ax.grid(True, linestyle=":", alpha=0.5, color="#bdc3c7")
            sns.despine(ax=ax, top=True, right=True)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.07, hspace=0.45, wspace=0.30)
    plt.show()

    # Jednovýběrové testy o poloze (proti celkovému průměru) pro jednotlivé odrůdy a proměnné
    print_sub_chapter("1.2 Jednovýběrové testy o poloze (proti celkovému průměru)")

    mu_0_alc = calculate_average_of_all(wines_barolo.alcohol, wines_grignolino.alcohol, wines_barbera.alcohol)
    mu_0_ma = calculate_average_of_all(wines_barolo.malic_acid, wines_grignolino.malic_acid, wines_barbera.malic_acid)

    # Výpočet testů - Alkohol
    t_bor_alc, p_t_bor_alc = stats.ttest_1samp(wines_barolo.alcohol, mu_0_alc)
    w_bor_alc, p_w_bor_alc = stats.wilcoxon(np.array(wines_barolo.alcohol) - mu_0_alc)
    t_gri_alc, p_t_gri_alc = stats.ttest_1samp(wines_grignolino.alcohol, mu_0_alc)
    w_gri_alc, p_w_gri_alc = stats.wilcoxon(np.array(wines_grignolino.alcohol) - mu_0_alc)
    t_bar_alc, p_t_bar_alc = stats.ttest_1samp(wines_barbera.alcohol, mu_0_alc)
    w_bar_alc, p_w_bar_alc = stats.wilcoxon(np.array(wines_barbera.alcohol) - mu_0_alc)

    # Výpočet testů - Malická kyselina
    t_bor_ma, p_t_bor_ma = stats.ttest_1samp(wines_barolo.malic_acid, mu_0_ma)
    w_bor_ma, p_w_bor_ma = stats.wilcoxon(np.array(wines_barolo.malic_acid) - mu_0_ma)
    t_gri_ma, p_t_gri_ma = stats.ttest_1samp(wines_grignolino.malic_acid, mu_0_ma)
    w_gri_ma, p_w_gri_ma = stats.wilcoxon(np.array(wines_grignolino.malic_acid) - mu_0_ma)
    t_bar_ma, p_t_bar_ma = stats.ttest_1samp(wines_barbera.malic_acid, mu_0_ma)
    w_bar_ma, p_w_bar_ma = stats.wilcoxon(np.array(wines_barbera.malic_acid) - mu_0_ma)

    h0_alc = "Průměrný obsah alkoholu je roven celkovému průměru."
    h1_alc = "Průměrný obsah alkoholu se liší od celkového průměru."
    h0_ma = "Průměrný obsah malické kyseliny je roven celkovému průměru."
    h1_ma = "Průměrný obsah malické kyseliny se liší od celkového průměru."

    print_results_one_sample_test("Barolo", "Alkohol", t_bor_alc, p_t_bor_alc, w_bor_alc, p_w_bor_alc, h0_alc, borolo_alc_normal, alpha)
    print_results_one_sample_test("Grignolino", "Alkohol", t_gri_alc, p_t_gri_alc, w_gri_alc, p_w_gri_alc, h0_alc, grignolino_alc_normal, alpha)
    print_results_one_sample_test("Barbera", "Alkohol", t_bar_alc, p_t_bar_alc, w_bar_alc, p_w_bar_alc, h0_alc, barbera_alc_normal, alpha)

    print_results_one_sample_test("Barolo", "Malická kyselina", t_bor_ma, p_t_bor_ma, w_bor_ma, p_w_bor_ma, h0_ma, borolo_ma_normal, alpha)
    print_results_one_sample_test("Grignolino", "Malická kyselina", t_gri_ma, p_t_gri_ma, w_gri_ma, p_w_gri_ma, h0_ma, grignolino_ma_normal, alpha)
    print_results_one_sample_test("Barbera", "Malická kyselina", t_bar_ma, p_t_bar_ma, w_bar_ma, p_w_bar_ma, h0_ma, barbera_ma_normal, alpha)


    # =========================================================================
    # BOD 3: DVOJOVÝBĚROVÉ TESTY O POLOZE
    # =========================================================================

    print_main_chapter("BOD 3: DVOJOVÝBĚROVÉ TESTY O POLOZE")
    # Testujeme vždy jednotlivé spojité proměnné mezi jednotlivými odrůdami vína.
    # Borolo vs Grignolino (alkohol, malická kyselina)
    # Barolo vs Barbera (alkohol, malická kyselina)
    # Grignolino vs Barbera (alkohol, malická kyselina)

    h0_alc = "Průměrný obsah alkoholu je stejný u obou odrůd."
    h1_alc = "Průměrný obsah alkoholu se liší mezi odrůdami."
    h0_ma = "Průměrný obsah malické kyseliny je stejný u obou odrůd."
    h1_ma = "Průměrný obsah malické kyseliny se liší mezi odrůdami."

    t_stat_bor_gri_alc, p_t_bor_gri_alc = stats.ttest_ind(wines_barolo.alcohol, wines_grignolino.alcohol, equal_var=False)
    t_stat_bor_gri_ma, p_t_bor_gri_ma = stats.ttest_ind(wines_barolo.malic_acid, wines_grignolino.malic_acid, equal_var=False)
    t_stat_bor_bar_alc, p_t_bor_bar_alc = stats.ttest_ind(wines_barolo.alcohol, wines_barbera.alcohol, equal_var=False)
    t_stat_bor_bar_ma, p_t_bor_bar_ma = stats.ttest_ind(wines_barolo.malic_acid, wines_barbera.malic_acid, equal_var=False)
    t_stat_gri_bar_alc, p_t_gri_bar_alc = stats.ttest_ind(wines_grignolino.alcohol, wines_barbera.alcohol, equal_var=False)
    t_stat_gri_bar_ma, p_t_gri_bar_ma = stats.ttest_ind(wines_grignolino.malic_acid, wines_barbera.malic_acid, equal_var=False)

    w_stat_bor_gri_alc, p_w_bor_gri_alc = stats.mannwhitneyu(wines_barolo.alcohol, wines_grignolino.alcohol, alternative='two-sided')
    w_stat_bor_gri_ma, p_w_bor_gri_ma = stats.mannwhitneyu(wines_barolo.malic_acid, wines_grignolino.malic_acid, alternative='two-sided')
    w_stat_bor_bar_alc, p_w_bor_bar_alc = stats.mannwhitneyu(wines_barolo.alcohol, wines_barbera.alcohol, alternative='two-sided')
    w_stat_bor_bar_ma, p_w_bor_bar_ma = stats.mannwhitneyu(wines_barolo.malic_acid, wines_barbera.malic_acid, alternative='two-sided')
    w_stat_gri_bar_alc, p_w_gri_bar_alc = stats.mannwhitneyu(wines_grignolino.alcohol, wines_barbera.alcohol, alternative='two-sided')
    w_stat_gri_bar_ma, p_w_gri_bar_ma = stats.mannwhitneyu(wines_grignolino.malic_acid, wines_barbera.malic_acid, alternative='two-sided')

    print_sub_chapter("2.1 Dvouvýběrové testy o poloze (mezi sebou)")

    print_results_two_sample_test("Barolo", "Grignolino", "Alkohol", t_stat_bor_gri_alc, p_t_bor_gri_alc, w_stat_bor_gri_alc, p_w_bor_gri_alc, h0_alc, borolo_alc_normal, grignolino_alc_normal, alpha)
    print_results_two_sample_test("Barolo", "Grignolino", "Malická kyselina", t_stat_bor_gri_ma, p_t_bor_gri_ma, w_stat_bor_gri_ma, p_w_bor_gri_ma, h0_ma, borolo_ma_normal, grignolino_ma_normal, alpha)
    print_results_two_sample_test("Barolo", "Barbera", "Alkohol", t_stat_bor_bar_alc, p_t_bor_bar_alc, w_stat_bor_bar_alc, p_w_bor_bar_alc, h0_alc, borolo_alc_normal, barbera_alc_normal, alpha)
    print_results_two_sample_test("Barolo", "Barbera", "Malická kyselina", t_stat_bor_bar_ma, p_t_bor_bar_ma, w_stat_bor_bar_ma, p_w_bor_bar_ma, h0_ma, borolo_ma_normal, barbera_ma_normal, alpha)
    print_results_two_sample_test("Grignolino", "Barbera", "Alkohol", t_stat_gri_bar_alc, p_t_gri_bar_alc, w_stat_gri_bar_alc, p_w_gri_bar_alc, h0_alc, grignolino_alc_normal, barbera_alc_normal, alpha)
    print_results_two_sample_test("Grignolino", "Barbera", "Malická kyselina", t_stat_gri_bar_ma, p_t_gri_bar_ma, w_stat_gri_bar_ma, p_w_gri_bar_ma, h0_ma, grignolino_ma_normal, barbera_ma_normal, alpha)

    # Vizualizace boxplotů pro jednotlivé odrůdy a proměnné (Alkohol a Malická kyselina)

    sns.set_theme(style="white")
    data_list_boxplot = []
    for i in range(len(wines_barolo.alcohol)):
        data_list_boxplot.append({"Odrůda": "Barolo", "Alkohol": wines_barolo.alcohol[i], "Malická_kyselina": wines_barolo.malic_acid[i]})
    for i in range(len(wines_grignolino.alcohol)):
        data_list_boxplot.append({"Odrůda": "Grignolino", "Alkohol": wines_grignolino.alcohol[i], "Malická_kyselina": wines_grignolino.malic_acid[i]})
    for i in range(len(wines_barbera.alcohol)):
        data_list_boxplot.append({"Odrůda": "Barbera", "Alkohol": wines_barbera.alcohol[i], "Malická_kyselina": wines_barbera.malic_acid[i]})
    
    df_wine = pd.DataFrame(data_list_boxplot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot pro Alkohol
    sns.boxplot(x="Odrůda", y="Alkohol", data=df_wine, hue="Odrůda", palette="Set2", width=0.4, linewidth=1.5, boxprops=dict(alpha=0.8), legend=False, ax=axes[0])
    sns.stripplot(x="Odrůda", y="Alkohol", data=df_wine, hue="Odrůda", palette="Set2", alpha=0.6, size=5, jitter=0.2, legend=False, ax=axes[0])
    axes[0].set_title("Porovnání obsahu alkoholu", fontsize=12, fontweight="bold", pad=15, color="#2c3e50")
    axes[0].set_xlabel("Odrůda vína", fontsize=10, color="#7f8c8d", labelpad=8)
    axes[0].set_ylabel("Obsah alkoholu", fontsize=10, color="#7f8c8d", labelpad=8)
    axes[0].grid(True, linestyle=":", alpha=0.5, color="#bdc3c7")

    # Boxplot pro Malickou kyselinu
    sns.boxplot(x="Odrůda", y="Malická_kyselina", data=df_wine, hue="Odrůda", palette="Set2", width=0.4, linewidth=1.5, boxprops=dict(alpha=0.8), legend=False, ax=axes[1])
    sns.stripplot(x="Odrůda", y="Malická_kyselina", data=df_wine, hue="Odrůda", palette="Set2", alpha=0.6, size=5, jitter=0.2, legend=False, ax=axes[1])
    axes[1].set_title("Porovnání obsahu malické kyseliny", fontsize=12, fontweight="bold", pad=15, color="#2c3e50")
    axes[1].set_xlabel("Odrůda vína", fontsize=10, color="#7f8c8d", labelpad=8)
    axes[1].set_ylabel("Obsah malické kyseliny", fontsize=10, color="#7f8c8d", labelpad=8)
    axes[1].grid(True, linestyle=":", alpha=0.5, color="#bdc3c7")

    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()

    cultivar_names = ["Barolo", "Grignolino", "Barbera"]
    
    # Matice pro t-test
    p_matrix_t_alc = np.array([
        [1.0000, p_t_bor_gri_alc, p_t_bor_bar_alc],
        [p_t_bor_gri_alc, 1.0000, p_t_gri_bar_alc],
        [p_t_bor_bar_alc, p_t_gri_bar_alc, 1.0000]
    ])
    p_matrix_t_ma = np.array([
        [1.0000, p_t_bor_gri_ma, p_t_bor_bar_ma],
        [p_t_bor_gri_ma, 1.0000, p_t_gri_bar_ma],
        [p_t_bor_bar_ma, p_t_gri_bar_ma, 1.0000]
    ])
    
    df_t_alc = pd.DataFrame(p_matrix_t_alc, index=cultivar_names, columns=cultivar_names)
    df_t_ma = pd.DataFrame(p_matrix_t_ma, index=cultivar_names, columns=cultivar_names)

    # Matice pro Mann-Whitney
    p_matrix_w_alc = np.array([
        [1.0000, p_w_bor_gri_alc, p_w_bor_bar_alc],
        [p_w_bor_gri_alc, 1.0000, p_w_gri_bar_alc],
        [p_w_bor_bar_alc, p_w_gri_bar_alc, 1.0000]
    ])
    p_matrix_w_ma = np.array([
        [1.0000, p_w_bor_gri_ma, p_w_bor_bar_ma],
        [p_w_bor_gri_ma, 1.0000, p_w_gri_bar_ma],
        [p_w_bor_bar_ma, p_w_gri_bar_ma, 1.0000]
    ])

    df_w_alc = pd.DataFrame(p_matrix_w_alc, index=cultivar_names, columns=cultivar_names)
    df_w_ma = pd.DataFrame(p_matrix_w_ma, index=cultivar_names, columns=cultivar_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Srovnání p-hodnot (t-test vs. Mann-Whitney)", fontsize=16, fontweight="bold", y=0.98, color="#2c3e50")

    sns.heatmap(df_t_alc, annot=True, fmt=".4f", cmap="YlOrRd_r", cbar_kws={'label': 'P-hodnota'}, linewidths=1, linecolor='white', vmin=0, vmax=0.1, ax=axes[0, 0], annot_kws={"size": 10})
    axes[0, 0].set_title("ALKOHOL: t-test", fontsize=12, fontweight="bold", pad=10, color="#2c3e50")

    sns.heatmap(df_t_ma, annot=True, fmt=".4f", cmap="YlOrRd_r", cbar_kws={'label': 'P-hodnota'}, linewidths=1, linecolor='white', vmin=0, vmax=0.1, ax=axes[0, 1], annot_kws={"size": 10})
    axes[0, 1].set_title("MALICKÁ KYSELINA: t-test", fontsize=12, fontweight="bold", pad=10, color="#2c3e50")

    sns.heatmap(df_w_alc, annot=True, fmt=".4f", cmap="YlOrRd_r", cbar_kws={'label': 'P-hodnota'}, linewidths=1, linecolor='white', vmin=0, vmax=0.1, ax=axes[1, 0], annot_kws={"size": 10})
    axes[1, 0].set_title("ALKOHOL: Mann-Whitney", fontsize=12, fontweight="bold", pad=10, color="#2c3e50")

    sns.heatmap(df_w_ma, annot=True, fmt=".4f", cmap="YlOrRd_r", cbar_kws={'label': 'P-hodnota'}, linewidths=1, linecolor='white', vmin=0, vmax=0.1, ax=axes[1, 1], annot_kws={"size": 10})
    axes[1, 1].set_title("MALICKÁ KYSELINA: Mann-Whitney", fontsize=12, fontweight="bold", pad=10, color="#2c3e50")

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # BOD 4: VÍCEVÝBĚROVÉ TESTY O POLOZE
    # =========================================================================

    print_main_chapter("BOD 4: VÍCEVÝBĚROVÉ TESTY O POLOZE")

    h0_all_alc = "Průměrný obsah alkoholu je stejný u všech odrůd."
    h1_all_alc = "Průměrný obsah alkoholu se liší mezi odrůdami."
    h0_all_ma = "Průměrný obsah malické kyseliny je stejný u všech odrůd."
    h1_all_ma = "Průměrný obsah malické kyseliny se liší mezi odrůdami."

    t_stat_all_alc, p_t_all_alc = stats.f_oneway(wines_barolo.alcohol, wines_grignolino.alcohol, wines_barbera.alcohol)
    t_stat_all_ma, p_t_all_ma = stats.f_oneway(wines_barolo.malic_acid, wines_grignolino.malic_acid, wines_barbera.malic_acid)
    w_stat_all_alc, p_w_all_alc = stats.kruskal(wines_barolo.alcohol, wines_grignolino.alcohol, wines_barbera.alcohol)
    w_stat_all_ma, p_w_all_ma = stats.kruskal(wines_barolo.malic_acid, wines_grignolino.malic_acid, wines_barbera.malic_acid)

    print_results_anova_kruskal_test("Alkohol", t_stat_all_alc, p_t_all_alc, w_stat_all_alc, p_w_all_alc, borolo_alc_normal, grignolino_alc_normal, barbera_alc_normal, h0_all_alc, alpha)
    print_results_anova_kruskal_test("Malická kyselina", t_stat_all_ma, p_t_all_ma, w_stat_all_ma, p_w_all_ma, borolo_ma_normal, grignolino_ma_normal, barbera_ma_normal, h0_all_ma, alpha)

    df_alc = pd.DataFrame([
    {"Odrůda": "Barolo", "Hodnota": v} for v in wines_barolo.alcohol
    ] + [
        {"Odrůda": "Grignolino", "Hodnota": v} for v in wines_grignolino.alcohol
    ] + [
        {"Odrůda": "Barbera", "Hodnota": v} for v in wines_barbera.alcohol
    ])

    df_ma = pd.DataFrame([
        {"Odrůda": "Barolo", "Hodnota": v} for v in wines_barolo.malic_acid
    ] + [
        {"Odrůda": "Grignolino", "Hodnota": v} for v in wines_grignolino.malic_acid
    ] + [
        {"Odrůda": "Barbera", "Hodnota": v} for v in wines_barbera.malic_acid
    ])

    # Post-hoc analýza (Tukey HSD pro ANOVA, Dunnův test pro Kruskal-Wallis)
    # Tuckey je vhodný pro parametrické testy (ANOVA)
    # Dunn je vhodný pro neparametrické testy (Kruskal-Wallis)

    tukey_alc = pairwise_tukeyhsd(endog=df_alc['Hodnota'], groups=df_alc['Odrůda'], alpha=alpha)
    print_results_posthoc_tukey("Alkohol", tukey_alc, alpha)

    dunn_ma = sp.posthoc_dunn(df_ma, val_col='Hodnota', group_col='Odrůda', p_adjust='bonferroni')
    print_results_posthoc_dunn("Malická kyselina", dunn_ma, alpha)

    # =========================================================================
    # BOD 5: VÍCEVÝBĚROVÉ TESTY O POLOZE
    # =========================================================================

    print_main_chapter("BOD 5: VÍCEROZMĚRNÝ PŘÍSTUP")

    data_list = []
    for i in range(len(wines_barolo.alcohol)):
        data_list.append({"Odrůda": "Barolo", "Alkohol": wines_barolo.alcohol[i], "Malická_kyselina": wines_barolo.malic_acid[i]})
    for i in range(len(wines_grignolino.alcohol)):
        data_list.append({"Odrůda": "Grignolino", "Alkohol": wines_grignolino.alcohol[i], "Malická_kyselina": wines_grignolino.malic_acid[i]})
    for i in range(len(wines_barbera.alcohol)):
        data_list.append({"Odrůda": "Barbera", "Alkohol": wines_barbera.alcohol[i], "Malická_kyselina": wines_barbera.malic_acid[i]})

    df_multi = pd.DataFrame(data_list)
    promenne = ["Alkohol", "Malická_kyselina"]

    # Manova analýza (MANOVA) pro vícerozměrné testy o poloze
    h0_manova = "Vícerozměrné průměry (profily) jsou stejné u všech odrůd."
    manova = MANOVA.from_formula('Alkohol + Malická_kyselina ~ Odrůda', data=df_multi)
    print_results_manova(manova.mv_test(), h0_manova, alpha)

    # Hottelingův T² test pro párové porovnání mezi odrůdami
    h0_hotelling = "Vícerozměrné průměry (profily) jsou u obou odrůd stejné."
    pairs = [("Barolo", "Grignolino"), ("Barolo", "Barbera"), ("Grignolino", "Barbera")]
    for g1, g2 in pairs:
        data_g1 = df_multi[df_multi["Odrůda"] == g1][promenne]
        data_g2 = df_multi[df_multi["Odrůda"] == g2][promenne]
        res = pg.multivariate_ttest(data_g1, data_g2)
        print_results_hotelling(g1, g2, res, h0_hotelling, alpha)
    print("   " + "—" * 70)

    # M-Testy (Box M test) pro ověření homogenity kovariančních matic mezi odrůdami
    h0_box = "Kovarianční matice (rozptyly a vztahy proměnných) jsou stejné u všech odrůd."
    box_m = pg.box_m(df_multi, dvs=promenne, group="Odrůda")
    print_results_box_m(box_m, h0_box, alpha)

    # Ověření vícerozměrné normality pro jednotlivé odrůdy (Mardia, Henze-Zirkler, Royston)
    h0_norm_multi = "Vícerozměrná data mají normální rozdělení."
    for odruda in ["Barolo", "Grignolino", "Barbera"]:
        norm_res = pg.multivariate_normality(df_multi[df_multi["Odrůda"] == odruda][promenne])
        print_results_multivariate_normality(odruda, norm_res, h0_norm_multi, alpha)
    print("   " + "—" * 70)

    sns.set_theme(style="whitegrid")

    # Elipsy pro vizualizaci vícerozměrných dat (Mahalanobisova vzdálenost)
    def draw_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
        scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    g = sns.pairplot(df_multi, hue="Odrůda", palette="Set2", diag_kind="kde", markers=["o", "s", "D"], height=3)

    g.fig.set_size_inches(15, 7) 
    g.fig.subplots_adjust(right=0.45, top=0.9)
    g.fig.suptitle("Komplexní vícerozměrná vizualizace (Bod 5)", y=0.98, fontweight="bold", fontsize=14, color="#2c3e50")
    ax_ell = g.fig.add_axes([0.55, 0.55, 0.40, 0.35])   
    ax_ci1 = g.fig.add_axes([0.55, 0.10, 0.17, 0.35]) 
    ax_ci2 = g.fig.add_axes([0.78, 0.10, 0.17, 0.35]) 

    colors = {"Barolo": "#66c2a5", "Grignolino": "#fc8d62", "Barbera": "#8da0cb"}
    for odruda in ["Barolo", "Grignolino", "Barbera"]:
        subset = df_multi[df_multi["Odrůda"] == odruda]
        sns.scatterplot(x=subset["Alkohol"], y=subset["Malická_kyselina"], color=colors[odruda], label=odruda, s=50, alpha=0.7, ax=ax_ell)
        draw_confidence_ellipse(subset["Alkohol"], subset["Malická_kyselina"], ax_ell, n_std=2.0, edgecolor=colors[odruda], facecolor='none', linewidth=2)
    ax_ell.set_title("Konfidenční elipsy (Mahalanobis)", fontweight="bold")
    ax_ell.set_xlabel("Alkohol")
    ax_ell.set_ylabel("Malická kyselina")
    ax_ell.legend(loc='upper right', fontsize=9)

    sns.pointplot(data=df_multi, x="Odrůda", y="Alkohol", hue="Odrůda", capsize=.1, palette="Set2", ax=ax_ci1, legend=False)
    ax_ci1.set_title("95% CI (Alkohol)", fontweight="bold", fontsize=11)
    ax_ci1.set_xlabel("")
    ax_ci1.tick_params(axis='x', rotation=30)

    sns.pointplot(data=df_multi, x="Odrůda", y="Malická_kyselina", hue="Odrůda", capsize=.1, palette="Set2", ax=ax_ci2, legend=False)
    ax_ci2.set_title("95% CI (Malická kyselina)", fontweight="bold", fontsize=11)
    ax_ci2.set_xlabel("")
    ax_ci2.tick_params(axis='x', rotation=30)
    plt.show()