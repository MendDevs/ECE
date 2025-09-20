##############################################
# TP2: Emmanuel MORRIS & LELE KOMGUEM ADRIEN #
##############################################
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

############### Part A: Chargement ex exploration de données ##########


### Affichage du premier ligne ######
df = pd.read_csv(
                 "tabBats.txt",
                 sep=r"\s+", #seperator = spaces
                 quotechar='"'
                 )

st.subheader('General Data')
df.set_index('id', inplace=True)
st.dataframe(df)

st.title("Part A")
### Affichage des premieres lignes, types et stats ###
st.header("Data Analyses: ")

st.subheader("Display of Data")
st.dataframe(df.head())

st.subheader("Types of Colonnes")
st.write(df.dtypes)

st.subheader("Statistiques descriptives")
st.write(df.describe())

df_reg = df[["Species", "BOW", "BRW"]]
st.subheader("Variables Pertinents")
st.dataframe(df_reg.head())


# -------------------------
# Partie B : Première régression linéaire simple
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set(style="whitegrid")

st.title("Partie B — Régression linéaire simple")

# 1) Nuage de points (BRW en fonction de BOW)
st.subheader("1) Nuage de points — BRW vs BOW")

fig, ax = plt.subplots(figsize=(14, 6))             # plus large
sns.scatterplot(data=df_reg, x="BOW", y="BRW", hue="Species", ax=ax, legend=False)
ax.set_title("Nuage de points : Masse cérébrale (BRW) vs Masse corporelle (BOW)")
ax.set_xlabel("BOW (g?)")
ax.set_ylabel("BRW (mg? or same unit as file)")

# Option : afficher légende compacte à droite (si peu de species)
#leg = ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.tight_layout()
st.pyplot(fig)

st.write("**Observation (à rédiger)** : décrire la tendance (p.ex. relation positive attendue). Repérer visuellement les espèces atypiques (outliers), par ex. Pteropus vampyrus si présent.")

# 2) Ajuster modèle linéaire simple (BRW ~ BOW) avec statsmodels OLS
st.subheader("2) Ajustement du modèle — BRW ~ BOW")

# Préparer X,y et fit OLS (avec intercept)
X = sm.add_constant(df_reg["BOW"])   # ajoute colonne const = 1
y = df_reg["BRW"]
model = sm.OLS(y, X).fit()

# 3) Résumé simplifié du modèle : coefficients, p-values, IC, R², RMSE
ci = model.conf_int()
results_df = pd.DataFrame({
    "param": model.params.index,
    "coef": model.params.values,
    "pvalue": model.pvalues.values,
    "CI_low": ci[0].values,
    "CI_high": ci[1].values
})
results_df = results_df.set_index("param")

st.subheader("Coefficients et intervalle de confiance (95%)")
st.table(results_df)

st.markdown(f"- **R²** : `{model.rsquared:.4f}`  \n- **Adj. R²** : `{model.rsquared_adj:.4f}`")
rmse = np.sqrt(np.mean(model.resid**2))
st.write(f"- **RMSE (résidus)** : {rmse:.4f}")

# Si tu veux voir le résumé complet :
with st.expander("Afficher le résumé complet (statsmodels)"):
    st.text(model.summary().as_text())

# 4) Analyse des résidus : Residuals vs Fitted + QQ-plot
st.subheader("Analyse des résidus")

fig_res, axes = plt.subplots(1, 2, figsize=(14,4))

# Residuals vs Fitted
fitted_vals = model.fittedvalues
resids = model.resid
axes[0].scatter(fitted_vals, resids, alpha=0.7)
axes[0].axhline(0, color='k', linestyle='--')
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")

# QQ-plot
sm.qqplot(resids, line='45', ax=axes[1])
axes[1].set_title("QQ-plot des résidus")

fig_res.tight_layout()
st.pyplot(fig_res)

st.write("""
Interprétation rapide à inclure dans le rapport :
- **Coefficients** : expliquer l'intercept et la pente (ex : une augmentation de 1 unité de BOW est associée à ~b unités de BRW).
- **p-values** : si p < 0.05, le coefficient est statistiquement significatif.
- **R²** : proportion de variance expliquée par BOW.
- **Résidus** : vérifier la non-présence de tendance (homoscédasticité) et la normalité approximative via QQ-plot.
""")

# 5) Droite de régression sur le nuage de points
st.subheader("Droite de régression superposée")

fig_reg, axr = plt.subplots(figsize=(14,6))
# scatter without legend to avoid huge legend; optionally color by species but hide legend
sns.scatterplot(data=df_reg, x="BOW", y="BRW", hue="Species", ax=axr, legend=False, alpha=0.7)

# Prepare line (use a sorted x for smooth line)
x_line = np.linspace(df_reg["BOW"].min(), df_reg["BOW"].max(), 200)
X_line = sm.add_constant(x_line)
y_line = model.predict(X_line)

axr.plot(x_line, y_line, color="red", linewidth=2, label="Droite de régression")
axr.set_xlabel("BOW")
axr.set_ylabel("BRW")
axr.set_title("BRW vs BOW avec droite de régression")
axr.legend(fontsize=10, loc="upper left")
fig_reg.tight_layout()
st.pyplot(fig_reg)

# -------------------------
# Partie C : Analyse avec retrait d'une espèce atypique
# -------------------------
st.title("Partie C — Analyse sans l'espèce atypique Pteropus vampyrus")

# 1) Créer tab2 sans Pteropus vampyrus
tab2 = df_reg[df_reg["Species"] != "Pteropus vampyrus"].copy()

# 2) Comparer visuellement les nuages de points
st.subheader("1) Nuages de points comparatifs (avec vs sans Pteropus vampyrus)")

fig_comp, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)

sns.scatterplot(data=df_reg, x="BOW", y="BRW", hue="Species", ax=axes[0], legend=False)
axes[0].set_title("Avec Pteropus vampyrus")
axes[0].set_xlabel("BOW")
axes[0].set_ylabel("BRW")

sns.scatterplot(data=tab2, x="BOW", y="BRW", hue="Species", ax=axes[1], legend=False)
axes[1].set_title("Sans Pteropus vampyrus")
axes[1].set_xlabel("BOW")
axes[1].set_ylabel("")

fig_comp.tight_layout()
st.pyplot(fig_comp)

st.write("Observation : en retirant *Pteropus vampyrus*, on réduit l’influence d’un point extrême de grande masse corporelle et cérébrale.")

# 3) Ajuster un second modèle sur tab2
st.subheader("2) Ajustement du modèle sans Pteropus vampyrus")

X2 = sm.add_constant(tab2["BOW"])
y2 = tab2["BRW"]
model2 = sm.OLS(y2, X2).fit()

# Résultats comparatifs
ci2 = model2.conf_int()
results_df2 = pd.DataFrame({
    "param": model2.params.index,
    "coef": model2.params.values,
    "pvalue": model2.pvalues.values,
    "CI_low": ci2[0].values,
    "CI_high": ci2[1].values
}).set_index("param")

st.subheader("Coefficients du modèle 2 (sans Pteropus vampyrus)")
st.table(results_df2)

st.markdown(f"- **R² (avec Pteropus vampyrus)** : `{model.rsquared:.4f}`  \n"
            f"- **R² (sans Pteropus vampyrus)** : `{model2.rsquared:.4f}`")

rmse2 = np.sqrt(np.mean(model2.resid**2))
st.write(f"- **RMSE (sans Pteropus vampyrus)** : {rmse2:.4f}")

# Analyse des résidus pour le modèle sans Pteropus
fig_res2, axes2 = plt.subplots(1, 2, figsize=(14,4))
fitted_vals2 = model2.fittedvalues
resids2 = model2.resid

axes2[0].scatter(fitted_vals2, resids2, alpha=0.7)
axes2[0].axhline(0, color='k', linestyle='--')
axes2[0].set_xlabel("Fitted values")
axes2[0].set_ylabel("Residuals")
axes2[0].set_title("Residuals vs Fitted (sans Pteropus)")

sm.qqplot(resids2, line='45', ax=axes2[1])
axes2[1].set_title("QQ-plot des résidus (sans Pteropus)")

fig_res2.tight_layout()
st.pyplot(fig_res2)

# 4) Superposer les deux droites de régression
st.subheader("3) Superposition des droites de régression")

fig_lines, axl = plt.subplots(figsize=(14,6))
sns.scatterplot(data=df_reg, x="BOW", y="BRW", color="gray", alpha=0.5, ax=axl, legend=False)

# Droite 1 (avec Pteropus)
x_line = np.linspace(df_reg["BOW"].min(), df_reg["BOW"].max(), 200)
y_line = model.predict(sm.add_constant(x_line))
axl.plot(x_line, y_line, color="red", linewidth=2, label="Avec Pteropus")

# Droite 2 (sans Pteropus)
x_line2 = np.linspace(tab2["BOW"].min(), tab2["BOW"].max(), 200)
y_line2 = model2.predict(sm.add_constant(x_line2))
axl.plot(x_line2, y_line2, color="blue", linewidth=2, linestyle="--", label="Sans Pteropus")

axl.set_title("Comparaison des droites de régression")
axl.set_xlabel("BOW")
axl.set_ylabel("BRW")
axl.legend()

st.pyplot(fig_lines)

st.write("""
**Commentaire :**  
- En retirant *Pteropus vampyrus*, la pente de la droite peut changer (souvent plus forte car l’outlier « écrasait » la relation).  
- Le R² peut augmenter si l’espèce atypique perturbait la relation.  
- Les résidus semblent mieux répartis sans ce point extrême.  
""")

