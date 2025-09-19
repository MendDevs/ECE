import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# ------------------------------------------------------------------------
# PARTIE A : Chargement et exploration des données
# ------------------------------------------------------------------------

# Charger le fichier tabBats.txt
df = pd.read_csv("tabBats.txt", sep=r"\s+", quotechar='"')

# Nettoyer les noms de colonnes (au cas où il y aurait encore des guillemets)
df.columns = df.columns.str.replace('"', '')

# Afficher les 5 premières lignes
print("Aperçu du DataFrame :")
print(df.head(), "\n")

# Colonnes utiles pour la régression
df_reg = df[["Species", "BOW", "BRW"]].dropna()

# Vérifier les dimensions
print(f"Nombre d'observations : {df_reg.shape[0]}")

# ------------------------------------------------------------------------
# PARTIE B : Régression linéaire BRW ~ BOW
# ------------------------------------------------------------------------

X = df_reg["BOW"]
y = df_reg["BRW"]

# Ajouter une constante pour l'intercept
X_const = sm.add_constant(X)

# Ajuster le modèle OLS
model = sm.OLS(y, X_const).fit()

print("\nRésumé du modèle :")
print(model.summary())

# Graphique nuage de points + droite de régression
plt.figure(figsize=(8,6))
plt.scatter(df_reg["BOW"], df_reg["BRW"], label="Observations")
plt.plot(df_reg["BOW"], model.predict(X_const), color='red', label="Régression")
plt.xlabel("BOW (Brain Weight)")
plt.ylabel("BRW (Body Weight)")
plt.title("Régression BRW ~ BOW")
plt.legend()
plt.show()

# ------------------------------------------------------------------------
# PARTIE C : Retrait de l'espèce atypique et comparaison des droites
# ------------------------------------------------------------------------

# Identifier l'espèce atypique : Pteropus vampyrus (id=27 dans ton dataset)
df_no_outlier = df_reg[df_reg["Species"] != "Pteropus vampyrus"]

X2 = df_no_outlier["BOW"]
y2 = df_no_outlier["BRW"]
X2_const = sm.add_constant(X2)

model2 = sm.OLS(y2, X2_const).fit()

print("\nRésumé du modèle sans l'espèce atypique :")
print(model2.summary())

# Graphique comparant les deux droites
plt.figure(figsize=(8,6))
plt.scatter(df_reg["BOW"], df_reg["BRW"], label="Toutes les espèces", alpha=0.6)
plt.plot(df_reg["BOW"], model.predict(X_const), color='red', label="Avec l'espèce atypique")
plt.plot(df_no_outlier["BOW"], model2.predict(X2_const), color='green', label="Sans l'espèce atypique")
plt.xlabel("BOW (Brain Weight)")
plt.ylabel("BRW (Body Weight)")
plt.title("Comparaison des droites de régression")
plt.legend()
plt.show()

# ------------------------------------------------------------------------
# (Optionnel) Affichage des résidus
# ------------------------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(model.fittedvalues, model.resid, label="Résidus avec espèce atypique", alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Valeurs ajustées")
plt.ylabel("Résidus")
plt.title("Résidus du modèle initial")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(model2.fittedvalues, model2.resid, label="Résidus sans espèce atypique", alpha=0.6, color='green')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Valeurs ajustées")
plt.ylabel("Résidus")
plt.title("Résidus du modèle sans espèce atypique")
plt.show()
