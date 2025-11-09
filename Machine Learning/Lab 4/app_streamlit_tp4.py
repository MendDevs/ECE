# app_streamlit_tp4.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import io

st.set_page_config(layout="wide", page_title="TP4 - Clustering (Iris, Planete, Pipeline)")

st.title("TP 4 — Classification non supervisée (Streamlit)")
st.markdown("""
Ce notebook/streamlit couvre :
- Analyse Iris : PCA + K-Means (visualisation, centroides, répétitions, tableau de contingence, silhouette).
- Clustering hiérarchique : dendrogramme, linkage complet vs moyen.
- Nombre optimal de clusters pour `planete.csv` via Davies-Bouldin & Calinski-Harabasz (méthode du coude).
- Pipeline complet pour un jeu réel (prétraitement, imputation, standardisation, non-supervisé & supervisé).
""")

###########################
# Helper functions
###########################
def load_iris(filepath_or_buffer):
    # iris.data format: 4 numeric columns + label string
    df = pd.read_csv(filepath_or_buffer, header=None)
    # Some iris.data versions have no header; ensure only 5 columns
    if df.shape[1] > 5:
        df = df.iloc[:, :5]
    df.columns = ['sepal_length','sepal_width','petal_length','petal_width','label']
    return df

def plot_scatter_with_centroids(X2d, labels, centroids=None, title=""):
    fig, ax = plt.subplots(figsize=(6,5))
    scatter = ax.scatter(X2d[:,0], X2d[:,1], c=labels, cmap='tab10', s=40, edgecolor='k', alpha=0.7)
    if centroids is not None:
        ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='red', label='centroides', edgecolor='k')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

###########################
# Section 1: Iris + KMeans + PCA
###########################
st.header("1) Analyse des Iris - PCA + K-Means")

uploaded_iris = st.file_uploader("Upload iris.data (ou laisse vide pour utiliser l'exemple intégré)", type=['csv','data','txt'])
if uploaded_iris is None:
    st.info("Utilisation de l'iris intégré (pandas read_csv depuis scikit-learn si besoin).")
    from sklearn.datasets import load_iris
    iris_bunch = load_iris()
    data_df = pd.DataFrame(iris_bunch.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
    labels = pd.Series(iris_bunch.target).map({0:'setosa',1:'versicolor',2:'virginica'})
else:
    df_iris = load_iris(uploaded_iris)
    labels = df_iris['label']
    data_df = df_iris.drop(columns=['label'])

st.write("Aperçu des données (Iris):")
st.dataframe(data_df.head())

# Standardize before PCA/KMeans
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_df)

# PCA -> 2 composantes
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(data_scaled)

st.subheader("PCA (2 composantes)")
st.write("Explained variance ratio:", pca.explained_variance_ratio_)

# KMeans on PCA projection
n_clusters = st.slider("Nombre de clusters K (K-Means) sur la projection PCA", 2, 6, 3)
n_repeats = st.slider("Nombre de répétitions (different init)", 1, 10, 3)

st.write("Exécuter KMeans plusieurs fois (différentes initialisations):")
kmeans_results = []
for i in range(n_repeats):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42+i)
    y_km = km.fit_predict(X_pca)
    kmeans_results.append((y_km, km.cluster_centers_, km.inertia_))

    st.write(f"Répétition {i+1} — inertia: {km.inertia_}")
    plot_scatter_with_centroids(X_pca, y_km, centroids=km.cluster_centers_, title=f"KMeans répétition {i+1}")

st.write("Observation : les partitions peuvent changer selon l'initialisation — inertia et silhouette aident à choisir la meilleure.")

# Compare with true labels (projection)
st.subheader("Comparaison visuelle avec labels réels (projection PCA)")
plot_scatter_with_centroids(X_pca, labels.map({'setosa':0,'versicolor':1,'virginica':2}).values, centroids=None, title="Labels réels projetés (PCA)")

# Contingency table between one chosen clustering and true labels
chosen_idx = st.selectbox("Choisis la solution KMeans (par index) pour la table de contingence", options=list(range(len(kmeans_results))), format_func=lambda x: f"Solution {x+1}")
y_chosen, centers_chosen, _ = kmeans_results[chosen_idx]
ct = pd.crosstab(y_chosen, labels)
st.subheader("Tableau de contingence (clustering vs labels théoriques)")
st.dataframe(ct)

# Silhouette
if len(set(y_chosen)) > 1:
    sil = silhouette_score(X_pca, y_chosen)
    st.write(f"Silhouette score (solution choisie, sur projection PCA) = {sil:.4f}")
else:
    st.write("Silhouette non défini (1 cluster seulement).")

#################################################
# Refaire sur données originales (non-projetées)
#################################################
st.header("1.9) Refaire KMeans sur les données originales (standardisées)")

kmeans_orig = KMeans(n_clusters=n_clusters, n_init=10, random_state=123)
y_km_orig = kmeans_orig.fit_predict(data_scaled)
centers_orig = kmeans_orig.cluster_centers_

# Projetons les centroïdes par PCA pour visualiser dans l'espace 2D
centers_orig_proj = pca.transform(centers_orig)

st.subheader("Clustering KMeans (données originales) projeté sur les 2 PC pour visualisation")
plot_scatter_with_centroids(X_pca, y_km_orig, centroids=centers_orig_proj, title="KMeans (données originales) projetées")

ct_orig = pd.crosstab(y_km_orig, labels)
st.subheader("Tableau de contingence (KMeans sur données originales vs labels)")
st.dataframe(ct_orig)

if len(set(y_km_orig)) > 1:
    sil_orig = silhouette_score(data_scaled, y_km_orig)
    st.write(f"Silhouette (données originales) = {sil_orig:.4f}")
else:
    st.write("Silhouette non défini.")

st.markdown("""
**Remarque (avantages/inconvénients)**:
- PCA réduit la dimension, accélère et facilite la visualisation ; mais peut perdre de l'information pertinente si les classes ne sont pas bien séparées dans les premières composantes.
- Utiliser les données originales conserve toute l'information mais rend la visualisation directe difficile et peut ralentir l'algorithme si la dimension est élevée.
""")

###########################
# Section 2: Clustering hiérarchique
###########################
st.header("2) Clustering hiérarchique sur Iris (linkage complet vs moyen)")

# AgglomerativeClustering params et dendrogramme
st.write("On utilise AgglomerativeClustering de sklearn et scipy pour le dendrogramme.")

linkage_choice = st.selectbox("Choix du linkage pour le dendrogramme", options=['complete','average'])
st.write("Calcul du dendrogramme (avec scipy)")
# Compute distance matrix on scaled data
dist_mat = pdist(data_scaled)
Z = hierarchy.linkage(dist_mat, method=linkage_choice)

fig_dend, ax_d = plt.subplots(figsize=(8,4))
dn = hierarchy.dendrogram(Z, ax=ax_d, color_threshold=None)
ax_d.set_title(f"Dendrogramme (linkage={linkage_choice})")
st.pyplot(fig_dend)

# Couper à 3 clusters et comparer
st.write("Couper le dendrogramme pour obtenir 3 clusters (partition).")
from scipy.cluster.hierarchy import fcluster
y_hier = fcluster(Z, t=3, criterion='maxclust') - 1  # make 0-based
ct_hier = pd.crosstab(y_hier, labels)
st.subheader(f"Tableau de contingence (linkage={linkage_choice})")
st.dataframe(ct_hier)

# silhouette for hierarchical partition
if len(set(y_hier)) > 1:
    sil_hier = silhouette_score(data_scaled, y_hier)
    st.write(f"Silhouette (hierarchical {linkage_choice}) = {sil_hier:.4f}")
else:
    st.write("Silhouette non défini.")

st.markdown("""
Refaites maintenant avec linkage='complete' et linkage='average' et comparez : 
- le dendrogramme change ; certaines fusions sont plus conservatrices (complete tends to produce compact clusters), average lisse les distances.
- Comparez tableaux de contingence et silhouette pour décider quelle méthode est meilleure.
""")

###########################
# Section 3: Planete.csv - nombre optimal de clusters
###########################
st.header("3) Nombre optimal de clusters — jeu `planete.csv`")

uploaded_planete = st.file_uploader("Upload planete.csv (ou laisse vide si tu veux sauter)", type=['csv'])
if uploaded_planete is not None:
    df_plan = pd.read_csv(uploaded_planete)
    st.write("Aperçu planete.csv")
    st.dataframe(df_plan.head())

    # Assume last column is label
    X_plan = df_plan.iloc[:, :-1].values
    labels_plan = df_plan.iloc[:, -1].values
    # standardize
    X_plan_scaled = StandardScaler().fit_transform(X_plan)

    ks = list(range(2,11))
    db_scores = []
    ch_scores = []
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        ypred = km.fit_predict(X_plan_scaled)
        inertias.append(km.inertia_)
        db_scores.append(davies_bouldin_score(X_plan_scaled, ypred))
        ch_scores.append(calinski_harabasz_score(X_plan_scaled, ypred))

    fig_elbow, ax_el = plt.subplots(figsize=(8,4))
    ax_el.plot(ks, db_scores, marker='o', label='Davies-Bouldin (plus bas = mieux)')
    ax_el.plot(ks, ch_scores, marker='s', label='Calinski-Harabasz (plus haut = mieux)')
    ax_el.set_xlabel('k')
    ax_el.set_title('Indices pour choix du nombre de clusters')
    ax_el.legend()
    st.pyplot(fig_elbow)

    st.write("Interprétation :")
    st.write("- Choisir k qui minimise Davies-Bouldin et qui maximise Calinski-Harabasz. Souvent on prend un compromis (elbow).")
    best_k_db = ks[int(np.argmin(db_scores))]
    best_k_ch = ks[int(np.argmax(ch_scores))]
    st.write(f"Meilleur k (DB) = {best_k_db}; meilleur k (CH) = {best_k_ch}")

    st.write("Expliquer la méthode du coude : on regarde la courbe (indice) et cherche un point après lequel l'amélioration est marginale (coude).")
else:
    st.info("Upload planete.csv pour exécuter l'analyse du nombre optimal de clusters.")

###########################
# Section 4: Pipeline complet (jeu réel)
###########################
st.header("4) Pipeline d'analyse de données & apprentissage")

st.markdown("""
Téléverse un jeu de données (CSV). Le pipeline fera :
- aperçu, types, valeurs manquantes,
- imputation (moyenne pour numériques, mode pour catégoriques),
- encodage (one-hot simplifié), standardisation,
- clustering (KMeans, DBSCAN, Hierarchical) et indices internes (silhouette, DB),
- si étiquettes présentes -> classification supervisée (KNN, Decision Tree) et métriques.
""")

uploaded_ds = st.file_uploader("Upload dataset pour pipeline (CSV) — facultatif", type=['csv'], key='pipeline_ds')
if uploaded_ds is not None:
    df = pd.read_csv(uploaded_ds)
    st.subheader("Aperçu dataset")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # Basic stats
    st.subheader("Statistiques descriptives")
    st.write(df.describe(include='all'))

    # Missing values
    st.subheader("Valeurs manquantes")
    st.write(df.isnull().sum())

    # Separate label if user indicates
    has_label = st.checkbox("Le fichier contient-il une colonne 'label' (ou y a-t-il une colonne cible) ?", value=False)
    label_col = None
    if has_label:
        label_col = st.text_input("Nom de la colonne cible (ex: label)", value='label')
        if label_col not in df.columns:
            st.error("Colonne introuvable — vérifie le nom exact.")
    # Preprocessing: simple imputation
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    st.write("Numerical cols:", num_cols)
    st.write("Categorical cols:", cat_cols)

    imputer_num = SimpleImputer(strategy='mean')
    if len(num_cols) > 0:
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    if len(cat_cols) > 0:
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # Simple encoding for categoricals (one-hot)
    if len(cat_cols) > 0:
        df_enc = pd.get_dummies(df.drop(columns=[label_col] if (has_label and label_col in df.columns) else []), columns=cat_cols, drop_first=True)
    else:
        df_enc = df.drop(columns=[label_col] if (has_label and label_col in df.columns) else [])

    # Standardize features
    scaler_pipe = StandardScaler()
    X_pipe = scaler_pipe.fit_transform(df_enc.values)

    # Clustering choices
    st.subheader("Clustering (KMeans / DBSCAN / Agglomerative)")
    k_clust = st.slider("K pour KMeans", 2, 10, 3)
    km_pipe = KMeans(n_clusters=k_clust, n_init=10, random_state=1)
    y_km_pipe = km_pipe.fit_predict(X_pipe)
    st.write("Silhouette (KMeans):", silhouette_score(X_pipe, y_km_pipe) if len(set(y_km_pipe))>1 else "N/A")
    st.write("Davies-Bouldin (KMeans):", davies_bouldin_score(X_pipe, y_km_pipe))

    # DBSCAN
    eps = st.number_input("eps pour DBSCAN", value=0.5, format="%.3f")
    min_samples = st.number_input("min_samples pour DBSCAN", value=5, format="%d")
    db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X_pipe)
    labels_db = db.labels_
    if len(set(labels_db))>1 and -1 not in set(labels_db):
        st.write("Silhouette (DBSCAN):", silhouette_score(X_pipe, labels_db))
    else:
        st.write("Silhouette (DBSCAN): non défini ou trop de bruit (-1).")
    # Agglomerative
    ac = AgglomerativeClustering(n_clusters=k_clust).fit(X_pipe)
    labels_ac = ac.labels_
    st.write("Silhouette (Agglomerative):", silhouette_score(X_pipe, labels_ac) if len(set(labels_ac))>1 else "N/A")

    # If labels present -> supervised classification
    if has_label and (label_col in df.columns):
        st.subheader("Apprentissage supervisé (si étiquettes présentes)")
        y_super = df[label_col].values
        # encode labels if strings
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y_super)
        X_train, X_test, y_train, y_test = train_test_split(X_pipe, y_enc, test_size=0.25, random_state=42)
        # KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        acc_knn = knn.score(X_test, y_test)
        st.write("K-NN accuracy (test):", acc_knn)
        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        acc_dt = dt.score(X_test, y_test)
        st.write("Decision Tree accuracy (test):", acc_dt)
        st.write("Note: pour un rapport, présente la matrice de confusion et F1-score selon la tâche.")

    st.success("Pipeline exécuté — intègre ces sorties dans ton rapport Streamlit.")
else:
    st.info("Upload un dataset si tu veux que j'exécute le pipeline dessus.")

###########################
# Wrap-up / conseils pour rapport
###########################
st.header("Conseils pour rédiger le rapport (à déposer sur Moodle)")
st.markdown("""
- **Introduction** : objectif du TP, description brève des jeux de données.
- **Méthodologie** : étapes d'analyse (PCA, standardisation, choix d'indices).
- **Résultats** : figures (scatter avec centroides), dendrogrammes, tableaux de contingence, valeurs des indices (silhouette, DB, CH).
- **Discussion** : expliquer pourquoi KMeans varie selon l'initialisation, avantages/inconvénients PCA vs données originales, comparaison linkage complet vs moyen, justification du k retenu pour `planete.csv`.
- **Conclusion** : synthèse et pistes d'amélioration.
- **Annexe** : code source (ce streamlit), commandes pour exécuter.
""")

st.markdown("**Bon travail !** Si tu veux, je peux :\n- te générer un README prêt à coller sur Moodle, \n- extraire les figures en PNG/ZIP prêtes à joindre,\n- adapter le notebook pour une version `.py` ou `.ipynb` spécifique.")

