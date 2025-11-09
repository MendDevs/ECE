import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                            confusion_matrix, adjusted_rand_score, accuracy_score, 
                            classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(layout="wide", page_title="TP4 - Classification Non Supervisée")

# Header
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 30px;'>
    <h1> Machine Learning I</h1>
    <h2>TP4 - Classification Non Supervisée (Clustering)</h2>
    <p><em>Lecture 4 & 4bis: Introduction to unsupervised learning clustering</em></p>
    <p><strong>By: Emmanuel Morris </strong></p>
    <p><strong>Supervised By: Issam Falih - Department of Computer Science</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title(" Navigation")
section = st.sidebar.selectbox(
    "Sélectionner l'exercice",
    [
        "Exercice 1: Iris de Fisher - K-Moyennes",
        "Exercice 2: Clustering Hiérarchique - Iris",
        "Exercice 3: Nombre Optimal de Clusters - Exoplanètes",
        "Exercice 4: Pipeline Complet d'Analyse"
    ]
)

# Sidebar - Rappels théoriques
with st.sidebar.expander(" Rappels Théoriques", expanded=False):
    st.markdown("""
    **Types d'algorithmes:**
    - **Prototype-based**: K-Means, Fuzzy C-Means
    - **Hierarchical**: HCA (Agglomerative/Divisive)
    - **Density-based**: DBSCAN, OPTICS
    - **Distribution-based**: EM (Gaussian Mixture)
    
    **Indices d'évaluation:**
    - **Internes**: Silhouette, Davies-Bouldin, Calinski-Harabasz
    - **Externes**: Rand Index, ARI, Accuracy
    
    **Critères de linkage:**
    - **Single**: Distance minimale
    - **Complete**: Distance maximale
    - **Average**: Distance moyenne
    - **Ward**: Minimise la variance
    """)

# Utility Functions
@st.cache_data
def load_iris_data():
    """Charge les données Iris"""
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        data['species'] = iris.target_names[iris.target]
        return data
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

@st.cache_data
def load_planete_data():
    """Charge les données d'exoplanètes"""
    try:
        data = pd.read_csv("planete.csv", delimiter=';')
        return data
    except:
        # Données simulées
        np.random.seed(42)
        n_samples = 300
        data = pd.DataFrame({
            'temperature': np.random.normal(500, 200, n_samples),
            'pressure': np.random.exponential(2, n_samples),
            'oxygen': np.random.beta(2, 5, n_samples),
            'methane': np.random.gamma(2, 0.5, n_samples),
            'water_vapor': np.random.uniform(0, 1, n_samples),
            'cluster_true': np.random.choice([0, 1, 2, 3], n_samples)
        })
        return data

def plot_clusters_2d(X, labels, centroids=None, title="Clustering Results"):
    """Visualise les clusters en 2D"""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 3)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                  label=f'Cluster {label}', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='X', s=300, linewidths=3, 
                  label='Centroïdes', edgecolors='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Première Composante', fontsize=12)
    ax.set_ylabel('Deuxième Composante', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_dendrogram_custom(linkage_matrix, title="Dendrogramme"):
    """Affiche un dendrogramme optimisé"""
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(linkage_matrix, ax=ax, color_threshold=0.7*max(linkage_matrix[:,2]))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Échantillons', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def calculate_contingency_table(true_labels, pred_labels):
    """Calcule le tableau de contingence"""
    return pd.crosstab(true_labels, pred_labels, rownames=['Vrais'], colnames=['Prédits'], margins=True)

def safe_data_preparation(df, target_col):
    """Préparation sécurisée des données"""
    if target_col not in df.columns:
        st.error(f"Colonne {target_col} introuvable")
        return None, None
    
    y = df[target_col].copy()
    X = df.drop(target_col, axis=1).copy()
    
    # Garder uniquement colonnes numériques
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.error("Aucune colonne numérique")
        return None, None
    
    X = X[numeric_cols].copy()
    
    # Supprimer valeurs manquantes
    if X.isnull().any().any():
        mask = ~X.isnull().any(axis=1)
        X = X[mask].copy()
        y = y[mask].copy()
    
    # Vérifier valeurs finies
    data_array = X.values.astype(float)
    if not np.isfinite(data_array).all():
        finite_mask = np.isfinite(data_array).all(axis=1)
        X = X[finite_mask].copy()
        y = y.iloc[finite_mask].copy()
    
    return X, y

###################### EXERCICE 1 ######################
if section == "Exercice 1: Iris de Fisher - K-Moyennes":
    st.markdown("#  Exercice 1: Analyse des Iris de Fisher avec K-Moyennes")
    st.markdown("---")
    
    # Question 1
    st.markdown("##  Question 1: Chargement des données")
    iris_data = load_iris_data()
    
    if iris_data is not None:
        st.success(" Données chargées")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(iris_data.head(10))
        with col2:
            st.write(f"**Dimensions**: {iris_data.shape}")
            st.write(f"**Classes**: {iris_data['species'].unique()}")
            st.write(iris_data['species'].value_counts())
        
        st.markdown("---")
        
        # Question 2
        st.markdown("## ✂️ Question 2: Séparation des labels")
        X, y = safe_data_preparation(iris_data, 'species')
        
        if X is not None:
            st.success(" Labels séparés")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Données (features)**")
                st.dataframe(X.head())
            with col2:
                st.markdown("**Labels**")
                st.write(y.head(10).tolist())
            
            st.markdown("---")
            
            # Question 3
            st.markdown("##  Question 3: Analyse en Composantes Principales (PCA)")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            st.success("PCA appliquée")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Variance expliquée**: {pca.explained_variance_ratio_}")
                st.write(f"**Total**: {pca.explained_variance_ratio_.sum():.3f}")
                
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                st.dataframe(pca_df.head())
            
            with col2:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, 
                                   cmap='viridis', alpha=0.7, s=60, edgecolors='black')
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
                ax.set_title('Données Iris - PCA')
                plt.colorbar(scatter, ax=ax, label='Espèce')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Question 4
            st.markdown("##  Question 4: K-Means sur données PCA")
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters_pca = kmeans.fit_predict(X_pca)
            centroids = kmeans.cluster_centers_
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Inertie", f"{kmeans.inertia_:.3f}")
                st.metric("Itérations", kmeans.n_iter_)
                st.write("**Distribution des clusters:**")
                st.write(pd.Series(clusters_pca).value_counts().sort_index())
            
            with col2:
                fig = plot_clusters_2d(X_pca, clusters_pca, centroids, "K-Means sur PCA (k=3)")
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Question 5
            st.markdown("##  Question 5: Répétition de K-Means - Stabilité")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            results = []
            for i in range(5):
                kmeans_iter = KMeans(n_clusters=3, random_state=i, n_init=1)
                clusters = kmeans_iter.fit_predict(X_pca)
                
                unique_labels = np.unique(clusters)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = clusters == label
                    axes[i].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                  c=[colors[j]], alpha=0.7, edgecolors='black')
                
                axes[i].scatter(kmeans_iter.cluster_centers_[:, 0], 
                              kmeans_iter.cluster_centers_[:, 1], 
                              c='red', marker='X', s=200, linewidths=3)
                axes[i].set_title(f'Run {i+1} (Inertie: {kmeans_iter.inertia_:.2f})')
                axes[i].grid(True, alpha=0.3)
                
                results.append({
                    'Run': i+1,
                    'Inertie': kmeans_iter.inertia_,
                    'Iterations': kmeans_iter.n_iter_
                })
            
            fig.delaxes(axes[5])
            plt.tight_layout()
            st.pyplot(fig)
            
            results_df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(results_df)
            
            with col2:
                st.markdown(f"""
                **Analyse:**
                - **Inertie moyenne**: {results_df['Inertie'].mean():.3f} ± {results_df['Inertie'].std():.3f}
                - **Stabilité**: {'🟢 Bonne' if results_df['Inertie'].std() < 1 else '🟡 Variable'}
                
                **💡 Explication:**
                - K-Means converge vers des **minima locaux**
                - L'**initialisation** influence le résultat
                - Solutions peuvent être **équivalentes** (permutation labels)
                """)
            
            st.markdown("---")
            
            # Question 6
            st.markdown("## ⚖️ Question 6: Comparaison avec les vrais labels")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # K-Means
                unique_clusters = np.unique(clusters_pca)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
                
                for i, cluster in enumerate(unique_clusters):
                    mask = clusters_pca == cluster
                    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
                
                ax1.scatter(centroids[:, 0], centroids[:, 1], 
                           c='red', marker='X', s=200, linewidths=3)
                ax1.set_title('K-Means')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Vrais labels
                for i, species in enumerate(le.classes_):
                    mask = y_encoded == i
                    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=species, alpha=0.7)
                
                ax2.set_title('Vrais Labels')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                ari_score = adjusted_rand_score(y_encoded, clusters_pca)
                
                # Alignement optimal
                cm = confusion_matrix(y_encoded, clusters_pca)
                row_ind, col_ind = linear_sum_assignment(-cm)
                aligned_clusters = np.zeros_like(clusters_pca)
                for i, j in zip(row_ind, col_ind):
                    aligned_clusters[clusters_pca == j] = i
                
                accuracy = accuracy_score(y_encoded, aligned_clusters)
                
                st.markdown(f"""
                ** Métriques:**
                - **ARI**: {ari_score:.3f}
                - **Accuracy**: {accuracy:.3f}
                
                **Interprétation:**
                - ARI > 0.7: 🟢 Excellent
                - ARI > 0.5: 🟡 Bon
                - ARI > 0.3: 🟠 Modéré
                """)
            
            st.markdown("---")
            
            # Question 7
            st.markdown("## 📋 Question 7: Tableau de contingence")
            
            contingency_table = calculate_contingency_table(y, clusters_pca)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(contingency_table)
            
            with col2:
                cluster_purities = []
                for cluster in range(3):
                    cluster_mask = clusters_pca == cluster
                    if cluster_mask.sum() > 0:
                        cluster_labels = y[cluster_mask]
                        most_common = cluster_labels.mode()[0]
                        purity = (cluster_labels == most_common).mean()
                        cluster_purities.append(purity)
                
                overall_purity = np.mean(cluster_purities)
                
                st.markdown(f"""
                ** Pureté:**
                - **Moyenne**: {overall_purity:.3f}
                - **Par cluster**: {[f'{p:.3f}' for p in cluster_purities]}
                """)
            
            st.markdown("---")
            
            # Question 8
            st.markdown("##  Question 8: Indice de Silhouette")
            
            silhouette_avg = silhouette_score(X_pca, clusters_pca)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                
                if silhouette_avg > 0.70:
                    interpretation = "🟢 Structures fortes"
                elif silhouette_avg > 0.50:
                    interpretation = "🟡 Structures raisonnables"
                elif silhouette_avg > 0.25:
                    interpretation = "🟠 Structures faibles"
                else:
                    interpretation = "🔴 Pas de structure"
                
                st.markdown(f"**Interprétation**: {interpretation}")
            
            with col2:
                st.markdown("""
                **📚 Formule:**
                ```
                s(x) = (b(x) - a(x)) / max(a(x), b(x))
                ```
                
                - **a(x)**: Distance intra-cluster
                - **b(x)**: Distance au cluster proche
                
                **Valeurs:**
                - s ≈ 1: Bien assigné ✅
                - s ≈ 0: Frontière ⚠️
                - s < 0: Mal assigné ❌
                """)
            
            st.markdown("---")
            
            # Question 9
            st.markdown("## 🔬 Question 9: Données originales vs PCA")
            
            kmeans_orig = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters_orig = kmeans_orig.fit_predict(X_scaled)
            
            # Métriques
            metrics_data = {
                'Métrique': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'ARI', 'Inertie'],
                'Données Originales': [
                    silhouette_score(X_scaled, clusters_orig),
                    davies_bouldin_score(X_scaled, clusters_orig),
                    calinski_harabasz_score(X_scaled, clusters_orig),
                    adjusted_rand_score(y_encoded, clusters_orig),
                    kmeans_orig.inertia_
                ],
                'Données PCA': [
                    silhouette_avg,
                    davies_bouldin_score(X_pca, clusters_pca),
                    calinski_harabasz_score(X_pca, clusters_pca),
                    ari_score,
                    kmeans.inertia_
                ]
            }
            
            comparison_df = pd.DataFrame(metrics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(comparison_df.round(3))
            
            with col2:
                st.markdown(f"""
                **✅ Avantages PCA:**
                - Réduction: 4D → 2D
                - Visualisation possible
                - Bruit réduit
                
                **❌ Inconvénients PCA:**
                - Perte: {(1-pca.explained_variance_ratio_.sum())*100:.1f}% variance
                - Composantes abstraites
                """)
            
            # Visualisation comparative
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for i, cluster in enumerate(unique_clusters):
                mask = clusters_pca == cluster
                axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
            
            axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                          c='red', marker='X', s=200, linewidths=3)
            axes[0].set_title(f'PCA (Silhouette: {silhouette_avg:.3f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            for i, cluster in enumerate(unique_clusters):
                mask = clusters_orig == cluster
                axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
            
            centroids_orig_proj = pca.transform(kmeans_orig.cluster_centers_)
            axes[1].scatter(centroids_orig_proj[:, 0], centroids_orig_proj[:, 1], 
                          c='red', marker='X', s=200, linewidths=3)
            axes[1].set_title(f'Originales proj. (Silhouette: {silhouette_score(X_scaled, clusters_orig):.3f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

###################### EXERCICE 2 ######################
elif section == "Exercice 2: Clustering Hiérarchique - Iris":
    st.markdown("# Exercice 2: Clustering Hiérarchique")
    st.markdown("---")
    
    iris_data = load_iris_data()
    if iris_data is not None:
        X, y = safe_data_preparation(iris_data, 'species')
        
        if X is not None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            st.success(" Données préparées")
            
            # Question 2
            st.markdown("## 🔗 Question 2: Linkage Complet")
            
            linkage_complete = linkage(X_scaled, method='complete', metric='euclidean')
            agg_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
            clusters_complete = agg_complete.fit_predict(X_scaled)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_dendrogram_custom(linkage_complete, "Linkage Complet")
                st.pyplot(fig)
            
            with col2:
                silhouette_complete = silhouette_score(X_scaled, clusters_complete)
                davies_complete = davies_bouldin_score(X_scaled, clusters_complete)
                calinski_complete = calinski_harabasz_score(X_scaled, clusters_complete)
                
                st.markdown(f"""
                ** Métriques:**
                - **Silhouette**: {silhouette_complete:.3f}
                - **Davies-Bouldin**: {davies_complete:.3f}
                - **Calinski-Harabasz**: {calinski_complete:.3f}
                
                **Distribution:**
                """)
                st.write(pd.Series(clusters_complete).value_counts().sort_index())
            
            st.markdown("---")
            
            # Question 5
            st.markdown("## ⚖️ Question 5: Linkage Moyen vs Complet")
            
            linkage_average = linkage(X_scaled, method='average', metric='euclidean')
            agg_average = AgglomerativeClustering(n_clusters=3, linkage='average')
            clusters_average = agg_average.fit_predict(X_scaled)
            
            silhouette_average = silhouette_score(X_scaled, clusters_average)
            davies_average = davies_bouldin_score(X_scaled, clusters_average)
            calinski_average = calinski_harabasz_score(X_scaled, clusters_average)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                
                dendrogram(linkage_complete, ax=ax1, color_threshold=0.7*max(linkage_complete[:,2]))
                ax1.set_title('Linkage Complet')
                ax1.grid(True, alpha=0.3, axis='y')
                
                dendrogram(linkage_average, ax=ax2, color_threshold=0.7*max(linkage_average[:,2]))
                ax2.set_title('Linkage Moyen')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                comparison_hier = pd.DataFrame({
                    'Métrique': ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'],
                    'Complet': [silhouette_complete, davies_complete, calinski_complete],
                    'Moyen': [silhouette_average, davies_average, calinski_average]
                })
                
                st.dataframe(comparison_hier.round(3))
                
                best = 'Moyen' if silhouette_average > silhouette_complete else 'Complet'
                st.markdown(f"""
                **🏆 Meilleur**: Linkage {best}
                
                **🔗 Complet:**
                - Clusters compacts
                - Moins sensible bruit
                
                **🔗 Moyen:**
                - Compromis équilibré
                - Moins sensible outliers
                """)

###################### EXERCICE 3 ######################
elif section == "Exercice 3: Nombre Optimal de Clusters - Exoplanètes":
    st.markdown("# Exercice 3: Nombre optimal de clusters - Atmosphères d'exoplanètes")
    st.markdown("---")
    
    # Question 1: Chargement des données planètes
    st.markdown("## Question 1: Chargement des données d'exoplanètes")
    
    planete_data = load_planete_data()
    if planete_data is not None:
        st.success("Données d'exoplanètes chargées avec succès")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Aperçu des données")
            st.dataframe(planete_data.head(10))
            
            st.markdown("#### Informations sur le dataset")
            st.write(f"**Dimensions**: {planete_data.shape}")
            st.write(f"**Colonnes**: {list(planete_data.columns)}")
            
        with col2:
            st.markdown("#### Statistiques descriptives")
            st.dataframe(planete_data.describe())
        
        # Suppression de la dernière colonne (labels)
        if 'cluster_true' in planete_data.columns:
            true_labels = planete_data['cluster_true'].copy()
            data_features = planete_data.drop('cluster_true', axis=1)
            st.success("Labels supprimés du dataset")
        else:
            true_labels = planete_data.iloc[:, -1].copy()
            data_features = planete_data.iloc[:, :-1]
            st.success("Dernière colonne supprimée")
        
        st.write(f"**Nouvelles dimensions**: {data_features.shape}")
        
        # Standardisation des données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_features)
        
        st.markdown("---")
        
        # Question 2: Rappel des propriétés des indices
        st.markdown("## Question 2: Propriétés des indices d'évaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Indice de Calinski-Harabasz")
            st.markdown("""
            **Formule:**
            ```
            CH(k) = [B/(k-1)] / [W/(N-k)]
            ```
            
            **Où:**
            - **B**: Somme des carrés entre clusters
            - **W**: Somme des carrés intra-clusters  
            - **k**: Nombre de clusters
            - **N**: Nombre d'observations
            
            **Propriétés:**
            - Plus élevé = Meilleur
            - Favorise les clusters bien séparés
            - Pénalise les clusters trop nombreux
            - Sensible à la forme des clusters
            - Favorise les clusters sphériques
            """)
        
        with col2:
            st.markdown("#### Indice de Davies-Bouldin")
            st.markdown("""
            **Formule:**
            ```
            DB = (1/k) * Σ max[(Si + Sj)/Mij]
            ```
            
            **Où:**
            - **Si**: Dispersion intra-cluster i
            - **Mij**: Distance entre centroids i et j
            - **k**: Nombre de clusters
            
            **Propriétés:**
            - Plus faible = Meilleur
            - Évalue compacité et séparation
            - Facile à interpréter
            - Favorise les clusters sphériques
            - Sensible aux outliers
            """)
        
        st.markdown("---")
        
        # Question 3: Méthode du coude (Elbow method)
        st.markdown("## Question 3: Méthode du coude pour déterminer le nombre optimal")
        
        st.markdown("#### Analyse avec la méthode du coude")
        
        # Plage de nombres de clusters à tester
        k_range = range(2, 11)
        
        # Stockage des métriques
        inertias = []
        calinski_scores = []
        davies_bouldin_scores = []
        silhouette_scores = []
        
        # Calcul pour chaque k
        progress_bar = st.progress(0)
        for i, k in enumerate(k_range):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_scaled)
            
            inertias.append(kmeans.inertia_)
            calinski_scores.append(calinski_harabasz_score(data_scaled, clusters))
            davies_bouldin_scores.append(davies_bouldin_score(data_scaled, clusters))
            silhouette_scores.append(silhouette_score(data_scaled, clusters))
            
            progress_bar.progress((i + 1) / len(k_range))
        
        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Inertie (méthode du coude classique)
        axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Méthode du Coude - Inertie')
        axes[0, 0].set_xlabel('Nombre de clusters (k)')
        axes[0, 0].set_ylabel('Inertie')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Calinski-Harabasz (plus élevé = meilleur)
        axes[0, 1].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Indice de Calinski-Harabasz')
        axes[0, 1].set_xlabel('Nombre de clusters (k)')
        axes[0, 1].set_ylabel('Score Calinski-Harabasz')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Marquer le maximum
        max_calinski_idx = np.argmax(calinski_scores)
        max_calinski_k = k_range[max_calinski_idx]
        axes[0, 1].axvline(x=max_calinski_k, color='red', linestyle='--', 
                          label=f'Optimum: k={max_calinski_k}')
        axes[0, 1].legend()
        
        # 3. Davies-Bouldin (plus faible = meilleur)
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Indice de Davies-Bouldin')
        axes[1, 0].set_xlabel('Nombre de clusters (k)')
        axes[1, 0].set_ylabel('Score Davies-Bouldin')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Marquer le minimum
        min_davies_idx = np.argmin(davies_bouldin_scores)
        min_davies_k = k_range[min_davies_idx]
        axes[1, 0].axvline(x=min_davies_k, color='red', linestyle='--', 
                          label=f'Optimum: k={min_davies_k}')
        axes[1, 0].legend()
        
        # 4. Silhouette
        axes[1, 1].plot(k_range, silhouette_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Score de Silhouette')
        axes[1, 1].set_xlabel('Nombre de clusters (k)')
        axes[1, 1].set_ylabel('Score de Silhouette')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Marquer le maximum
        max_silhouette_idx = np.argmax(silhouette_scores)
        max_silhouette_k = k_range[max_silhouette_idx]
        axes[1, 1].axvline(x=max_silhouette_k, color='red', linestyle='--', 
                          label=f'Optimum: k={max_silhouette_k}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tableau récapitulatif
        st.markdown("#### Tableau récapitulatif des métriques")
        
        metrics_df = pd.DataFrame({
            'k': list(k_range),
            'Inertie': inertias,
            'Calinski-Harabasz': calinski_scores,
            'Davies-Bouldin': davies_bouldin_scores,
            'Silhouette': silhouette_scores
        }).round(3)
        
        st.dataframe(metrics_df)
        
        # Analyse des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Recommandations par indice")
            st.markdown(f"""
            **Optimums détectés:**
            - **Calinski-Harabasz**: k = {max_calinski_k}
            - **Davies-Bouldin**: k = {min_davies_k}
            - **Silhouette**: k = {max_silhouette_k}
            
            **Méthode du coude (Inertie):**
            - Rechercher le "coude" dans la courbe
            - Point où la diminution devient moins prononcée
            """)
            
            # Calcul automatique du coude pour l'inertie
            if len(inertias) >= 3:
                second_derivatives = []
                for i in range(1, len(inertias)-1):
                    second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_derivatives.append(second_deriv)
                
                elbow_idx = np.argmax(second_derivatives) + 1
                elbow_k = k_range[elbow_idx]
                st.markdown(f"- **Coude détecté automatiquement**: k = {elbow_k}")
        
        with col2:
            st.markdown("#### Explication de la méthode du coude")
            st.markdown("""
            **Principe de la méthode du coude:**
            
            1. **Objectif**: Trouver le nombre optimal de clusters
            2. **Méthode**: Analyser l'évolution des métriques
            3. **Critère**: Identifier le point d'inflexion
            
            **Comment ça fonctionne:**
            - **Trop peu de clusters**: Métriques sous-optimales
            - **Trop de clusters**: Amélioration marginale
            - **Optimum**: Point où l'amélioration devient négligeable
            
            **Interprétation:**
            - **Coude net**: Nombre optimal clair
            - **Coude flou**: Plusieurs solutions possibles
            - **Pas de coude**: Données sans structure claire
            """)
        
        # Consensus et recommandation finale
        st.markdown("#### Recommandation finale")
        
        # Calcul du consensus
        recommendations = [max_calinski_k, min_davies_k, max_silhouette_k]
        if len(inertias) >= 3:
            recommendations.append(elbow_k)
        
        # Trouver la valeur la plus fréquente
        from collections import Counter
        consensus = Counter(recommendations).most_common(1)[0]
        
        st.markdown(f"""
        **Analyse de consensus:**
        - **Recommandations**: {recommendations}
        - **Consensus**: k = {consensus[0]} (recommandé {consensus[1]} fois)
        
        **Validation avec les vrais labels** (si disponibles):
        """)
        
        if len(np.unique(true_labels)) > 1:
            true_k = len(np.unique(true_labels))
            st.markdown(f"- **Nombre réel de clusters**: {true_k}")
            
            if consensus[0] == true_k:
                st.success(f"Le consensus (k={consensus[0]}) correspond au nombre réel!")
            else:
                st.warning(f"Le consensus (k={consensus[0]}) diffère du nombre réel (k={true_k})")
            
            # Test avec le nombre réel
            kmeans_true = KMeans(n_clusters=true_k, random_state=42)
            clusters_true_k = kmeans_true.fit_predict(data_scaled)
            ari_true = adjusted_rand_score(true_labels, clusters_true_k)
            
            st.markdown(f"**Performance avec k={true_k}**: ARI = {ari_true:.3f}")

###################### EXERCICE 4: PIPELINE COMPLET ######################
elif section == "Exercice 4: Pipeline Complet d'Analyse":
    st.markdown("# Exercice 4: Pipeline Complet d'Analyse de Données")
    st.markdown("---")
    
    st.markdown("""
    ## Instructions du Pipeline
    
    Ce pipeline complet comprend:
    1. **Prétraitement et Transformation**
    2. **Analyse Exploratoire et Visualisation**  
    3. **Modélisation Non Supervisée**
    4. **Modélisation Supervisée**
    5. **Évaluation et Comparaison**
    """)
    
    # Sélection du dataset
    st.markdown("## Sélection du Dataset")
    dataset_choice = st.selectbox(
        "Choisir un dataset pour l'analyse complète",
        ["Iris (exemple)", "Wine Dataset", "Breast Cancer Wisconsin", "Dataset personnalisé"]
    )
    
    # Chargement du dataset selon le choix
    df = None
    target_names = None
    
    if dataset_choice == "Iris (exemple)":
        from sklearn.datasets import load_iris
        data_sklearn = load_iris()
        df = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
        df['target'] = data_sklearn.target
        target_names = data_sklearn.target_names
        
    elif dataset_choice == "Wine Dataset":
        from sklearn.datasets import load_wine
        data_sklearn = load_wine()
        df = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
        df['target'] = data_sklearn.target
        target_names = data_sklearn.target_names
        
    elif dataset_choice == "Breast Cancer Wisconsin":
        from sklearn.datasets import load_breast_cancer
        data_sklearn = load_breast_cancer()
        df = pd.DataFrame(data_sklearn.data, columns=data_sklearn.feature_names)
        df['target'] = data_sklearn.target
        target_names = data_sklearn.target_names
        
    else:
        st.info("Pour un dataset personnalisé, uploadez un fichier CSV")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset personnalisé chargé")
        else:
            df = load_iris_data()
            if df is not None:
                target_names = df['species'].unique()
    
    if df is not None:
        st.success(f"Dataset '{dataset_choice}' chargé avec succès")
        
        # Section 1: Prétraitement
        st.markdown("## Section 1: Prétraitement et Transformation des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Caractéristiques du dataset")
            st.write(f"**Dimensions**: {df.shape}")
            st.write(f"**Colonnes**: {len(df.columns)}")
            st.write(f"**Types de données**:")
            st.write(df.dtypes.value_counts())
            
            # Identification de la colonne cible
            if 'target' in df.columns:
                target_col = 'target'
            elif 'species' in df.columns:
                target_col = 'species'
            else:
                target_col = st.selectbox("Sélectionner la colonne cible", df.columns)
            
            st.write(f"**Colonne cible**: {target_col}")
            
        with col2:
            st.markdown("#### Aperçu des données")
            st.dataframe(df.head())
            
            st.markdown("#### Valeurs manquantes")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Colonne': missing_data.index,
                'Valeurs manquantes': missing_data.values,
                'Pourcentage': missing_percent.values
            })
            missing_df = missing_df[missing_df['Valeurs manquantes'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df)
                st.warning("Valeurs manquantes détectées")
            else:
                st.success("Aucune valeur manquante")
        
        # Statistiques descriptives
        st.markdown("#### Statistiques descriptives")
        st.dataframe(df.describe())
        
        # Séparation features/target
        y = df[target_col].copy()
        X = df.drop(target_col, axis=1).copy()
        
        # Encodage des variables catégorielles si nécessaire
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            le_dict = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                le_dict[col] = le
        
        # Standardisation
        standardization = st.checkbox("Appliquer la standardisation (recommandé pour le clustering)", value=True)
        
        if standardization:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_for_clustering = X_scaled
            st.success("Standardisation appliquée")
        else:
            X_for_clustering = X.values
        
        st.markdown("---")
        
        # Section 2: Analyse Exploratoire
        st.markdown("## Section 2: Analyse Exploratoire et Visualisation")
        
        # Distribution des variables
        st.markdown("#### Distribution des variables")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(X[col], bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution de {col}')
                    axes[i].grid(True, alpha=0.3)
            
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Matrice de corrélation
        if len(numeric_cols) > 1:
            st.markdown("#### Matrice de corrélation")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = X[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
            ax.set_title('Matrice de Corrélation')
            st.pyplot(fig)
        
        # Distribution de la variable cible
        st.markdown("#### Distribution de la variable cible")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        target_counts = y.value_counts()
        
        if len(target_counts) <= 10:
            bars = ax.bar(range(len(target_counts)), target_counts.values)
            ax.set_xticks(range(len(target_counts)))
            ax.set_xticklabels(target_counts.index, rotation=45)
            ax.set_title('Distribution de la Variable Cible')
            ax.set_ylabel('Fréquence')
            
            for bar, count in zip(bars, target_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
        else:
            ax.hist(y, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title('Distribution de la Variable Cible')
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Fréquence')
        
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Section 3: Modélisation Non Supervisée
        st.markdown("## Section 3: Modélisation Non Supervisée")
        
        unsupervised_algos = st.multiselect(
            "Sélectionner les algorithmes de clustering",
            ["K-Means", "Clustering Hiérarchique", "DBSCAN"],
            default=["K-Means", "Clustering Hiérarchique"]
        )
        
        if len(unsupervised_algos) > 0:
            n_true_clusters = len(np.unique(y))
            st.info(f"Nombre de classes réelles: {n_true_clusters}")
            
            clustering_results = {}
            
            # K-Means
            if "K-Means" in unsupervised_algos:
                st.markdown("#### K-Means Clustering")
                
                kmeans = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X_for_clustering)
                
                silhouette_kmeans = silhouette_score(X_for_clustering, kmeans_labels)
                davies_bouldin_kmeans = davies_bouldin_score(X_for_clustering, kmeans_labels)
                calinski_kmeans = calinski_harabasz_score(X_for_clustering, kmeans_labels)
                
                clustering_results['K-Means'] = {
                    'labels': kmeans_labels,
                    'silhouette': silhouette_kmeans,
                    'davies_bouldin': davies_bouldin_kmeans,
                    'calinski': calinski_kmeans
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_kmeans:.3f}")
                    st.metric("Davies-Bouldin", f"{davies_bouldin_kmeans:.3f}")
                    st.metric("Calinski-Harabasz", f"{calinski_kmeans:.3f}")
                
                with col2:
                    if X_for_clustering.shape[1] > 2:
                        pca_viz = PCA(n_components=2)
                        X_pca_viz = pca_viz.fit_transform(X_for_clustering)
                    else:
                        X_pca_viz = X_for_clustering
                    
                    centroids_2d = kmeans.cluster_centers_[:, :2] if X_for_clustering.shape[1] == 2 else pca_viz.transform(kmeans.cluster_centers_)
                    fig = plot_clusters_2d(X_pca_viz, kmeans_labels, centroids_2d, "K-Means Clustering")
                    st.pyplot(fig)
            
            # Clustering Hiérarchique
            if "Clustering Hiérarchique" in unsupervised_algos:
                st.markdown("#### Clustering Hiérarchique")
                
                agg_clustering = AgglomerativeClustering(n_clusters=n_true_clusters, linkage='ward')
                agg_labels = agg_clustering.fit_predict(X_for_clustering)
                
                silhouette_agg = silhouette_score(X_for_clustering, agg_labels)
                davies_bouldin_agg = davies_bouldin_score(X_for_clustering, agg_labels)
                calinski_agg = calinski_harabasz_score(X_for_clustering, agg_labels)
                
                clustering_results['Hiérarchique'] = {
                    'labels': agg_labels,
                    'silhouette': silhouette_agg,
                    'davies_bouldin': davies_bouldin_agg,
                    'calinski': calinski_agg
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette_agg:.3f}")
                    st.metric("Davies-Bouldin", f"{davies_bouldin_agg:.3f}")
                    st.metric("Calinski-Harabasz", f"{calinski_agg:.3f}")
                
                with col2:
                    if len(X_for_clustering) <= 100:
                        linkage_matrix = linkage(X_for_clustering, method='ward')
                        fig = plot_dendrogram(linkage_matrix, "Dendrogramme - Ward Linkage")
                        st.pyplot(fig)
                    else:
                        if X_for_clustering.shape[1] > 2:
                            pca_viz = PCA(n_components=2)
                            X_pca_viz = pca_viz.fit_transform(X_for_clustering)
                        else:
                            X_pca_viz = X_for_clustering
                        
                        fig = plot_clusters_2d(X_pca_viz, agg_labels, title="Clustering Hiérarchique")
                        st.pyplot(fig)
            
            # DBSCAN
            if "DBSCAN" in unsupervised_algos:
                st.markdown("#### DBSCAN Clustering")
                
                from sklearn.cluster import DBSCAN
                from sklearn.neighbors import NearestNeighbors
                
                k = 4
                nbrs = NearestNeighbors(n_neighbors=k).fit(X_for_clustering)
                distances, indices = nbrs.kneighbors(X_for_clustering)
                distances = np.sort(distances[:, k-1], axis=0)
                
                eps_estimate = np.percentile(distances, 95)
                
                dbscan = DBSCAN(eps=eps_estimate, min_samples=k)
                dbscan_labels = dbscan.fit_predict(X_for_clustering)
                
                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clusters détectés", n_clusters_dbscan)
                    st.metric("Points de bruit", n_noise)
                    st.metric("Eps utilisé", f"{eps_estimate:.3f}")
                    
                    if n_clusters_dbscan > 1:
                        valid_mask = dbscan_labels != -1
                        if np.sum(valid_mask) > 0:
                            silhouette_dbscan = silhouette_score(X_for_clustering[valid_mask], dbscan_labels[valid_mask])
                            st.metric("Silhouette Score", f"{silhouette_dbscan:.3f}")
                            
                            clustering_results['DBSCAN'] = {
                                'labels': dbscan_labels,
                                'silhouette': silhouette_dbscan,
                                'n_clusters': n_clusters_dbscan,
                                'n_noise': n_noise
                            }
                
                with col2:
                    if X_for_clustering.shape[1] > 2:
                        pca_viz = PCA(n_components=2)
                        X_pca_viz = pca_viz.fit_transform(X_for_clustering)
                    else:
                        X_pca_viz = X_for_clustering
                    
                    fig = plot_clusters_2d(X_pca_viz, dbscan_labels, title="DBSCAN Clustering")
                    st.pyplot(fig)
            
            # Comparaison
            if len(clustering_results) > 1:
                st.markdown("#### Comparaison des Algorithmes Non Supervisés")
                
                comparison_data = []
                for algo, results in clustering_results.items():
                    row = {
                        'Algorithme': algo,
                        'Silhouette': results.get('silhouette', 'N/A')
                    }
                    if 'davies_bouldin' in results:
                        row['Davies-Bouldin'] = results['davies_bouldin']
                    if 'calinski' in results:
                        row['Calinski-Harabasz'] = results['calinski']
                    if 'n_clusters' in results:
                        row['Clusters détectés'] = results['n_clusters']
                    
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Validation externe
                st.markdown("#### Validation Externe (ARI avec vrais labels)")
                
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                
                ari_results = []
                for algo, results in clustering_results.items():
                    if 'labels' in results:
                        ari = adjusted_rand_score(y_encoded, results['labels'])
                        ari_results.append({'Algorithme': algo, 'ARI': ari})
                
                ari_df = pd.DataFrame(ari_results)
                st.dataframe(ari_df)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(ari_df['Algorithme'], ari_df['ARI'])
                ax.set_title('Adjusted Rand Index - Comparaison des Algorithmes')
                ax.set_ylabel('ARI')
                ax.set_ylim(0, 1)
                
                for bar, ari in zip(bars, ari_df['ARI']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{ari:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Section 4: Modélisation Supervisée
        st.markdown("## Section 4: Modélisation Supervisée")
        
        supervised_algos = st.multiselect(
            "Sélectionner les algorithmes supervisés",
            ["K-NN", "Arbre de Décision", "Régression Logistique", "Naive Bayes"],
            default=["K-NN", "Arbre de Décision"]
        )
        
        if len(supervised_algos) > 0:
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.metrics import classification_report, f1_score
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_for_clustering, y, test_size=0.3, random_state=42, stratify=y
            )
            
            supervised_results = {}
            
            # K-NN
            if "K-NN" in supervised_algos:
                st.markdown("#### K-Nearest Neighbors")
                
                k_values = range(1, min(21, len(X_train)//2))
                best_k = 5
                best_score = 0
                
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)
                    score = knn.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                knn_best = KNeighborsClassifier(n_neighbors=best_k)
                knn_best.fit(X_train, y_train)
                knn_pred = knn_best.predict(X_test)
                
                knn_accuracy = accuracy_score(y_test, knn_pred)
                knn_f1 = f1_score(y_test, knn_pred, average='weighted')
                
                supervised_results['K-NN'] = {
                    'accuracy': knn_accuracy,
                    'f1': knn_f1,
                    'predictions': knn_pred,
                    'best_params': f'k={best_k}'
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{knn_accuracy:.3f}")
                    st.metric("F1-Score", f"{knn_f1:.3f}")
                    st.metric("Meilleur k", best_k)
                
                with col2:
                    cm = confusion_matrix(y_test, knn_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Matrice de Confusion - K-NN')
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Valeurs Réelles')
                    st.pyplot(fig)
            
            # Arbre de Décision
            if "Arbre de Décision" in supervised_algos:
                st.markdown("#### Arbre de Décision")
                
                dt = DecisionTreeClassifier(random_state=42, max_depth=5)
                dt.fit(X_train, y_train)
                dt_pred = dt.predict(X_test)
                
                dt_accuracy = accuracy_score(y_test, dt_pred)
                dt_f1 = f1_score(y_test, dt_pred, average='weighted')
                
                supervised_results['Arbre de Décision'] = {
                    'accuracy': dt_accuracy,
                    'f1': dt_f1,
                    'predictions': dt_pred,
                    'feature_importance': dt.feature_importances_
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{dt_accuracy:.3f}")
                    st.metric("F1-Score", f"{dt_f1:.3f}")
                    
                    if hasattr(X, 'columns'):
                        feature_names = X.columns
                    else:
                        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': dt.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("**Top 5 Features importantes:**")
                    st.dataframe(importance_df.head())
                
                with col2:
                    cm = confusion_matrix(y_test, dt_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                    ax.set_title('Matrice de Confusion - Arbre de Décision')
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Valeurs Réelles')
                    st.pyplot(fig)
            
            # Régression Logistique
            if "Régression Logistique" in supervised_algos:
                st.markdown("#### Régression Logistique")
                
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)
                
                lr_accuracy = accuracy_score(y_test, lr_pred)
                lr_f1 = f1_score(y_test, lr_pred, average='weighted')
                
                supervised_results['Régression Logistique'] = {
                    'accuracy': lr_accuracy,
                    'f1': lr_f1,
                    'predictions': lr_pred
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{lr_accuracy:.3f}")
                    st.metric("F1-Score", f"{lr_f1:.3f}")
                
                with col2:
                    cm = confusion_matrix(y_test, lr_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
                    ax.set_title('Matrice de Confusion - Régression Logistique')
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Valeurs Réelles')
                    st.pyplot(fig)
            
            # Naive Bayes
            if "Naive Bayes" in supervised_algos:
                st.markdown("#### Naive Bayes")
                
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                nb_pred = nb.predict(X_test)
                
                nb_accuracy = accuracy_score(y_test, nb_pred)
                nb_f1 = f1_score(y_test, nb_pred, average='weighted')
                
                supervised_results['Naive Bayes'] = {
                    'accuracy': nb_accuracy,
                    'f1': nb_f1,
                    'predictions': nb_pred
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{nb_accuracy:.3f}")
                    st.metric("F1-Score", f"{nb_f1:.3f}")
                
                with col2:
                    cm = confusion_matrix(y_test, nb_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
                    ax.set_title('Matrice de Confusion - Naive Bayes')
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Valeurs Réelles')
                    st.pyplot(fig)
            
            # Section 5: Comparaison
            st.markdown("## Section 5: Comparaison et Analyse Finale")
            
            if len(supervised_results) > 1:
                st.markdown("#### Comparaison des Algorithmes Supervisés")
                
                comparison_supervised = []
                for algo, results in supervised_results.items():
                    comparison_supervised.append({
                        'Algorithme': algo,
                        'Accuracy': results['accuracy'],
                        'F1-Score': results['f1'],
                        'Paramètres': results.get('best_params', 'Défaut')
                    })
                
                comparison_sup_df = pd.DataFrame(comparison_supervised)
                st.dataframe(comparison_sup_df.round(3))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                bars1 = ax1.bar(comparison_sup_df['Algorithme'], comparison_sup_df['Accuracy'])
                ax1.set_title('Comparaison des Accuracy')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1)
                ax1.tick_params(axis='x', rotation=45)
                
                for bar, acc in zip(bars1, comparison_sup_df['Accuracy']):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
                
                bars2 = ax2.bar(comparison_sup_df['Algorithme'], comparison_sup_df['F1-Score'])
                ax2.set_title('Comparaison des F1-Scores')
                ax2.set_ylabel('F1-Score')
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, f1 in zip(bars2, comparison_sup_df['F1-Score']):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{f1:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                best_algo = comparison_sup_df.loc[comparison_sup_df['F1-Score'].idxmax(), 'Algorithme']
                best_f1 = comparison_sup_df['F1-Score'].max()
                
                st.success(f"Meilleur algorithme: {best_algo} (F1-Score: {best_f1:.3f})")
            
            # Rapport détaillé
            if len(supervised_results) > 0:
                st.markdown("#### Rapport Détaillé du Meilleur Algorithme")
                
                best_results = supervised_results[best_algo]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Algorithme**: {best_algo}")
                    st.text("Rapport de Classification:")
                    st.text(classification_report(y_test, best_results['predictions']))
                
                with col2:
                    errors = y_test != best_results['predictions']
                    error_rate = errors.mean()
                    
                    st.markdown(f"""
                    **Analyse des Performances:**
                    - **Taux d'erreur**: {error_rate:.3f}
                    - **Nombre d'erreurs**: {errors.sum()}/{len(y_test)}
                    - **Accuracy**: {best_results['accuracy']:.3f}
                    - **F1-Score**: {best_results['f1']:.3f}
                    """)
        
        # Compte-rendu final
        st.markdown("## Compte-rendu Final")
        
        with st.expander("Résumé de l'Analyse Complète"):
            st.markdown(f"""
            ### Caractéristiques du Dataset
            - **Dataset**: {dataset_choice}
            - **Dimensions**: {df.shape}
            - **Classes**: {len(np.unique(y))}
            - **Features**: {X.shape[1]}
            
            ### Prétraitement Appliqué
            - **Encodage**: {'Appliqué' if len(categorical_cols) > 0 else 'Non nécessaire'}
            - **Standardisation**: {'Oui' if standardization else 'Non'}
            
            ### Résultats du Clustering (Non Supervisé)
            """)
            
            if 'clustering_results' in locals() and len(clustering_results) > 0:
                for algo, results in clustering_results.items():
                    st.markdown(f"- **{algo}**: Silhouette = {results.get('silhouette', 'N/A')}")
            
            st.markdown("### Résultats de la Classification (Supervisé)")
            
            if 'supervised_results' in locals() and len(supervised_results) > 0:
                for algo, results in supervised_results.items():
                    st.markdown(f"- **{algo}**: Accuracy = {results['accuracy']:.3f}, F1 = {results['f1']:.3f}")
            
            st.markdown("""
            ### Conclusions et Recommandations
            
            **Pour le clustering non supervisé:**
            - Les algorithmes basés sur la distance fonctionnent bien sur des données standardisées
            - DBSCAN est utile pour détecter des formes non-sphériques et des outliers
            - La validation externe (ARI) permet de comparer avec les vraies classes
            
            **Pour la classification supervisée:**
            - Les performances dépendent de la qualité des features et du prétraitement
            - L'optimisation des hyperparamètres améliore significativement les résultats
            - La matrice de confusion révèle les classes les plus difficiles à distinguer
            
            **Recommandations générales:**
            - Toujours standardiser les données pour les algorithmes basés sur la distance
            - Utiliser plusieurs métriques d'évaluation pour une analyse complète
            - Considérer le contexte métier pour choisir entre précision et rappel
            - Valider les résultats avec des experts du domaine
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Machine Learning I - TP4 Classification Non Supervisée</strong></p>
    <p><em>Concepts appliqués: K-Means, Clustering Hiérarchique, DBSCAN, Indices d'évaluation</em></p>
    <p>Issam Falih - Department of Computer Science | Développé avec Streamlit</p>
    <p>Méthodes: Elbow Method, Silhouette Analysis, Davies-Bouldin, Calinski-Harabasz</p>
</div>
""", unsafe_allow_html=True)

# Sidebar avec aide contextuelle
st.sidebar.markdown("---")
st.sidebar.markdown("### Aide Contextuelle")

if st.sidebar.button("Algorithmes de Clustering"):
    st.sidebar.markdown("""
    **K-Means:**
    - Prototype-based
    - Clusters sphériques
    - Nécessite k prédéfini
    
    **Hiérarchique:**
    - Dendrogramme
    - Pas de k prédéfini
    - Différents linkages
    
    **DBSCAN:**
    - Density-based
    - Détecte les outliers
    - Formes arbitraires
    """)

if st.sidebar.button("Métriques d'Évaluation"):
    st.sidebar.markdown("""
    **Silhouette Score:**
    - [-1, 1], plus élevé = meilleur
    - Mesure cohésion/séparation
    
    **Davies-Bouldin:**
    - [0, inf], plus faible = meilleur
    - Ratio compacité/séparation
    
    **Calinski-Harabasz:**
    - [0, inf], plus élevé = meilleur
    - Variance inter/intra clusters
    """)

if st.sidebar.button("Méthode du Coude"):
    st.sidebar.markdown("""
    **Principe:**
    1. Tester plusieurs valeurs de k
    2. Calculer les métriques
    3. Chercher le "coude"
    4. Point d'inflexion optimal
    
    **Interprétation:**
    - Coude net = k optimal clair
    - Coude flou = plusieurs solutions
    - Pas de coude = pas de structure
    """)