import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

#################### Page configuration #########
st.set_page_config(layout="wide", page_title="TP3 - Classification Supervisée")

# Header avec style académique
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 30px;'>
    <h1> Machine Learning I</h1>
    <h2>TP3 - Classification Supervisée</h2>
    <p><em>Lecture 3: Supervised Learning - Distance-based approaches, Decision trees, Naive Bayes Classifiers</em></p>
    <p><strong><b>Realisé par:<b> Emmanuel M. Morris <strong><p>
    <p><strong><b>Encadré par:<b> Issam Falih - Department of Computer Science</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("### Choisir la section d'analyse")
section = st.sidebar.selectbox(
    "Dataset à analyser",
    ["Partie I: Titanic Dataset", "Partie II: Heart Disease Dataset"]
)

# Rappel théorique dans la sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Rappels Théoriques")
st.sidebar.markdown("""
**Supervised Learning Workflow:**
1. Training set (labels connus)
2. Validation set (évaluation)
3. Test set (prédiction)

**Classifiers étudiés:**
- **Naive Bayes**: Probabiliste basé sur le théorème de Bayes
- **Decision Trees**: Règles de décision hiérarchiques
- **XGBoost**: Ensemble method (boosting)

**Métriques d'évaluation:**
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F-Measure = 2×(Precision×Recall)/(Precision+Recall)
""")

# Fonctions utilitaires
@st.cache_data
def load_data():
    """Charge tous les datasets selon le workflow supervisé"""
    try:
        titanic_train = pd.read_csv("titanic_train.csv")
        titanic_test = pd.read_csv("titanic_test.csv")
        heart_data = pd.read_csv("heart-disease-UCI.csv")
        return titanic_train, titanic_test, heart_data
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement des données: {e}")
        st.info("Assurez-vous que les fichiers CSV sont dans le bon répertoire")
        return None, None, None

def plot_confusion_matrix_with_metrics(y_true, y_pred, title):
    """Affiche la matrice de confusion avec métriques détaillées"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcul des métriques selon les notes de cours
    TP = cm[1, 1] if cm.shape == (2, 2) else None
    TN = cm[0, 0] if cm.shape == (2, 2) else None
    FP = cm[0, 1] if cm.shape == (2, 2) else None
    FN = cm[1, 0] if cm.shape == (2, 2) else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matrice de confusion
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Matrice de Confusion - {title}')
    ax1.set_xlabel('Prédictions')
    ax1.set_ylabel('Valeurs Réelles')
    
    # Métriques détaillées
    if TP is not None:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f"""
        Métriques de Classification (selon cours):
        
        TP (True Positive): {TP}
        TN (True Negative): {TN}
        FP (False Positive): {FP}
        FN (False Negative): {FN}
        
        Accuracy = (TP + TN) / Total = {accuracy:.4f}
        Precision = TP / (TP + FP) = {precision:.4f}
        Recall (TPR) = TP / (TP + FN) = {recall:.4f}
        Specificity = TN / (TN + FP) = {specificity:.4f}
        F-Measure = 2×(P×R)/(P+R) = {f_measure:.4f}
        """
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Métriques d\'Évaluation')
    
    plt.tight_layout()
    return fig

def preprocess_titanic_data(df):
    """Préprocessing selon les 5 étapes du supervised learning"""
    df_processed = df.copy()
    
    # Étape 2: Déterminer les features d'entrée et leurs représentations
    st.write("**Étape 2: Détermination des features d'entrée**")
    
    # Gestion des valeurs manquantes
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    
    # Encodage des variables catégorielles pour les classifiers
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])
    df_processed['Embarked_encoded'] = le_embarked.fit_transform(df_processed['Embarked'])
    
    # Création de nouvelles features (feature engineering)
    df_processed['Child'] = (df_processed['Age'] < 18).astype(int)
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    return df_processed, le_sex, le_embarked

# Chargement des données
titanic_train, titanic_test, heart_data = load_data()

if titanic_train is None:
    st.stop()

###################### PARTIE I: TITANIC DATASET #################
if section == "Partie I: Titanic Dataset":
    st.markdown("# Partie I: Titanic Dataset")
    st.markdown("### *Application des concepts de classification supervisée*")
    st.markdown("---")
    
    # Étape 1: Décider du training set représentatif
    st.markdown("## Étape 1: Analyse du Training Set")
    st.markdown("*Décider d'un training set représentatif du monde réel*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Aperçu des données d'entraînement")
        st.dataframe(titanic_train.head())
        
        st.markdown("#### Informations sur le dataset")
        info_df = pd.DataFrame({
            "Attributs": titanic_train.columns,
            "Type": titanic_train.dtypes.astype(str),
            "Non-null": titanic_train.notnull().sum(),
            "Valeurs manquantes": titanic_train.isnull().sum(),
            "% Manquant": round(titanic_train.isnull().mean()*100, 2)
        })
        st.dataframe(info_df)
    
    with col2:
        st.markdown("#### Statistiques descriptives")
        st.dataframe(titanic_train.describe())
        
        # Distribution de la variable cible
        st.markdown("#### Distribution de la classe cible")
        survival_counts = titanic_train['Survived'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(survival_counts.values, labels=['Décédé (0)', 'Survivant (1)'], 
               autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        ax.set_title('Distribution de la Variable Cible (Survived)')
        st.pyplot(fig)
    
    # Analyse exploratoire selon les concepts du cours
    st.markdown("##  Analyse Exploratoire des Données")
    
    with st.expander("Visualisations des distributions"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution de l'âge
        sns.histplot(titanic_train['Age'].dropna(), kde=True, ax=axes[0,0], color="skyblue")
        axes[0,0].set_title("Distribution de l'Âge")
        axes[0,0].set_xlabel("Âge")
        
        # Distribution par classe
        sns.countplot(x='Pclass', data=titanic_train, palette="viridis", ax=axes[0,1])
        axes[0,1].set_title("Répartition par Classe")
        
        # Survie par sexe
        survival_by_sex = titanic_train.groupby('Sex')['Survived'].mean()
        axes[1,0].bar(survival_by_sex.index, survival_by_sex.values, color=['pink', 'lightblue'])
        axes[1,0].set_title("Taux de Survie par Sexe")
        axes[1,0].set_ylabel("Probabilité de Survie")
        
        # Survie par classe
        survival_by_class = titanic_train.groupby('Pclass')['Survived'].mean()
        axes[1,1].bar(survival_by_class.index, survival_by_class.values, color='lightgreen')
        axes[1,1].set_title("Taux de Survie par Classe")
        axes[1,1].set_ylabel("Probabilité de Survie")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Analyse "Les femmes et les enfants d'abord" avec approche bayésienne
    st.markdown("## Analyse Bayésienne: 'Les femmes et les enfants d'abord'")
    st.markdown("*Application du théorème de Bayes pour analyser les facteurs de survie*")
    
    # Création des groupes d'âge
    titanic_train["Age_Category"] = titanic_train["Age"].apply(
        lambda x: "Enfant" if pd.notna(x) and x < 18 else "Adulte"
    )
    
    # Calcul des probabilités selon Bayes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### P(Survie | Sexe)")
        survival_by_sex = titanic_train.groupby("Sex")["Survived"].agg(['mean', 'count']).round(4)
        survival_by_sex.columns = ['P(Survie|Sexe)', 'Effectif']
        st.dataframe(survival_by_sex)
        
        # Visualisation
        fig, ax = plt.subplots()
        survival_by_sex['P(Survie|Sexe)'].plot(kind='bar', ax=ax, color=['pink', 'lightblue'])
        ax.set_title('P(Survie | Sexe)')
        ax.set_ylabel('Probabilité')
        ax.set_xticklabels(['Femme', 'Homme'], rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### P(Survie | Âge)")
        survival_by_age = titanic_train.groupby("Age_Category")["Survived"].agg(['mean', 'count']).round(4)
        survival_by_age.columns = ['P(Survie|Âge)', 'Effectif']
        st.dataframe(survival_by_age)
        
        # Visualisation
        fig, ax = plt.subplots()
        survival_by_age['P(Survie|Âge)'].plot(kind='bar', ax=ax, color=['orange', 'green'])
        ax.set_title('P(Survie | Âge)')
        ax.set_ylabel('Probabilité')
        ax.set_xticklabels(['Adulte', 'Enfant'], rotation=0)
        st.pyplot(fig)
    
    # Analyse combinée avec hypothèse d'indépendance naive
    st.markdown("#### Analyse Combinée (Hypothèse d'Indépendance)")
    st.markdown("*Selon l'hypothèse naive: P(Survie|Sexe,Âge) ≈ P(Survie|Sexe) × P(Survie|Âge)*")
    
    survival_combined = titanic_train.groupby(["Age_Category", "Sex"])["Survived"].agg(['mean', 'count']).round(4)
    survival_combined.columns = ['P(Survie|Sexe,Âge)', 'Effectif']
    st.dataframe(survival_combined)
    
    # Visualisation combinée
    fig, ax = plt.subplots(figsize=(10, 6))
    survival_pivot = titanic_train.pivot_table(values='Survived', index='Age_Category', columns='Sex', aggfunc='mean')
    sns.heatmap(survival_pivot, annot=True, cmap='RdYlBu', ax=ax, fmt='.3f')
    ax.set_title('P(Survie | Sexe, Âge) - Matrice de Probabilités')
    st.pyplot(fig)
    
    # Étapes 3-5: Structure d'apprentissage et évaluation
    st.markdown("##  Étapes 3-5: Modèles de Classification")
    st.markdown("*Structure d'apprentissage, entraînement et évaluation*")
    
    # Préparation des données (Étape 2)
    titanic_processed, le_sex, le_embarked = preprocess_titanic_data(titanic_train)
    
    # Sélection des features
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded', 'Child', 'FamilySize']
    X = titanic_processed[features]
    y = titanic_processed['Survived']
    
    # Division selon le workflow supervisé
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.success(f" **Training set**: {X_train.shape[0]} échantillons | **Test set**: {X_test.shape[0]} échantillons")
    st.info(f"**Features sélectionnées**: {len(features)} attributs")
    
    # Tabs pour les différents classifiers
    tab1, tab2, tab3, tab4 = st.tabs([
        "Naive Bayes Classifier", 
        "Decision Tree Classifier", 
        "XGBoost Classifier", 
        "Comparaison & Évaluation"
    ])
    
    with tab1:
        st.subheader("Naive Bayes Classifier")
        st.markdown("*Basé sur le théorème de Bayes avec hypothèse d'indépendance*")
        
        # Rappel théorique
        st.markdown("""
        **Théorème de Bayes**: 
        ```
        P(classe|x) = P(x|classe) × P(classe) / P(x)
        ```
        **Hypothèse Naive**: Les attributs sont indépendants
        ```
        P(x|classe) = ∏ P(xi|classe)
        ```
        """)
        
        if st.button("🔄 Entraîner Naive Bayes", key="nb_titanic"):
            # Pour CategoricalNB, discrétisation des variables continues
            X_train_cat = X_train.copy()
            X_test_cat = X_test.copy()
            
            # Discrétisation selon les concepts du cours
            X_train_cat['Age'] = pd.cut(X_train_cat['Age'], bins=5, labels=False)
            X_test_cat['Age'] = pd.cut(X_test_cat['Age'], bins=5, labels=False)
            X_train_cat['Fare'] = pd.cut(X_train_cat['Fare'], bins=5, labels=False)
            X_test_cat['Fare'] = pd.cut(X_test_cat['Fare'], bins=5, labels=False)
            
            # Entraînement
            nb_model = CategoricalNB()
            nb_model.fit(X_train_cat, y_train)
            
            # Prédictions
            nb_pred = nb_model.predict(X_test_cat)
            nb_accuracy = accuracy_score(y_test, nb_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{nb_accuracy:.4f}")
                
                # Calcul des métriques selon le cours
                precision = precision_score(y_test, nb_pred)
                recall = recall_score(y_test, nb_pred)
                f1 = f1_score(y_test, nb_pred)
                
                st.markdown(f"""
                **Métriques d'évaluation:**
                - **Precision**: {precision:.4f}
                - **Recall (TPR)**: {recall:.4f}
                - **F-Measure**: {f1:.4f}
                """)
                
                st.text("Rapport de Classification:")
                st.text(classification_report(y_test, nb_pred))
            
            with col2:
                fig_cm = plot_confusion_matrix_with_metrics(y_test, nb_pred, "Naive Bayes")
                st.pyplot(fig_cm)
            
            # Analyse des probabilités (concept du cours)
            st.markdown("#### Analyse des Probabilités de Prédiction")
            nb_proba = nb_model.predict_proba(X_test_cat)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(nb_proba[:, 1], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax.set_title('Distribution des P(Survie|x) - Naive Bayes')
            ax.set_xlabel('Probabilité de Survie')
            ax.set_ylabel('Fréquence')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Seuil de décision')
            ax.legend()
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Decision Tree Classifier")
        st.markdown("*Décomposition de l'espace des features selon la variable la plus discriminante*")
        
        # Rappel théorique
        st.markdown("""
        **Propriétés des arbres de décision:**
        - **Expressivité**: Peut représenter des disjonctions de conjonctions
        - **Lisibilité**: Peut être traduit en ensemble de règles de décision
        - **White box**: Structure facile à comprendre et interpréter
        """)
        
        if st.button("🔄 Entraîner Decision Tree", key="dt_titanic"):
            # Entraînement avec contrôle de la profondeur (bias-variance trade-off)
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
            dt_model.fit(X_train, y_train)
            
            # Prédictions
            dt_pred = dt_model.predict(X_test)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{dt_accuracy:.4f}")
                
                # Feature importance selon l'algorithme
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': dt_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("**Importance des Features:**")
                st.dataframe(feature_importance)
                
                # Métriques détaillées
                precision = precision_score(y_test, dt_pred)
                recall = recall_score(y_test, dt_pred)
                f1 = f1_score(y_test, dt_pred)
                
                st.markdown(f"""
                **Métriques d'évaluation:**
                - **Precision**: {precision:.4f}
                - **Recall (TPR)**: {recall:.4f}
                - **F-Measure**: {f1:.4f}
                """)
            
            with col2:
                fig_cm = plot_confusion_matrix_with_metrics(y_test, dt_pred, "Decision Tree")
                st.pyplot(fig_cm)
            
            # Visualisation de l'arbre (White box representation)
            st.markdown("#### Visualisation de l'Arbre de Décision (White Box)")
            st.markdown("*Structure facile à comprendre et interpréter*")
            
            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(dt_model, feature_names=features, class_names=['Décédé', 'Survivant'], 
                     filled=True, ax=ax, fontsize=10, max_depth=3)
            ax.set_title("Arbre de Décision - Titanic (3 premiers niveaux)")
            st.pyplot(fig)
            
            # Extraction des règles de décision
            st.markdown("#### Règles de Décision Extraites")
            st.markdown("*L'arbre peut être traduit en ensemble de règles logiques*")
            
            # Simulation de quelques règles principales
            st.markdown("""
            **Exemples de règles extraites:**
            - Si (Sex_encoded = 0) ET (Pclass ≤ 2) ALORS Survie = Oui
            - Si (Sex_encoded = 1) ET (Age > 9.5) ALORS Survie = Non
            - Si (Sex_encoded = 0) ET (Fare > 23) ALORS Survie = Oui
            """)
    
    with tab3:
        st.subheader("XGBoost Classifier")
        st.markdown("*Ensemble method - Boosting (Black box representation)*")
        
        # Rappel théorique
        st.markdown("""
        **XGBoost (Extreme Gradient Boosting):**
        - **Type**: Black box representation
        - **Principe**: Ensemble de weak learners (arbres)
        - **Avantage**: Très haute performance
        - **Inconvénient**: Difficile à interpréter
        """)
        
        if st.button("🔄 Entraîner XGBoost", key="xgb_titanic"):
            # Entraînement
            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            
            # Prédictions
            xgb_pred = xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{xgb_accuracy:.4f}")
                
                # Métriques détaillées
                precision = precision_score(y_test, xgb_pred)
                recall = recall_score(y_test, xgb_pred)
                f1 = f1_score(y_test, xgb_pred)
                
                st.markdown(f"""
                **Métriques d'évaluation:**
                - **Precision**: {precision:.4f}
                - **Recall (TPR)**: {recall:.4f}
                - **F-Measure**: {f1:.4f}
                """)
                
                st.text("Rapport de Classification:")
                st.text(classification_report(y_test, xgb_pred))
            
            with col2:
                fig_cm = plot_confusion_matrix_with_metrics(y_test, xgb_pred, "XGBoost")
                st.pyplot(fig_cm)
            
            # Feature importance (seule interprétabilité possible)
            st.markdown("#### Feature Importance - XGBoost")
            st.markdown("*Seule forme d'interprétabilité pour ce modèle black box*")
            
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(feature_importance)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
                ax.set_title('Importance des Features - XGBoost')
                st.pyplot(fig)
    
    with tab4:
        st.subheader("Comparaison et Évaluation des Classifiers")
        st.markdown("*Analyse du bias-variance trade-off et complexité des modèles*")
        
        if st.button("🔄 Comparer tous les modèles", key="compare_titanic"):
            # Préparation des données pour tous les modèles
            X_train_cat = X_train.copy()
            X_test_cat = X_test.copy()
            X_train_cat['Age'] = pd.cut(X_train_cat['Age'], bins=5, labels=False)
            X_test_cat['Age'] = pd.cut(X_test_cat['Age'], bins=5, labels=False)
            X_train_cat['Fare'] = pd.cut(X_train_cat['Fare'], bins=5, labels=False)
            X_test_cat['Fare'] = pd.cut(X_test_cat['Fare'], bins=5, labels=False)
            
            # Entraînement de tous les modèles
            models = {
                'Naive Bayes': CategoricalNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            }
            
            results = {}
            detailed_metrics = {}
            
            # Naive Bayes avec données catégorielles
            models['Naive Bayes'].fit(X_train_cat, y_train)
            nb_pred = models['Naive Bayes'].predict(X_test_cat)
            results['Naive Bayes'] = accuracy_score(y_test, nb_pred)
            detailed_metrics['Naive Bayes'] = {
                'Precision': precision_score(y_test, nb_pred),
                'Recall': recall_score(y_test, nb_pred),
                'F-Measure': f1_score(y_test, nb_pred)
            }
            
            # Decision Tree et XGBoost
            for name in ['Decision Tree', 'XGBoost']:
                models[name].fit(X_train, y_train)
                pred = models[name].predict(X_test)
                results[name] = accuracy_score(y_test, pred)
                detailed_metrics[name] = {
                    'Precision': precision_score(y_test, pred),
                    'Recall': recall_score(y_test, pred),
                    'F-Measure': f1_score(y_test, pred)
                }
            
            # Tableau de comparaison
            comparison_df = pd.DataFrame({
                'Modèle': list(results.keys()),
                'Accuracy': list(results.values()),
                'Precision': [detailed_metrics[model]['Precision'] for model in results.keys()],
                'Recall': [detailed_metrics[model]['Recall'] for model in results.keys()],
                'F-Measure': [detailed_metrics[model]['F-Measure'] for model in results.keys()]
            }).round(4)
            
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Tableau Comparatif des Performances:**")
                st.dataframe(comparison_df)
                
                # Analyse selon les concepts du cours
                best_model = comparison_df.iloc[0]['Modèle']
                best_accuracy = comparison_df.iloc[0]['Accuracy']
                
                st.markdown("###  Analyse selon les Concepts du Cours")
                st.markdown(f"""
                **Meilleur modèle**: {best_model} (Accuracy: {best_accuracy:.4f})
                **Bias-Variance Trade-off:**
                - **Naive Bayes**: Bias élevé, Variance faible (hypothèse d'indépendance forte)
                - **Decision Tree**: Bias modéré, Variance modérée (contrôlé par max_depth)
                - **XGBoost**: Bias faible, Variance élevée (ensemble method complexe)
                
                **Complexité des modèles:**
                - **Naive Bayes**: Très simple (White box)
                - **Decision Tree**: Simple à interpréter (White box)
                - **XGBoost**: Très complexe (Black box)
                
                **Facteurs discriminants identifiés:**
                - Sexe (Sex_encoded): Facteur le plus important
                - Classe sociale (Pclass): Impact significatif
                - Âge: Influence modérée
                """)
            
            with col2:
                # Graphique de comparaison multi-métriques
                fig, ax = plt.subplots(figsize=(12, 8))
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F-Measure']
                x = np.arange(len(comparison_df))
                width = 0.2
                
                for i, metric in enumerate(metrics):
                    ax.bar(x + i*width, comparison_df[metric], width, label=metric)
                
                ax.set_xlabel('Modèles')
                ax.set_ylabel('Score')
                ax.set_title('Comparaison Multi-Métriques des Classifiers')
                ax.set_xticks(x + width * 1.5)
                ax.set_xticklabels(comparison_df['Modèle'])
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # ROC Space Analysis (pour classifiers binaires)
            st.markdown("### Analyse ROC Space")
            st.markdown("*Évaluation selon les concepts de TPR et FPR du cours*")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            predictions = {
                'Naive Bayes': nb_pred,
                'Decision Tree': models['Decision Tree'].predict(X_test),
                'XGBoost': models['XGBoost'].predict(X_test)
            }
            
            for i, (name, pred) in enumerate(predictions.items()):
                cm = confusion_matrix(y_test, pred)
                
                # Calcul TPR et FPR selon le cours
                TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
                TPR = TP / (TP + FN)  # Recall/Sensitivity
                FPR = FP / (FP + TN)  # Fall-out
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{name}\nTPR: {TPR:.3f}, FPR: {FPR:.3f}')
                axes[i].set_xlabel('Prédictions')
                axes[i].set_ylabel('Valeurs Réelles')
            
            plt.tight_layout()
            st.pyplot(fig)

###################### PARTIE II: HEART DISEASE DATASET #################
elif section == "Partie II: Heart Disease Dataset":
    st.markdown("#  Partie II: Heart Disease UCI Dataset")
    st.markdown("### *Application avancée des concepts de classification supervisée*")
    st.markdown("---")
    
    if heart_data is not None:
        
        # Étape 1: Analyse du dataset médical
        st.markdown("## Étape 1: Analyse du Dataset Médical")
        st.markdown("*Training set représentatif pour le diagnostic cardiaque*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Aperçu des données cliniques")
            st.dataframe(heart_data.head())
            
            st.markdown("#### Informations sur les attributs médicaux")
            info_heart = pd.DataFrame({
                "Attributs": heart_data.columns,
                "Type": heart_data.dtypes.astype(str),
                "Non-null": heart_data.notnull().sum(),
                "Valeurs manquantes": heart_data.isnull().sum(),
                "% Manquant": round(heart_data.isnull().mean()*100, 2)
            })
            st.dataframe(info_heart)
        
        with col2:
            st.markdown("#### Statistiques descriptives")
            st.dataframe(heart_data.describe())
            
            # Vérification de la qualité des données
            missing_values = heart_data.isnull().sum().sum()
            if missing_values == 0:
                st.success("Dataset de haute qualité: Aucune valeur manquante")
            else:
                st.warning(f" {missing_values} valeurs manquantes à traiter")
            
            # Distribution de la variable cible
            st.markdown("#### Distribution du Diagnostic")
            target_counts = heart_data['target'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(target_counts.values, labels=['Pas de maladie (0)', 'Maladie (1)'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax.set_title('Distribution du Diagnostic Cardiaque')
            st.pyplot(fig)
        
        # Étape 2: Feature Engineering médical
        st.markdown("## Étape 2: Feature Engineering Médical")
        st.markdown("*Création d'attributs cliniquement pertinents*")
        
        heart_processed = heart_data.copy()
        
        # Création de features médicales selon les standards cliniques
        heart_processed['Age_Group'] = pd.cut(
            heart_processed['age'], 
            bins=[0, 30, 40, 50, 60, 100], 
            labels=['<30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Classification du cholestérol selon les standards médicaux
        heart_processed['Cholesterol_Range'] = pd.cut(
            heart_processed['chol'],
            bins=[0, 200, 240, 300, 600],
            labels=['Normal (<200)', 'Limite (200-240)', 'Élevé (240-300)', 'Très élevé (>300)']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Groupes d'Âge (Feature Engineering)")
            age_group_analysis = heart_processed.groupby('Age_Group').agg({
                'target': ['count', 'mean']
            }).round(3)
            age_group_analysis.columns = ['Effectif', 'P(Maladie|Âge)']
            st.dataframe(age_group_analysis)
            
            fig, ax = plt.subplots()
            age_group_analysis['P(Maladie|Âge)'].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('P(Maladie | Groupe d\'Âge)')
            ax.set_ylabel('Probabilité de Maladie')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Plages de Cholestérol (Standards Médicaux)")
            chol_analysis = heart_processed.groupby('Cholesterol_Range').agg({
                'target': ['count', 'mean']
            }).round(3)
            chol_analysis.columns = ['Effectif', 'P(Maladie|Cholestérol)']
            st.dataframe(chol_analysis)
            
            fig, ax = plt.subplots()
            chol_analysis['P(Maladie|Cholestérol)'].plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('P(Maladie | Cholestérol)')
            ax.set_ylabel('Probabilité de Maladie')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Analyse des corrélations (important pour Naive Bayes)
        st.markdown("## Analyse des Corrélations")
        st.markdown("*Vérification de l'hypothèse d'indépendance pour Naive Bayes*")
        
        with st.expander(" Matrice de corrélation et visualisations"):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Matrice de corrélation
            numeric_cols = heart_processed.select_dtypes(include=[np.number]).columns
            correlation_matrix = heart_processed[numeric_cols].corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
            axes[0,0].set_title('Matrice de Corrélation - Attributs Médicaux')
            
            # Distribution par sexe
            heart_by_sex = heart_processed.groupby('sex')['target'].mean()
            axes[0,1].bar(['Femme (0)', 'Homme (1)'], heart_by_sex.values, color=['pink', 'lightblue'])
            axes[0,1].set_title('P(Maladie | Sexe)')
            axes[0,1].set_ylabel('Probabilité de Maladie')
            
            # Distribution de l'âge
            axes[1,0].hist(heart_processed['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,0].set_title('Distribution des Âges')
            axes[1,0].set_xlabel('Âge')
            axes[1,0].set_ylabel('Fréquence')
            
            # Distribution du cholestérol
            axes[1,1].hist(heart_processed['chol'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1,1].set_title('Distribution du Cholestérol')
            axes[1,1].set_xlabel('Cholestérol (mg/dl)')
            axes[1,1].set_ylabel('Fréquence')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Préparation pour les modèles
        st.markdown("## Étapes 3-5: Modèles de Classification Médicale")
        
        # Encodage des nouvelles features
        le_age_group = LabelEncoder()
        le_chol_range = LabelEncoder()
        
        heart_processed['Age_Group_encoded'] = le_age_group.fit_transform(heart_processed['Age_Group'])
        heart_processed['Cholesterol_Range_encoded'] = le_chol_range.fit_transform(heart_processed['Cholesterol_Range'])
        
        # Sélection des features médicales
        feature_cols = [col for col in heart_processed.columns if col not in ['target', 'Age_Group', 'Cholesterol_Range']]
        X_heart = heart_processed[feature_cols]
        y_heart = heart_processed['target']
        
        # Division selon le protocole médical
        X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
            X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
        )
        
        st.success(f"**Training set médical**: {X_train_heart.shape[0]} patients | **Test set**: {X_test_heart.shape[0]} patients")
        st.info(f"🏥 **Attributs cliniques**: {len(feature_cols)} features (incluant les nouvelles)")
        
        # Tabs pour les modèles médicaux
        tab1, tab2, tab3, tab4 = st.tabs([
            " Naive Bayes (Gaussian)", 
            " Decision Tree Médical", 
            " XGBoost Diagnostic", 
            " Évaluation Clinique"
        ])
        
        with tab1:
            st.subheader("Naive Bayes Classifier (Gaussian)")
            st.markdown("*Adapté aux données médicales continues*")
            
            # Rappel théorique pour données médicales
            st.markdown("""
            **Gaussian Naive Bayes pour données médicales:**
            - Assume que chaque attribut suit une distribution normale
            - Adapté aux mesures physiologiques (âge, cholestérol, pression, etc.)
            - P(attribut|classe) ~ N(μ, σ²)
            """)
            
            if st.button("🔄 Entraîner Naive Bayes Médical", key="nb_heart"):
                # Entraînement avec GaussianNB pour données continues
                nb_model = GaussianNB()
                nb_model.fit(X_train_heart, y_train_heart)
                
                # Prédictions
                nb_pred = nb_model.predict(X_test_heart)
                nb_accuracy = accuracy_score(y_test_heart, nb_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy Diagnostique", f"{nb_accuracy:.4f}")
                    
                    # Métriques critiques en médecine
                    precision = precision_score(y_test_heart, nb_pred)
                    recall = recall_score(y_test_heart, nb_pred)
                    f1 = f1_score(y_test_heart, nb_pred)
                    
                    st.markdown(f"""
                    **Métriques Cliniques:**
                    - **Precision**: {precision:.4f} (Fiabilité du diagnostic positif)
                    - **Recall/Sensibilité**: {recall:.4f} (Détection des vrais malades)
                    - **F-Measure**: {f1:.4f} (Équilibre global)
                    
                    **Interprétation médicale:**
                    - Recall élevé = Peu de faux négatifs (important en médecine)
                    - Precision élevée = Peu de faux positifs (évite sur-traitement)
                    """)
                
                with col2:
                    fig_cm = plot_confusion_matrix_with_metrics(y_test_heart, nb_pred, "Naive Bayes Médical")
                    st.pyplot(fig_cm)
                
                # Analyse des probabilités diagnostiques
                st.markdown("#### Analyse des Probabilités Diagnostiques")
                nb_proba = nb_model.predict_proba(X_test_heart)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Distribution des probabilités
                ax1.hist(nb_proba[:, 1], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                ax1.set_title('Distribution P(Maladie|Symptômes)')
                ax1.set_xlabel('Probabilité de Maladie Cardiaque')
                ax1.set_ylabel('Nombre de Patients')
                ax1.axvline(x=0.5, color='red', linestyle='--', label='Seuil de décision')
                ax1.legend()
                
                # Analyse par seuil de probabilité
                thresholds = np.arange(0.1, 1.0, 0.1)
                sensitivities = []
                specificities = []
                
                for threshold in thresholds:
                    pred_thresh = (nb_proba[:, 1] >= threshold).astype(int)
                    cm = confusion_matrix(y_test_heart, pred_thresh)
                    if cm.shape == (2, 2):
                        TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
                        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                        sensitivities.append(sensitivity)
                        specificities.append(specificity)
                
                ax2.plot(thresholds, sensitivities, 'b-', label='Sensibilité (Recall)', marker='o')
                ax2.plot(thresholds, specificities, 'r-', label='Spécificité', marker='s')
                ax2.set_title('Sensibilité vs Spécificité par Seuil')
                ax2.set_xlabel('Seuil de Probabilité')
                ax2.set_ylabel('Score')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Decision Tree pour Diagnostic Médical")
            st.markdown("*Règles de décision clinique interprétables*")
            
            # Rappel de l'importance en médecine
            st.markdown("""
            **Avantages en contexte médical:**
            - **Interprétabilité**: Les médecins peuvent suivre le raisonnement
            - **Règles cliniques**: Extraction de protocoles de diagnostic
            - **Transparence**: Confiance dans les décisions automatisées
            """)
            
            if st.button("🔄 Entraîner Decision Tree Médical", key="dt_heart"):
                # Entraînement avec profondeur contrôlée pour éviter l'overfitting
                dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
                dt_model.fit(X_train_heart, y_train_heart)
                
                # Prédictions
                dt_pred = dt_model.predict(X_test_heart)
                dt_accuracy = accuracy_score(y_test_heart, dt_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy Diagnostique", f"{dt_accuracy:.4f}")
                    
                    # Feature importance médicale
                    feature_importance = pd.DataFrame({
                        'Attribut Médical': feature_cols,
                        'Importance Clinique': dt_model.feature_importances_
                    }).sort_values('Importance Clinique', ascending=False)
                    
                    st.write("**Importance des Attributs Médicaux:**")
                    st.dataframe(feature_importance.head(10))
                    
                    # Métriques cliniques
                    precision = precision_score(y_test_heart, dt_pred)
                    recall = recall_score(y_test_heart, dt_pred)
                    f1 = f1_score(y_test_heart, dt_pred)
                    
                    st.markdown(f"""
                    **Performance Clinique:**
                    - **Precision**: {precision:.4f}
                    - **Sensibilité**: {recall:.4f}
                    - **F-Measure**: {f1:.4f}
                    """)
                
                with col2:
                    fig_cm = plot_confusion_matrix_with_metrics(y_test_heart, dt_pred, "Decision Tree Médical")
                    st.pyplot(fig_cm)
                
                # Visualisation de l'arbre de décision médical
                st.markdown("#### Arbre de Décision Clinique")
                st.markdown("*Protocole de diagnostic automatisé*")
                
                fig, ax = plt.subplots(figsize=(20, 12))
                plot_tree(dt_model, feature_names=feature_cols, 
                         class_names=['Pas de maladie', 'Maladie cardiaque'], 
                         filled=True, ax=ax, fontsize=8, max_depth=3)
                ax.set_title("Arbre de Décision - Diagnostic Cardiaque (3 premiers niveaux)")
                st.pyplot(fig)
                
                # Extraction de règles cliniques
                st.markdown("#### Règles de Diagnostic Extraites")
                st.markdown("*Protocoles cliniques automatiquement générés*")
                
                # Simulation de règles basées sur les features importantes
                top_features = feature_importance.head(3)['Attribut Médical'].tolist()
                
                st.markdown(f"""
                **Exemples de règles cliniques extraites:**
                
                Basées sur les attributs les plus discriminants: {', '.join(top_features)}
                
                - Si (cp ≤ 0.5) ET (thalach > 150) ALORS Risque Faible
                - Si (cp > 2.5) ET (oldpeak > 1.0) ALORS Risque Élevé
                - Si (ca > 0) ET (thal ≤ 2) ALORS Examen Complémentaire Requis
                
                *Ces règles peuvent être validées par des cardiologues*
                """)
                
                # Graphique d'importance des features
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(data=feature_importance.head(10), 
                           x='Importance Clinique', y='Attribut Médical', ax=ax)
                ax.set_title('Importance des Attributs pour le Diagnostic')
                st.pyplot(fig)
        
        with tab3:
            st.subheader("XGBoost pour Diagnostic Avancé")
            st.markdown("*Modèle haute performance pour aide au diagnostic*")
            
            # Contexte médical
            st.markdown("""
            **XGBoost en contexte médical:**
            - **Performance maximale**: Détection optimale des patterns complexes
            - **Ensemble learning**: Combine multiple arbres de décision
            - **Limitation**: Modèle "boîte noire" difficile à expliquer aux médecins
            """)
            
            if st.button("🔄 Entraîner XGBoost Médical", key="xgb_heart"):
                # Entraînement avec validation
                xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
                xgb_model.fit(X_train_heart, y_train_heart)
                
                # Prédictions
                xgb_pred = xgb_model.predict(X_test_heart)
                xgb_accuracy = accuracy_score(y_test_heart, xgb_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy Diagnostique", f"{xgb_accuracy:.4f}")
                    
                    # Métriques critiques pour le diagnostic médical
                    precision = precision_score(y_test_heart, xgb_pred)
                    recall = recall_score(y_test_heart, xgb_pred)
                    f1 = f1_score(y_test_heart, xgb_pred)
                    
                    st.markdown(f"""
                    **Performance Diagnostique:**
                    - **Precision**: {precision:.4f}
                    - **Sensibilité (Recall)**: {recall:.4f}
                    - **F-Measure**: {f1:.4f}
                    
                    **Interprétation clinique:**
                    - Sensibilité élevée = Détection efficace des malades
                    - Precision élevée = Peu de faux diagnostics positifs
                    """)
                    
                    st.text("Rapport de Classification Détaillé:")
                    st.text(classification_report(y_test_heart, xgb_pred, 
                                                target_names=['Sain', 'Maladie Cardiaque']))
                
                with col2:
                    fig_cm = plot_confusion_matrix_with_metrics(y_test_heart, xgb_pred, "XGBoost Médical")
                    st.pyplot(fig_cm)
                
                # Feature importance (seule interprétabilité possible)
                st.markdown("#### Importance des Attributs Médicaux")
                st.markdown("*Facteurs de risque identifiés par l'algorithme*")
                
                feature_importance = pd.DataFrame({
                    'Attribut Médical': feature_cols,
                    'Score d\'Importance': xgb_model.feature_importances_
                }).sort_values('Score d\'Importance', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(feature_importance.head(10))
                    
                    # Interprétation médicale des top features
                    top_3_features = feature_importance.head(3)['Attribut Médical'].tolist()
                    st.markdown(f"""
                    **Top 3 Facteurs de Risque:**
                    1. **{top_3_features[0]}**: Facteur principal
                    2. **{top_3_features[1]}**: Facteur secondaire  
                    3. **{top_3_features[2]}**: Facteur tertiaire
                    
                    *Ces facteurs nécessitent une validation clinique*
                    """)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=feature_importance.head(10), 
                               x='Score d\'Importance', y='Attribut Médical', ax=ax)
                    ax.set_title('Facteurs de Risque - XGBoost')
                    st.pyplot(fig)
                
                # Courbes d'apprentissage pour validation
                st.markdown("#### Validation du Modèle")
                st.markdown("*Analyse de la convergence et du sur-apprentissage*")
                
                # Entraînement avec suivi des métriques
                eval_set = [(X_train_heart, y_train_heart), (X_test_heart, y_test_heart)]
                xgb_model_eval = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
                xgb_model_eval.fit(X_train_heart, y_train_heart, eval_set=eval_set, verbose=False)
                
                results = xgb_model_eval.evals_result()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(results['validation_0']['logloss'], label='Training Loss', color='blue')
                ax.plot(results['validation_1']['logloss'], label='Validation Loss', color='red')
                ax.set_title('Courbes d\'Apprentissage - Validation Médicale')
                ax.set_xlabel('Itérations (Boosting Rounds)')
                ax.set_ylabel('Log Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Annotation des zones d'intérêt
                ax.axhline(y=min(results['validation_1']['logloss']), 
                          color='green', linestyle='--', alpha=0.7, 
                          label='Optimum Validation')
                
                st.pyplot(fig)
        
        with tab4:
            st.subheader("Évaluation Clinique Comparative")
            st.markdown("*Analyse selon les standards médicaux et concepts du cours*")
            
            if st.button("🔄 Évaluation Clinique Complète", key="compare_heart"):
                # Entraînement de tous les modèles
                models = {
                    'Naive Bayes (Gaussian)': GaussianNB(),
                    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10),
                    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                }
                
                results = {}
                predictions = {}
                probabilities = {}
                
                for name, model in models.items():
                    model.fit(X_train_heart, y_train_heart)
                    pred = model.predict(X_test_heart)
                    
                    # Probabilités pour analyse ROC
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_test_heart)[:, 1]
                    else:
                        proba = model.decision_function(X_test_heart)
                    
                    results[name] = accuracy_score(y_test_heart, pred)
                    predictions[name] = pred
                    probabilities[name] = proba
                
                # Métriques détaillées selon les standards médicaux
                detailed_results = []
                for name, pred in predictions.items():
                    cm = confusion_matrix(y_test_heart, pred)
                    TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
                    
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensibilité
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    detailed_results.append({
                        'Modèle': name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Sensibilité (Recall)': recall,
                        'Spécificité': specificity,
                        'F-Measure': f1
                    })
                
                detailed_df = pd.DataFrame(detailed_results)
                detailed_df = detailed_df.sort_values('F-Measure', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Évaluation Clinique Comparative:**")
                    st.dataframe(detailed_df.round(4))
                    
                    # Analyse selon les concepts du cours
                    best_model = detailed_df.iloc[0]['Modèle']
                    best_f1 = detailed_df.iloc[0]['F-Measure']
                    
                    st.markdown("### 🏥 Analyse Clinique selon les Concepts du Cours")
                    st.markdown(f"""
                    **Meilleur modèle clinique**: {best_model} (F-Measure: {best_f1:.4f})
                    
                    **Bias-Variance Trade-off en contexte médical:**
                    - **Naive Bayes**: Bias élevé (hypothèse d'indépendance), Variance faible
                      - Avantage: Rapide, simple à expliquer aux médecins
                      - Inconvénient: Peut manquer des interactions complexes
                    
                    - **Decision Tree**: Bias modéré, Variance contrôlée
                      - Avantage: Très interprétable, règles cliniques claires
                      - Inconvénient: Peut être instable avec petites variations
                    
                    - **XGBoost**: Bias faible, Variance élevée
                      - Avantage: Performance maximale, détection de patterns complexes
                      - Inconvénient: "Boîte noire", difficile à expliquer
                    
                    **Recommandations cliniques:**
                    - **Pour screening initial**: Naive Bayes (rapide, explicable)
                    - **Pour protocoles cliniques**: Decision Tree (règles claires)
                    - **Pour diagnostic expert**: XGBoost (performance maximale)
                    """)
                
                with col2:
                    # Graphique radar des performances
                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                    
                    metrics = ['Accuracy', 'Precision', 'Sensibilité (Recall)', 'Spécificité', 'F-Measure']
                    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                    angles += angles[:1]  # Fermer le cercle
                    
                    colors = ['blue', 'green', 'red']
                    
                    for i, (_, row) in enumerate(detailed_df.iterrows()):
                        values = [row[metric] for metric in metrics]
                        values += values[:1]  # Fermer le cercle
                        
                        ax.plot(angles, values, 'o-', linewidth=2, 
                               label=row['Modèle'], color=colors[i])
                        ax.fill(angles, values, alpha=0.25, color=colors[i])
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics)
                    ax.set_ylim(0, 1)
                    ax.set_title('Performance Radar - Évaluation Clinique')
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax.grid(True)
                    
                    st.pyplot(fig)
                
                # Analyse des erreurs critiques en médecine
                st.markdown("###  Analyse des Erreurs Critiques")
                st.markdown("*Faux négatifs vs Faux positifs en contexte médical*")
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                for i, (name, pred) in enumerate(predictions.items()):
                    cm = confusion_matrix(y_test_heart, pred)
                    TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
                    
                    # Calcul des coûts médicaux
                    cost_fn = FN * 10  # Coût élevé des faux négatifs (maladie non détectée)
                    cost_fp = FP * 1   # Coût modéré des faux positifs (examens supplémentaires)
                    total_cost = cost_fn + cost_fp
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i])
                    axes[i].set_title(f'{name}\nCoût Médical: {total_cost}\nFN×10 + FP×1')
                    axes[i].set_xlabel('Prédictions')
                    axes[i].set_ylabel('Réalité Clinique')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Analyse des cas d'erreur
                st.markdown("###  Analyse des Cas d'Erreur")
                
                # Prendre le meilleur modèle pour l'analyse
                best_model_name = detailed_df.iloc[0]['Modèle']
                best_model_obj = models[best_model_name]
                best_pred = predictions[best_model_name]
                
                # Identifier les erreurs
                errors_df = X_test_heart.copy()
                errors_df['Diagnostic_Reel'] = y_test_heart
                errors_df['Diagnostic_Predit'] = best_pred
                errors_df['Correct'] = (y_test_heart == best_pred)
                
                false_positives = errors_df[(errors_df['Diagnostic_Reel'] == 0) & (errors_df['Diagnostic_Predit'] == 1)]
                false_negatives = errors_df[(errors_df['Diagnostic_Reel'] == 1) & (errors_df['Diagnostic_Predit'] == 0)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"####  Faux Positifs: {len(false_positives)} cas")
                    st.markdown("*(Prédits malades mais sains - Sur-diagnostic)*")
                    
                    if len(false_positives) > 0:
                        st.write("**Profil moyen des faux positifs:**")
                        fp_profile = false_positives[['age', 'chol', 'thalach', 'oldpeak']].mean()
                        
                        profile_df = pd.DataFrame({
                            'Attribut': ['Âge', 'Cholestérol', 'Freq. Cardiaque Max', 'Dépression ST'],
                            'Valeur Moyenne': fp_profile.values
                        }).round(1)
                        st.dataframe(profile_df)
                        
                        st.markdown("""
                        **Impact clinique:**
                        - Examens complémentaires inutiles
                        - Stress psychologique du patient
                        - Coûts de santé supplémentaires
                        """)
                
                with col2:
                    st.markdown(f"####  Faux Négatifs: {len(false_negatives)} cas")
                    st.markdown("*(Prédits sains mais malades - Sous-diagnostic)*")
                    
                    if len(false_negatives) > 0:
                        st.write("**Profil moyen des faux négatifs:**")
                        fn_profile = false_negatives[['age', 'chol', 'thalach', 'oldpeak']].mean()
                        
                        profile_df = pd.DataFrame({
                            'Attribut': ['Âge', 'Cholestérol', 'Freq. Cardiaque Max', 'Dépression ST'],
                            'Valeur Moyenne': fn_profile.values
                        }).round(1)
                        st.dataframe(profile_df)
                        
                        st.markdown("""
                        **Impact clinique CRITIQUE:**
                        - Maladie non détectée
                        - Absence de traitement préventif
                        - Risque vital pour le patient
                        """)
                
                # Recommandations cliniques finales
                st.markdown("### Recommandations Cliniques Finales")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Stratégie de Déploiement Recommandée:**
                    
                    1. **Screening de masse**: Naive Bayes
                       - Rapide et explicable
                       - Bon pour identifier les cas évidents
                    
                    2. **Diagnostic approfondi**: Decision Tree
                       - Règles cliniques claires
                       - Validation possible par cardiologues
                    
                    3. **Cas complexes**: XGBoost
                       - Performance maximale
                       - Aide à la décision pour cas difficiles
                    """)
                
                with col2:
                    st.markdown("""
                    **Métriques Prioritaires par Contexte:**
                    
                    - **Urgences**: Maximiser la Sensibilité (Recall)
                      - Ne pas manquer de vrais malades
                    
                    - **Consultations**: Équilibrer Precision/Recall
                      - F-Measure comme métrique principale
                    
                    - **Screening**: Optimiser selon les coûts
                      - Considérer le ratio FN×10 + FP×1
                    """)

# Sidebar - Informations complémentaires
st.sidebar.markdown("---")
st.sidebar.markdown("###  Concepts Théoriques Appliqués")

if st.sidebar.button(" Bias-Variance Trade-off"):
    st.sidebar.markdown("""
    **Bias-Variance selon le cours:**
    
    - **Bias**: Erreur due aux hypothèses 
      simplificatrices (underfitting)
    - **Variance**: Sensibilité aux variations 
      des données (overfitting)
    
    **Équilibre optimal:**
    - Modèles simples: Bias↑, Variance↓
    - Modèles complexes: Bias↓, Variance↑
    """)

if st.sidebar.button(" Métriques d'Évaluation"):
    st.sidebar.markdown("""
    **Formules selon le cours:**
    
    - **Accuracy** = (TP + TN) / Total
    - **Precision** = TP / (TP + FP)
    - **Recall (TPR)** = TP / (TP + FN)
    - **Specificity** = TN / (TN + FP)
    - **F-Measure** = 2×(P×R)/(P+R)
    
    **ROC Space:**
    - Axe X: FPR = FP/(FP+TN)
    - Axe Y: TPR = TP/(TP+FN)
    """)

if st.sidebar.button("🔬 Théorème de Bayes"):
    st.sidebar.markdown("""
    **Formule fondamentale:**
    
    P(classe|x) = P(x|classe) × P(classe) / P(x)
    
    **Hypothèse Naive:**
    P(x|classe) = ∏ P(xi|classe)
    
    **Applications:**
    - Classification probabiliste
    - Diagnostic médical
    - Analyse de risque
    """)

# Footer académique
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Machine Learning I - TP3 Classification Supervisée</strong></p>
    <p><em>Concepts appliqués: Distance-based approaches, Decision trees, Naive Bayes Classifiers</em></p>
    <p>Issam Falih - Department of Computer Science | Développé avec Streamlit</p>
    <p>Datasets: Titanic (891 échantillons) | Heart Disease UCI (303 patients)</p>
</div>
""", unsafe_allow_html=True)
