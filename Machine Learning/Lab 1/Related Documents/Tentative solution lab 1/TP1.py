import streamlit as st

st.set_page_config(
    page_title="Accueil - TP 1",
    page_icon="📖",
)

st.sidebar.success("Sélectionnez une section du TP ci-dessus pour voir la solution.")

st.title("Apprentissage Artificiel - TP 1 : Introduction")
st.caption("Auteur: Issam Falih - issam.falih@uca.fr")

st.markdown("""
*Instructions : Préparez un rapport incluant le code source et vos résultats, et déposez-le sur Moodle. Il est recommandé d'utiliser [Streamlit](https://streamlit.io) pour afficher vos résultats.*
""")

st.divider()

st.header("A. Manipulation d’un jeu de données")
st.markdown("""
1.  Chargez les données `Iris` avec la commande :
    `iris = datasets.load_iris()`

    La variable `iris` est un objet qui contient :
    - la matrice des données (`iris.data`),
    - un vecteur de numéro de classe (`target`),
    - les noms des variables (`feature_names`),
    - le nom des classes (`target_names`).

2.  Affichez les données, les noms des variables et le nom des classes (utilisez `print`).

3.  Affichez le nom des classes pour chaque donnée.

4.  Affichez la moyenne (*mean*), l’écart-type (*std*), le min et le max pour chaque variable.

5.  En utilisant les attributs `size` et `shape`, affichez le nombre de données, le nombre de variables et le nombre de classes.
""")

st.header("B. Téléchargement et importation de données")
st.markdown("""
Il est possible de charger un jeu de données directement depuis des dépôts en ligne qui contiennent de nombreux jeux de données gratuits.

1.  Importez les données `'MNIST original'`.

2.  Affichez la matrice des données, le nombre de données et de variables, les numéros de classes pour chaque donnée, ainsi que la moyenne, l’écart-type, les valeurs min et max pour chaque variable ; enfin donnez le nombre de classes avec la fonction `unique`.
""")

st.header("C. Génération de données et affichage")
st.markdown("""
1.  Utilisez l’aide (`help`) pour voir comment utiliser la fonction `datasets.make_blobs`.

2.  Générez 1000 données de deux variables réparties en 4 groupes.

3.  Utilisez les fonctions `figure`, `scatter`, `title`, `xlim`, `ylim`, `xlabel`, `ylabel` et `show` pour afficher les données avec des couleurs correspondant aux classes.
    Les axes `x` et `y` seront dans l’intervalle [-15, 15] et devront avoir un titre. La figure doit aussi avoir un titre.

4.  Générez 100 données de deux variables réparties en 2 groupes, puis 500 données de deux variables réparties en 3 groupes. Concaténez (`vstack` et `hstack`) les deux jeux de données et les numéros de classe pour fusionner les deux jeux de données. Affichez les trois ensembles avec `scatter` comme précédemment.
""")
