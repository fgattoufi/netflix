import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Charger le dataset
df = pd.read_csv('https://github.com/fgattoufi/netflix/blob/4512070e5c1f4c0a390f90f66760c653fc385201/netflix_titles.csv')

# Examiner les premières lignes
print(df.head())

# Vérifier les informations sur le dataset
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())


# Convertir la colonne date_added en format datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extraire le mois et l'année pour l'analyse de saisonnalité
df['month_added'] = df['date_added'].dt.month
df['year_added'] = df['date_added'].dt.year

# Séparer les genres (qui sont souvent multiples et séparés par des virgules)
df['listed_in'] = df['listed_in'].str.split(', ')

# Séparer les pays (qui peuvent également être multiples)
df['country'] = df['country'].str.split(', ')

# Créer une liste de tous les genres
all_genres = []
for genres in df['listed_in'].dropna():
    all_genres.extend(genres)

# Compter la fréquence de chaque genre
genre_counts = pd.Series(all_genres).value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

# Visualisation des 10 genres les plus populaires
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Genre', data=genre_counts.head(10), color='#E50914')
plt.title('Top 10 des Genres sur Netflix', fontsize=16)
plt.xlabel('Nombre de Titres', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.tight_layout()
plt.savefig('top_genres.png')
plt.show()

# Compter le nombre unique de pays
all_countries = []
for countries in df['country'].dropna():
    all_countries.extend(countries)

unique_countries = len(set(all_countries))
print(f"Netflix est disponible dans {unique_countries} pays.")

# Compter la fréquence de chaque pays
country_counts = pd.Series(all_countries).value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Visualisation des 10 pays avec le plus de contenu
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Country', data=country_counts.head(10), color='#E50914')
plt.title('Top 10 des Pays par Nombre de Titres sur Netflix', fontsize=16)
plt.xlabel('Nombre de Titres', fontsize=12)
plt.ylabel('Pays', fontsize=12)
plt.tight_layout()
plt.savefig('top_countries.png')
plt.show()

# Créer un dataframe pour l'analyse des genres par pays
country_genre_df = pd.DataFrame(columns=['Country', 'Genre', 'Count'])

# Pour chaque pays, trouver le genre dominant
for country in country_counts['Country'].head(10):  # Limiter aux 10 premiers pays
    country_titles = df[df['country'].apply(lambda x: country in x if isinstance(x, list) else False)]
    
    country_genres = []
    for genres in country_titles['listed_in'].dropna():
        country_genres.extend(genres)
    
    if country_genres:
        top_genre = pd.Series(country_genres).value_counts().idxmax()
        count = pd.Series(country_genres).value_counts().max()
        country_genre_df = country_genre_df.append({'Country': country, 'Genre': top_genre, 'Count': count}, ignore_index=True)

# Visualisation
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Country', hue='Genre', data=country_genre_df, palette='Reds')
plt.title('Genre Dominant par Pays', fontsize=16)
plt.xlabel('Nombre de Titres', fontsize=12)
plt.ylabel('Pays', fontsize=12)
plt.tight_layout()
plt.savefig('dominant_genre_by_country.png')
plt.show()

# Analyser les tendances mensuelles des sorties
monthly_releases = df.groupby('month_added')['show_id'].count().reset_index()
monthly_releases.columns = ['Month', 'Count']

# Ajouter les noms des mois pour une meilleure lisibilité
month_names = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin', 
               7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
monthly_releases['Month_Name'] = monthly_releases['Month'].map(month_names)

# Visualisation
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Count', data=monthly_releases, marker='o', color='#E50914')
plt.xticks(monthly_releases['Month'], monthly_releases['Month_Name'])
plt.title('Saisonnalité des Sorties Netflix par Mois', fontsize=16)
plt.xlabel('Mois', fontsize=12)
plt.ylabel('Nombre de Titres Ajoutés', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonality.png')
plt.show()

# Analyser les tendances annuelles
yearly_releases = df.groupby('year_added')['show_id'].count().reset_index()
yearly_releases.columns = ['Year', 'Count']

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Count', data=yearly_releases, marker='o', color='#E50914')
plt.title('Évolution des Sorties Netflix par Année', fontsize=16)
plt.xlabel('Année', fontsize=12)
plt.ylabel('Nombre de Titres Ajoutés', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('yearly_trend.png')
plt.show()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Netflix Data Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour appliquer le style Netflix
def netflix_style():
    st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #221F1F;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #E50914;
    }
    .stButton>button {
        background-color: #E50914;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

netflix_style()

# Chargement des données
@st.cache
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['month_added'] = df['date_added'].dt.month
    df['year_added'] = df['date_added'].dt.year
    df['listed_in'] = df['listed_in'].str.split(', ')
    df['country'] = df['country'].str.split(', ')
    return df

df = load_data()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Genres", "Pays", "Saisonnalité"])

# Page d'accueil
if page == "Accueil":
    st.title("Analyse du Catalogue Netflix")
    st.image("https://assets.brand.microsites.netflix.io/assets/7dc497e2-4975-11ec-a9ce-066b49664af6_cm_1440w.jpg?v=4", width=600)
    st.write("""
    Ce dashboard présente une analyse complète du catalogue Netflix, mettant en évidence:
    - Les genres les plus populaires
    - La présence mondiale de Netflix
    - Les pays avec le plus de contenu
    - Les genres dominants par pays
    - La saisonnalité des sorties
    """)
    
    # Afficher quelques statistiques de base
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre Total de Titres", len(df))
    with col2:
        st.metric("Films", len(df[df['type'] == 'Movie']))
    with col3:
        st.metric("Séries TV", len(df[df['type'] == 'TV Show']))

# Page des genres
elif page == "Genres":
    st.title("Analyse des Genres sur Netflix")
    
    # Créer une liste de tous les genres
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend(genres)
    
    # Compter la fréquence de chaque genre
    genre_counts = pd.Series(all_genres).value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    # Visualisation des 10 genres les plus populaires
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Count', y='Genre', data=genre_counts.head(10), color='#E50914', ax=ax)
    ax.set_title('Top 10 des Genres sur Netflix', fontsize=16)
    ax.set_xlabel('Nombre de Titres', fontsize=12)
    ax.set_ylabel('Genre', fontsize=12)
    st.pyplot(fig)
    
    # Afficher le tableau des données
    st.subheader("Données des Genres")
    st.dataframe(genre_counts.head(20))

# Page des pays
elif page == "Pays":
    st.title("Analyse par Pays")
    
    # Compter le nombre unique de pays
    all_countries = []
    for countries in df['country'].dropna():
        all_countries.extend(countries)
    
    unique_countries = len(set(all_countries))
    st.metric("Nombre de Pays où Netflix est Disponible", unique_countries)
    
    # Compter la fréquence de chaque pays
    country_counts = pd.Series(all_countries).value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    # Visualisation des 10 pays avec le plus de contenu
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Count', y='Country', data=country_counts.head(10), color='#E50914', ax=ax)
    ax.set_title('Top 10 des Pays par Nombre de Titres sur Netflix', fontsize=16)
    ax.set_xlabel('Nombre de Titres', fontsize=12)
    ax.set_ylabel('Pays', fontsize=12)
    st.pyplot(fig)
    
    # Genre dominant par pays
    st.subheader("Genre Dominant par Pays")
    
    # Créer un dataframe pour l'analyse des genres par pays
    country_genre_df = pd.DataFrame(columns=['Country', 'Genre', 'Count'])
    
    # Pour chaque pays, trouver le genre dominant
    for country in country_counts['Country'].head(10):
        country_titles = df[df['country'].apply(lambda x: country in x if isinstance(x, list) else False)]
        
        country_genres = []
        for genres in country_titles['listed_in'].dropna():
            country_genres.extend(genres)
        
        if country_genres:
            top_genre = pd.Series(country_genres).value_counts().idxmax()
            count = pd.Series(country_genres).value_counts().max()
            country_genre_df = country_genre_df.append({'Country': country, 'Genre': top_genre, 'Count': count}, ignore_index=True)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Count', y='Country', hue='Genre', data=country_genre_df, palette='Reds', ax=ax)
    ax.set_title('Genre Dominant par Pays', fontsize=16)
    ax.set_xlabel('Nombre de Titres', fontsize=12)
    ax.set_ylabel('Pays', fontsize=12)
    st.pyplot(fig)

# Page de saisonnalité
elif page == "Saisonnalité":
    st.title("Saisonnalité des Sorties Netflix")
    
    # Analyser les tendances mensuelles des sorties
    monthly_releases = df.groupby('month_added')['show_id'].count().reset_index()
    monthly_releases.columns = ['Month', 'Count']
    
    # Ajouter les noms des mois pour une meilleure lisibilité
    month_names = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin', 
                   7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
    monthly_releases['Month_Name'] = monthly_releases['Month'].map(month_names)
    
    # Visualisation mensuelle
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Month', y='Count', data=monthly_releases, marker='o', color='#E50914', ax=ax)
    ax.set_xticks(monthly_releases['Month'])
    ax.set_xticklabels(monthly_releases['Month_Name'])
    ax.set_title('Saisonnalité des Sorties Netflix par Mois', fontsize=16)
    ax.set_xlabel('Mois', fontsize=12)
    ax.set_ylabel('Nombre de Titres Ajoutés', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    # Analyser les tendances annuelles
    yearly_releases = df.groupby('year_added')['show_id'].count().reset_index()
    yearly_releases.columns = ['Year', 'Count']
    
    # Visualisation annuelle
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='Year', y='Count', data=yearly_releases, marker='o', color='#E50914', ax=ax)
    ax.set_title('Évolution des Sorties Netflix par Année', fontsize=16)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Nombre de Titres Ajoutés', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

