import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from joblib import dump,load



rf_PO = joblib.load('random-forestPO')
rf_SR = joblib.load('random-forest')
scal_SR = joblib.load("scalerSR")
pca = joblib.load("PCA")


@st.cache
def load_data(fichier):
    shot_location = pd.read_csv(fichier, sep=',')
    
    top_players = ['LeBron James', 'Kobe Bryant', 'Tim Duncan', "Shaquille O'Neal", 'Stephen Curry', 'Kevin Durant', 'Dwyane Wade', 'Giannis Antetokounmpo', 'Kevin Garnett', 'Dirk Nowitzki', 'Kawhi Leonard', 'Allen Iverson', 'Steve Nash', 'Tony Parker', 'Damian Lillard', 'Paul Pierce', 'Jason Kidd', 'Russel Westbrook', 'Ray Allen', 'James Harden']
    shot_loc_top_players = shot_location[shot_location['Player Name'].isin(top_players)]

    variables_selection = ['Game Date', 'Action Type', 'Shot Made Flag', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 'X Location', 'Y Location', 'Season Type', 'Player Name']

    shot_loc = shot_loc_top_players[variables_selection]
    shot_loc.rename(columns={'Player Name': 'Player'}, inplace=True)

    shot_loc['Game Date'] = shot_loc['Game Date'].astype(str)
    shot_loc['Year'] = shot_loc['Game Date'].str[:4]
    shot_loc['Year'] = pd.to_numeric(shot_loc['Year'])
    shot_loc = shot_loc[(shot_loc['Year'] >= 2000) & (shot_loc['Year'] <= 2020)]
    
    return shot_loc

shot_loc = load_data('NBA Shot Locations 1997 - 2020.csv')

st.title("Analyse des tirs de joueurs NBA :basketball:")

with st.sidebar:
    st.title("Menu")
    selected = st.selectbox(
        "Sélectionnez une section",
        [
            "Description et objectif du projet",
            "Données",
            "Analyse exploratoire",
            "Visualisation des données",
            "Préparation des données",
            "Modélisation",
            "Prédiction",          
            "Conclusion et Perspectives"
        ],
        index=1
    )

##### Affichage'option sélectionnée#####
st.write(f"Vous avez sélectionné : {selected}")

if selected == "Description et objectif du projet":
    st.markdown('Projet réalisé par : Quentin Viriot')
    st.markdown('28 Août 2023')
    st.header('Description et objectif du projet :')
    st.markdown("\n")
    st.write(
        '''Les sports américains sont très friands de statistiques, et la NBA (National Basketball Association) ne fait pas exception à la règle. 
Le développement constant des nouvelles technologies et des outils numériques permet désormais de suivre en temps réel les déplacements de tous les joueurs sur un terrain de basketball. Les données recueillies sont ainsi très nombreuses et riches.'''
    )
    
    st.markdown("\n")
    st.write('''Le but de ce projet est de :''')
    st.markdown(
        """
    * Comparer les tirs (fréquence et efficacité par situation de jeu et par localisation sur le terrain) de 20 des meilleurs joueurs de NBA du 21ème siècle selon un classement ESPN
    * Pour chacun de ces 20 joueurs, estimer à l'aide d'un modèle la probabilité de réussite de leurs tirs en fonction de différents paramètres"""
    )
    st.markdown("\n")
    st.write("Nos 20 joueurs sont :")
    st.markdown(
        """
    * LeBron James
    * Kobe Bryant
    * Tim Duncan
    * Shaquille O'Neal
    * Stephen Curry
    * Kevin Durant
    * Dwyane Wade
    * Giannis Antetokounmpo
    * Kevin Garnett
    * Dirk Nowitzki
    * Kawhi Leonard
    * Allen Iverson
    * Steve Nash 
    * Tony Parker
    * Damian Lillard
    * Paul Pierce
    * Russel Westbrook
    * Jason Kidd
    * Ray Allen
    * James Harden\n
        """)
    
#####DONNEES#####

if selected == "Données":
    st.title('Les données :')
    st.info("Les données proviennent du site Kaggle https://www.kaggle.com/jonathangmwl/nba-shot-locations.")
    st.write("Nous restreignons notre étude au panel de 20 joueurs désignés précédemment. Nous avons donc filtré nos données à l'aide de la variable 'Player', pour les saisons de 2000 à 2020")
    st.write("Le tableau suivant récapitule les données présentes :")
   
    st.dataframe(shot_loc.head())
    st.markdown("\n")
    st.write("La figure suivante représente le pourcentage de données associées à chaque joueur :")
    
    fig = px.histogram(shot_loc, y="Player", title="Modalités de la variable Player:", histnorm='percent')
    fig.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey', showline=True, linewidth=2, linecolor='black')
    fig.update_layout(bargap=0.2, title={"x": 0.5}, grid_xaxes=list('x'), plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("\n")
    st.write("Nous n’avons pas trouvé de doublons ni de données manquantes.")
st.markdown("\n")
st.markdown("\n")

##### ANALYSE EXPLORATOIRE #####

if selected == "Analyse exploratoire":
    st.title('Analyse exploratoire des données :')
    st.markdown("\n")
    st.info("""Dans cette partie, nous faisons une analyse des variables les plus représentatives du jeu de données.""")
    st.markdown("\n")
    
    
    st.write("Représentation des modalités de la variable cible :")
    fig3 = px.histogram(shot_loc, y="Shot Made Flag", title="Modalités de la variable Shot Made Flag :", histnorm='percent')
    fig3.update_yaxes(categoryorder="total ascending", nticks=3, showline=True, linewidth=2, linecolor='black')
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey', showline=True, linewidth=2, linecolor='black')
    fig3.update_layout(bargap=0.2, title={"x": 0.5}, grid_xaxes=list('x'), plot_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("\n")

    st.write("La figure suivante représente les modalités de la variable Action type :")
    fig4 = px.histogram(shot_loc, y="Action Type", title="Modalités de la variable Action Type :", histnorm='percent')
    fig4.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black')
    fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey', showline=True, linewidth=2, linecolor='black')
    fig4.update_layout(bargap=0.2, title={"x": 0.5}, grid_xaxes=list('x'), plot_bgcolor='white')
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("\n")

    st.write("La figure suivante représente les modalités de la variable Shot Type :")
    fig5 = px.histogram(shot_loc, y="Shot Type", title="Modalités de la variable Shot Type :", histnorm='percent')
    fig5.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black')
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey', showline=True, linewidth=2, linecolor='black')
    fig5.update_layout(bargap=0.2, title={"x": 0.5}, grid_xaxes=list('x'), plot_bgcolor='white')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("\n")

    st.write("La figure suivante représente les modalités de la variable Season Type :")
    fig6 = px.histogram(shot_loc, y="Season Type", title="Modalités de la variable Season Type :", histnorm='percent')
    fig6.update_yaxes(categoryorder="total ascending", showline=True, linewidth=2, linecolor='black')
    fig6.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Lightgrey', showline=True, linewidth=2, linecolor='black')
    fig6.update_layout(bargap=0.2, title={"x": 0.5}, grid_xaxes=list('x'), plot_bgcolor='white')
    st.plotly_chart(fig6, use_container_width=True)




##### VISUALISATION DES DONNEES #####

if selected == "Visualisation des données":
    st.title('Visualisation')
    st.markdown("\n")
    st.info("""Ici, nous présentons les différentes visualisations réalisées pour chaque joueur""")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    

    st.write("Cartes des tirs des joueurs de la NBA :")

    court_shapes = []
    
    outer_lines_shape = dict(
        type='rect',
        xref='x',
        yref='y',
        x0='-250',
        y0='-47.5',
        x1='250',
        y1='422.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(outer_lines_shape)

    #Hoop Shape
    hoop_shape = dict(
        type='circle',
        xref='x',
        yref='y',
        x0='7.5',
        y0='7.5',
        x1='-7.5',
        y1='-7.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )   
    )
    court_shapes.append(hoop_shape)

    #Basket Backboard
    backboard_shape = dict(
        type='rect',
        xref='x',
        yref='y',
        x0='-30',
        y0='-7.5',
        x1='30',
        y1='-6.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        ),
        fillcolor='rgba(10, 10, 10, 1)'
    )
    court_shapes.append(backboard_shape)

    #Outer Box of Three-Second Area
    outer_three_sec_shape = dict(
        type='rect',
        xref='x',
        yref='y',
        x0='-80',
        y0='-47.5',
        x1='80',
        y1='143.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(outer_three_sec_shape)

    #Inner Box of Three-Second Area
    inner_three_sec_shape = dict(
        type='rect',
        xref='x',
        yref='y',
        x0='-60',
        y0='-47.5',
        x1='60',
        y1='143.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(inner_three_sec_shape)
    
    #Three Point Line (Left)
    left_line_shape = dict(
        type='line',
        xref='x',
        yref='y',
        x0='-220',
        y0='-47.5',
        x1='-220',
        y1='92.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(left_line_shape)

    #Three Point Line (Right)
    right_line_shape = dict(
        type='line',
        xref='x',
        yref='y',
        x0='220',
        y0='-47.5',
        x1='220',
        y1='92.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )   
    )
    court_shapes.append(right_line_shape)

    #Three Point Line Arc
    three_point_arc_shape = dict(
        type='path',
        xref='x',
        yref='y',
        path='M -220 92.5 C -70 300, 70 300, 220 92.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(three_point_arc_shape)

    #Center Circle
    center_circle_shape = dict(
        type='circle',
        xref='x',
        yref='y',
        x0='60',
        y0='482.5',
        x1='-60',
        y1='362.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(center_circle_shape)

    #Restraining Circle
    res_circle_shape = dict(
        type='circle',
        xref='x',
        yref='y',
        x0='20',
        y0='442.5',
        x1='-20',
        y1='402.5',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(res_circle_shape)

    #Free Throw Circle
    free_throw_circle_shape = dict(
        type='circle',
        xref='x',
        yref='y',
        x0='60',
        y0='200',
        x1='-60',
        y1='80',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )
    court_shapes.append(free_throw_circle_shape)

    #Restricted Area
    res_area_shape = dict(
        type='circle',
        xref='x',
        yref='y',
        x0='40',
        y0='40',
        x1='-40',
        y1='-40',
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1,
            dash='dot'
        )
    )
    court_shapes.append(res_area_shape)

saison = st.selectbox(label="Choisissez la saison :", options=['2000/2001', '2019/2020'])


court_shapes = []

if saison == '2000/2001':
    season20002001 = shot_loc[(shot_loc['Year'] == 2000) | (shot_loc['Year'] == 2001)]
    
    def updateVisibility(selectedPlayer):
        visibilityValues = []
        for player in list(season20002001['Player'].unique()):
            if player == selectedPlayer:
                visibilityValues.append(True)
                visibilityValues.append(True)
            else:
                visibilityValues.append(False)
                visibilityValues.append(False)
        return visibilityValues

    data = []
    buttons_data = []
    for player in list(season20002001['Player'].unique()):
        shot_trace_made = go.Scatter(
            x = season20002001[(season20002001['Shot Made Flag'] == 1) & (season20002001['Player'] == player)]['X Location'],
            y = season20002001[(season20002001['Shot Made Flag'] == 1) & (season20002001['Player'] == player)]['Y Location'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(63, 191, 63, 0.9)',
            ), 
            name = 'Made',
            text = season20002001[(season20002001['Shot Made Flag'] == 1) & (season20002001['Player'] == player)],
            textfont = dict(
                color = 'rgba(75, 85, 102,0.7)'
            ),
            visible = (player == player)  
        )

        shot_trace_missed = go.Scatter(
            x = season20002001[(season20002001['Shot Made Flag'] == 0) & (season20002001['Player'] == player)]['X Location'],
            y = season20002001[(season20002001['Shot Made Flag'] == 0) & (season20002001['Player'] == player)]['Y Location'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(241, 18, 18, 0.9)',
            ), 
            name = 'Missed',
            text = season20002001[(season20002001['Shot Made Flag'] == 1) & (season20002001['Player'] == player)],
            textfont = dict(
                color = 'rgba(75, 85, 102,0.7)'
            ),
            visible = (player == player)  
        )

        data.append(shot_trace_made)
        data.append(shot_trace_missed)

        buttons_data.append(
            dict(
                label = player,
                method = 'update',
                args = [{'visible': updateVisibility(player)}]
            )
        )

    updatemenus = list([
        dict(
            active = 0,
            buttons = buttons_data,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 0.65,
            xanchor = 'left',
            y = 1.2,
            yanchor = 'top',
            font = dict(
                size = 14
            )
        )
    ])

    layout = go.Layout(
        title = '<b>Shot Chart - Season 2000/2001</b>',
        titlefont = dict(size = 17),
        hovermode = 'closest',
        updatemenus = updatemenus,
        showlegend = True,
        height = 600,
        width = 600, 
        shapes = court_shapes,
        xaxis = dict(showticklabels = False),
        yaxis = dict(showticklabels = False)
    )

    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(plot_bgcolor = 'rgba(255,255,255,1)')
    st.write(fig)

if saison == '2019/2020':
    season20192020 = shot_loc[(shot_loc['Year'] == 2019) | (shot_loc['Year'] == 2020)]
    
    def updateVisibility(selectedPlayer):
        visibilityValues = []
        for player in list(season20192020['Player'].unique()):
            if player == selectedPlayer:
                visibilityValues.append(True)
                visibilityValues.append(True)
            else:
                visibilityValues.append(False)
                visibilityValues.append(False)
        return visibilityValues

    data = []
    buttons_data = []
    for player in list(season20192020['Player'].unique()):
        shot_trace_made = go.Scatter(
            x = season20192020[(season20192020['Shot Made Flag'] == 1) & (season20192020['Player'] == player)]['X Location'],
            y = season20192020[(season20192020['Shot Made Flag'] == 1) & (season20192020['Player'] == player)]['Y Location'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(63, 191, 63, 0.9)',
            ), 
            name = 'Made',
                        text = season20192020[(season20192020['Shot Made Flag'] == 1) & (season20192020['Player'] == player)],
            textfont = dict(
                color = 'rgba(75, 85, 102,0.7)'
            ),
            visible = (player == player)  
        )

        shot_trace_missed = go.Scatter(
            x = season20192020[(season20192020['Shot Made Flag'] == 0) & (season20192020['Player'] == player)]['X Location'],
            y = season20192020[(season20192020['Shot Made Flag'] == 0) & (season20192020['Player'] == player)]['Y Location'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(241, 18, 18, 0.9)',
            ), 
            name = 'Missed',
            text = season20192020[(season20192020['Shot Made Flag'] == 1) & (season20192020['Player'] == player)],
            textfont = dict(
                color = 'rgba(75, 85, 102,0.7)'
            ),
            visible = (player == player)  
        )

        data.append(shot_trace_made)
        data.append(shot_trace_missed)

        buttons_data.append(
            dict(
                label = player,
                method = 'update',
                args = [{'visible': updateVisibility(player)}]
            )
        )

    updatemenus = list([
        dict(
            active = 0,
            buttons = buttons_data,
            direction = 'down',
            pad = {'r': 10, 't': 10},
            showactive = True,
            x = 0.65,
            xanchor = 'left',
            y = 1.2,
            yanchor = 'top',
            font = dict(
                size = 14
            )
        )
    ])

    layout = go.Layout(
        title = '<b>Shot Chart - Season 2019/2020</b>',
        titlefont = dict(size = 17),
        hovermode = 'closest',
        updatemenus = updatemenus,
        showlegend = True,
        height = 600,
        width = 600, 
        shapes = court_shapes,
        xaxis = dict(showticklabels = False),
        yaxis = dict(showticklabels = False)
    )

    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(plot_bgcolor = 'rgba(255,255,255,1)')
    st.write(fig)

st.write("Relation entre le type de tir et le type d'action pour chaque joueur (Top 5)")
    
top_action_types = shot_loc['Action Type'].value_counts().nlargest(5).index
shot_loc_filtered = shot_loc[shot_loc['Action Type'].isin(top_action_types)]
players = shot_loc_filtered['Player'].unique()
selected_player = st.selectbox("Sélectionnez un joueur", players)
fig1 = px.scatter(data_frame=shot_loc_filtered[shot_loc_filtered['Player'] == selected_player],
                     x='Shot Distance', y='Action Type', color='Player',
                     title=f"Relation entre le type de tir et le type d'action pour {selected_player}",
                     labels={'Shot Distance': "Distance de tir", 'Action Type': "Type d'action"})
fig1.update_layout(title_x=0.5)
st.plotly_chart(fig1)

st.write("La visualisation suivantes représente la moyenne de points par joueur sur toutes les saisons auxquelles ils ont participé:")
image = Image.open("moyenne FG%.png")  
st.image(image)

st.write("La figure suivante représente la moyenne de points par joueur:")
image1 = Image.open("moyennespoints.png")  
st.image(image1)

##### Prépa Des Données #####

if selected == "Préparation des données":
    st.title('Préparation des données :')
    st.markdown("\n")
    
    st.write("**Dans cette section, nous avons détaillé le cheminement complet du projet**")
    st.markdown("\n")

    st.write("1. **Chargement des données et sélection des joueurs :**")
    st.write("- Les données sont chargées à partir d'un fichier CSV intitulé 'NBA Shot Locations 1997 - 2020.csv'.")
    st.write("- Une liste des meilleurs joueurs est définie dans la variable 'top_players'.")
    st.write("- Les données pour les joueurs sélectionnés sont extraites dans le DataFrame 'shot_loc_top_players'.")
    st.markdown("\n")

    st.write("2. **Sélection des variables pertinentes :**")
    st.write("- Une liste 'variables_selection' est créée pour sélectionner les variables d'intérêt, telles que la date du match, le type d'action, le résultat du tir, etc.")
    st.write("- Le DataFrame 'shot_loc' est créé en utilisant les variables sélectionnées. La colonne 'Player Name' est renommée en 'Player'.")
    st.markdown("\n")

   
    st.write("3. **Traitement des dates et filtrage par année :**")
    st.write("- La colonne 'Game Date' est convertie en chaînes de caractères.")
    st.write("- La colonne 'Year' est créée en extrayant les 4 premiers caractères de la colonne 'Game Date' et en la convertissant en numérique.")
    st.write("- Les données sont filtrées pour conserver uniquement les années entre 2000 et 2020.")
    st.markdown("\n")

   
    st.write("4. **Visualisation de la distribution des variables catégorielles et numériques :**")
    st.write("- Les variables catégorielles sont visualisées à l'aide de 'sns.countplot' et 'sns.stripplot' pour montrer la distribution des catégories.")
    st.write("- Les variables numériques sont visualisées à l'aide d'histogrammes avec 'sns.histplot'.")
    st.markdown("\n")

   
    st.write("5. **Encodage des variables catégorielles et préparation des données pour la modélisation :**")
    st.write("- Les données sont encodées en utilisant 'pd.get_dummies' pour les variables catégorielles.")
    st.write("- Les jeux de données pour la saison régulière et les playoffs sont préparés en sélectionnant les variables appropriées.")
    st.markdown("\n")

   
    st.write("6. **Modélisation :**")
    st.write("- Les données sont divisées en ensembles d'entraînement et de test pour la saison régulière et les playoffs.")
    st.write("- Les modèles de classification tels que la régression logistique, la forêt aléatoire et le KNN sont entraînés et évalués sur les ensembles de test.")
    st.write("- Les performances des modèles sont évaluées en utilisant des métriques telles que l'exactitude, le rappel, la précision et le score F1.")
    st.markdown("\n")

    
    st.write("7. **Optimisation des hyperparamètres avec GridSearchCV :**")
    st.write("- GridSearchCV est utilisé pour rechercher les meilleurs hyperparamètres pour le modèle de la forêt aléatoire.")
    st.markdown("\n")

   
    st.write("8. **Validation croisée K-Fold et Shuffle-Split :**")
    st.write("- La validation croisée K-Fold est utilisée pour évaluer les performances des modèles sur plusieurs plis.")
    st.write("- Shuffle-Split est utilisé pour effectuer une validation croisée en mélangeant les données.")
    st.markdown("\n")

    # Séparation des données pour la saison régulière
    st.write("Séparation des données pour la saison régulière :")
    st.write("- Les données sont déjà préalablement préparées dans le DataFrame 'shot_loc_active_players'.")
    st.write("- Un sous-ensemble des données pour la saison régulière est créé en filtrant les lignes où la colonne 'Season Type' est égale à 'Regular Season'. Les données correspondantes sont stockées dans le DataFrame 'shot_loc_regular_season'.")
    st.markdown("\n")

    # Séparation des données pour les playoffs
    st.write("Séparation des données pour les playoffs :")
    st.write("- De manière similaire, un sous-ensemble des données pour les playoffs est créé en filtrant les lignes où la colonne 'Season Type' est égale à 'Playoffs'. Les données correspondantes sont stockées dans le DataFrame 'shot_loc_playoffs'.")
    st.markdown("\n")

    # Encodage des variables catégorielles pour la saison régulière et les playoffs
    st.write("Encodage des variables catégorielles pour la saison régulière et les playoffs :")
    st.write("- Les DataFrames 'shot_loc_regular_season' et 'shot_loc_playoffs' sont encodés en utilisant 'pd.get_dummies' pour les variables catégorielles spécifiques à chaque ensemble.")
    st.markdown("\n")

    # Séparation en ensembles d'entraînement et de test pour la saison régulière
    st.write("Séparation en ensembles d'entraînement et de test pour la saison régulière :")
    st.write("- Les features ('X_reg_season') et les labels ('y_reg_season') sont extraits du DataFrame 'shot_loc_regular_season_encoded'.")
    st.write("- Les données sont divisées en ensembles d'entraînement ('X_train_reg', 'y_train_reg') et de test ('X_test_reg', 'y_test_reg') en utilisant la fonction 'train_test_split'.")
    st.markdown("\n")

    # Séparation en ensembles d'entraînement et de test pour les playoffs
    st.write("Séparation en ensembles d'entraînement et de test pour les playoffs :")
    st.write("- De manière similaire, les features ('X_playoffs') et les labels ('y_playoffs') sont extraits du DataFrame 'shot_loc_playoffs_encoded'.")
    st.write("- Les données sont divisées en ensembles d'entraînement ('X_train_playoffs', 'y_train_playoffs') et de test ('X_test_playoffs', 'y_test_playoffs') en utilisant la fonction 'train_test_split'.")
    st.markdown("\n")

    # Mise à l'échelle des features pour la saison régulière et les playoffs
    st.write("Mise à l'échelle des features pour la saison régulière et les playoffs :")
    st.write("- Les ensembles d'entraînement ('X_train_reg', 'X_train_playoffs') sont mis à l'échelle à l'aide du 'StandardScaler'.")
    st.write("- Les ensembles de test ('X_test_reg', 'X_test_playoffs') sont également mis à l'échelle en utilisant la même instance de 'StandardScaler' qui a été ajustée sur les données d'entraînement.")
    st.markdown("\n")
   
    # Réduction de dimension avec PCA
    st.write("9. **Réduction de dimension avec PCA :**")
    st.write("- Les features sont réduites en dimension en utilisant l'analyse en composantes principales (PCA).")
    st.write("- Le PCA est ajusté sur les données d'entraînement de la saison régulière.")
    st.write("- Les composantes principales sont transformées sur les ensembles d'entraînement et de test de la saison régulière ainsi que sur les ensembles de test des playoffs.")
    st.write("- La variance expliquée par chaque composante est affichée pour aider à comprendre l'importance de chaque composante.")
    st.markdown("\n")
    

##### MODELISATION #####

if selected == "Modélisation":
    st.header("Méthode:")
    st.write("Notre projet de recherche s'apparente à un problème de Machine Learning, de classification :")
    st.markdown("""
    * La classe 1 : Tir réussi
    * La classe 0 : Tir raté
    """)
    st.markdown("\n")
    
    st.header("Mise en place des algorithmes :")
    st.write("Nous avons utilisé les modèles suivants pour chacun des 20 joueurs:")
    st.write("* Régression Logistique") 
    st.write("* Forêts aléatoires")
    st.write("* K-plus proches voisins (KNN)")
    
    st.subheader("Les modèles :")
    st.write("Nous avons utilisé deux modèles :")
    st.write("* Modèles par joueur en saison régulière avec PCA")
    st.write("* Modèles par joueur en playoffs avec PCA")
    st.info("L'intérêt d'utiliser la PCA est de réduire considérablement le temps d'apprentissage des modèles tout en améliorant la performance.")
    
    st.subheader("Les Résultats pour la Saison Régulière :")
    st.write("Notre étude des matchs de la saison régulière qui représentent la majorité des données dont nous disposons (Environ 87%)")
    imageSR = Image.open("ModèlesSR.png")  
    st.image(imageSR)
    st.markdown("\n")
    st.write("Les performances de nos modèles se situent globalement entre 48%, et 66%. Elles sont assez variables selon les joueurs et le modèle choisi.")
    st.markdown("\n")
    
    st.subheader("Modèles appliqués sur Allen Iverson et Kobe Bryant en saison régulière :")
    imageAIKB = Image.open("AIKB.png")  
    st.image(imageAIKB)
    st.markdown("\n")

    st.header(" Modèles par joueur avec les données concernant les playoffs avec PCA :")
    st.write("Nous avons procédé de la même manière que pour le modèle précédent mais en l'appliquants sur les données des playoffs seulement.")
    imagePO = Image.open("modelesPO.png")  
    st.image(imagePO)
    st.markdown("\n")
    st.write("Les performances de nos modèles se situent globalement entre 47%, et 66%. Elles sont assez variables selon les joueurs et le modèle choisi.")
    st.markdown("\n")
    
    st.subheader("Modèles appliqués sur Allen Iverson et Kobe Bryant en PlayOffs :")
    imageKBAIPO = Image.open("KBAIPO.png")  
    st.image(imageKBAIPO)
    st.markdown("\n")
    
    st.header("Modèle sélectionné :") 
    st.subheader("*Random Forest :")
    st.write("Le choix d'utiliser le modèle de Random Forest est justifié par plusieurs raisons : ")
    st.write("Les Forêts aléatoires ne nécessitent généralement pas beaucoup de pré-traitement des données et sont robustes face à des problèmes de sur-apprentissage (overfitting).")
    st.write("Performance acceptable de la métrique accuracy")
    st.write("Les résultats du F1-score sont raisonnables")
    st.markdown("\n")
    st.write("Cependant nous avons tout de même voulu optimiser la performances de notre modèle à l'aide d'une grille de recherche pour trouver les hyperparamètres les plus adaptés.")
    st.markdown("\n")
    st.write("La recherche sur grille n'a pas eu d'impact significatifs sur les performances")
    st.markdown("\n")
    
##### Modèle de prédiction en fonctionnement #####
if selected == "Prédiction":
    st.header("Prédiction de réussite de tir")
    
    # Widget de sélection du joueur
    selected_player = st.radio("Sélectionnez un joueur", players)

    # Widget de sélection du type de saison
    season_type = st.radio("Saison", ["Saison Régulière", "Playoffs"])
    
    # Widget de sélection de la distance de tir
    selected_distance = st.slider("Distance de tir (pieds)", min_value=0, max_value=50, step=1)
    if st.button("Prédire"):
        # Préparez les données pour la prédiction
        data_for_prediction = shot_loc.drop(['Shot Made Flag'], axis=1)

        # Appliquez le même prétraitement sur les données que lors de l'entraînement
        numeric_columns = data_for_prediction[['Shot Distance', 'X Location', 'Y Location']]
        scaler = StandardScaler()
        scaled_numeric_columns = scaler.fit_transform(numeric_columns)

        categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
        data_for_prediction_encoded = pd.get_dummies(data_for_prediction, columns=categorical_variables)
        columns_for_prediction = ['Shot Distance', 'X Location', 'Y Location'] + data_for_prediction_encoded.columns.tolist()
        columns_for_prediction = list(set(columns_for_prediction))

        data_for_prediction_encoded = data_for_prediction_encoded[columns_for_prediction]
        data_for_prediction_encoded2 = data_for_prediction_encoded[columns_for_prediction]

        if 'Season Type_Playoffs' in data_for_prediction_encoded.columns:
            data_for_prediction_encoded = data_for_prediction_encoded.drop(['Season Type_Playoffs'], axis=1)

        if season_type == "Saison Régulière":
            model = rf_SR
            prediction = model.predict(data_for_prediction_encoded)
            proportion_reussite = sum(prediction) / len(prediction)
            if proportion_reussite >= 0.5:
                st.success("Le tir est prédit comme réussi pour la saison régulière.")
            else:
                st.error("Le tir est prédit comme raté pour la saison régulière.")
        if season_type == "Playoffs":
            model2 = rf_PO
            prediction2 = model2.predict(data_for_prediction_encoded2)
            proportion_reussite2 = sum(prediction2) / len(prediction2)
            if proportion_reussite2 >= 0.5:
                st.success("Le tir est prédit comme réussi pour les playoffs.")
            else:
                st.error("Le tir est prédit comme raté pour les playoffs.")







    
if selected == "Conclusion et Perspectives":
    st.header("Bilan :")
    st.markdown("\n")
    st.write("L'objectif du projet était d'estimer, à l'aide d'un modèle, la probabilité de réussite des tirs des 20 meilleurs joueurs de la NBA selon le classement ESPN, en utilisant différents paramètres.")
    st.write("Nos analyses ont mis en évidence plusieurs résultats :")
    st.markdown("""
    * En fonction des joueurs et des modèles, nous avons obtenu une précision minimale de 55 % et maximale de 70 %.
    * La réduction de dimension (PCA) et la sélection de variables ont considérablement réduit les temps d'exécution.
    * Ces méthodes n'ont pas eu d'impact significatif sur les performances.
    * L'utilisation de modèles avancés (Bagging et Boosting) n'a pas eu d'impact révélateur sur les performances.
    * Nos modèles prédisent mieux les tirs ratés que les tirs réussis.
    """)
    st.markdown("\n")
    st.write("Les résultats sont corrects, mais plusieurs critiques peuvent être formulées :")
    st.write("")
    st.markdown(
        """
    * La disparité des données disponibles selon les joueurs a un impact sur les résultats.
    * Les performances des modèles sont encourageantes, mais elles peuvent encore être améliorées.
    """
    )

    st.header("*Les axes d'amélioration*")
    st.markdown("\n")
    st.write("Nous aurions pu ajuster les hyperparamètres de chaque modèle sélectionné afin de les comparer plus en détail.")
    st.markdown("\n")
    st.write("Nous pourrions incorporer les variables liées au temps restant dans un quart-temps ou à la fin d'un match, lorsqu'un joueur prend un tir.")
    st.markdown("\n")
    st.write("Nous aurions également pu tester les modèles en intégrant et en comparant la réussite aux tirs des joueurs sélectionnés en fonction de l'équipe à laquelle ils sont opposés.")
    st.markdown("\n")
    st.write("Utiliser le Web scraping afin d'augmenter nos données dans le but d'obtenir une homogénéité des données disponibles en fonction des joueurs.")
    st.markdown("\n")
    st.write("Nous aurions pu utiliser d'autres modèles tels que le SVM.")
    st.markdown("\n")
    st.write("Enfin, nous aurions pu incorporer des données sur les impacts défensifs (comme la 'defensive win share'), et étudier l'impact défensif pour chaque équipe opposée aux joueurs sélectionnés.")
    st.markdown("\n")
    st.write("Avec plus de temps, nous aurions pu réussir à analyser et intégrer d'autres variables nous permettant d'améliorer nos modèles de prédiction. Ainsi, nous aurions pu démontrer que les tirs réussis ou ratés par un joueur ne sont pas seulement dus à la technique de tir, mais sont influencés par de nombreux paramètres extérieurs, plus ou moins difficiles à quantifier.")
    st.subheader("*Conclusion")
    st.write("Le but de ce projet était de comparer les tirs de 20 des meilleurs joueurs de la NBA du 21ème siècle en termes de fréquence et d'efficacité, en fonction de différentes situations de jeu et de localisations sur le terrain. Pour cela, nous avons élaboré des modèles visant à estimer la probabilité que le tir de chaque joueur rentre dans le panier, en utilisant diverses métriques comme variables prédictives. L'évolution des fréquences et de l'efficacité des tirs a été représentée à l'aide de visuels dans la première partie.")
    st.write("Au cours de ce projet, nous avons suivi différentes étapes de traitement des données pour sélectionner les joueurs cibles et les variables pertinentes. Nous avons visualisé la distribution des données et analysé les relations entre les types d'action et les distances de tir pour chaque joueur. Ces étapes nous ont permis de mieux comprendre les caractéristiques des tirs des joueurs étudiés.")
    st.write("En ce qui concerne la modélisation, nous avons utilisé des algorithmes tels que la régression logistique, le Random Forest et le KNN pour prédire la probabilité de réussite d'un tir. Nous avons également appliqué la méthode d'Analyse en Composantes Principales (PCA) pour réduire la dimensionnalité des données et faciliter l'entraînement des modèles.")
    st.write("Cependant, il est important de noter que malgré nos efforts, certains paramètres difficiles à quantifier ou à trouver, tels que la pression défensive, la force du contact avec un défenseur, l'orientation des appuis du tireur, etc., pourraient avoir une influence significative sur l'issue d'un tir. Par conséquent, malgré les performances encourageantes de nos modèles, il est essentiel de garder à l'esprit que d'autres facteurs non pris en compte peuvent également jouer un rôle clé dans la réussite d'un tir.")
    st.write("En conclusion, ce projet nous a permis de comparer les tirs des meilleurs joueurs de la NBA du 21ème siècle et de développer des modèles prédictifs pour estimer la probabilité de réussite de leurs tirs (utilisant 'NBA Shot Locations 1997 - 2020'). Bien que le modèle obtenu soit prometteur, des recherches futures pourraient être menées pour inclure davantage de paramètres et de variables afin d'améliorer encore la fiabilité de nos prédictions.")
