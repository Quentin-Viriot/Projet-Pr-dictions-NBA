import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt
from joblib import dump, load

shot_location = pd.read_csv('NBA Shot Locations 1997 - 2020.csv', sep=',')

top_players = ['LeBron James', 'Kobe Bryant', 'Tim Duncan', "Shaquille O'Neal", 'Stephen Curry', 'Kevin Durant', 'Dwyane Wade', 'Giannis Antetokounmpo', 'Kevin Garnett', 'Dirk Nowitzki', 'Kawhi Leonard', 'Allen Iverson', 'Steve Nash', 'Tony Parker', 'Damian Lillard', 'Paul Pierce', 'Jason Kidd', 'Russell Westbrook', 'Ray Allen', 'James Harden']
shot_location.info()
shot_loc_top_players = shot_location[shot_location['Player Name'].isin(top_players)]

variables_selection = ['Game Date','Action Type','Shot Made Flag', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 'X Location', 'Y Location', 'Season Type','Player Name']

shot_loc = shot_loc_top_players[variables_selection]
shot_loc.rename(columns={'Player Name': 'Player'}, inplace=True)

print(shot_loc)
shot_loc.info()
shot_loc.isna().sum()
shot_loc.head()

shot_loc['Game Date'] = shot_loc['Game Date'].astype(str)
shot_loc['Year'] = shot_loc['Game Date'].str[:4]
shot_loc['Year'] = pd.to_numeric(shot_loc['Year'])

shot_loc = shot_loc[(shot_loc['Year'] >= 2000) & (shot_loc['Year'] <= 2020)]

print(shot_loc)
shot_loc.info()

#Visualisation distribution
import matplotlib.pyplot as plt

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']

for var in categorical_variables:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, data=shot_loc)
    plt.title(f"Distribution de {var}")
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(10, 6))
sns.stripplot(x='Action Type', data=shot_loc, jitter=True, size=5)
plt.title("Distribution de Action Type")
plt.xticks(rotation=45)
plt.show()

numerical_variables = ['Shot Distance', 'X Location', 'Y Location', 'Year']

for var in numerical_variables:
    plt.figure(figsize=(8, 5))
    sns.histplot(shot_loc[var], bins=20, kde=True)
    plt.title(f"Distribution de {var}")
    plt.xlabel(var)
    plt.ylabel("Count")
    plt.show()

# Filtre des 5 occurrences les plus représentées dans la colonne "Action Type"
top_action_types = shot_loc['Action Type'].value_counts().nlargest(5).index
shot_loc_filtered = shot_loc[shot_loc['Action Type'].isin(top_action_types)]

legend_ax = plt.gca()
legend_ax.set_axis_off()
legend_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure(figsize=(12, 8))
sns.relplot(data=shot_loc_filtered, x='Shot Distance', y='Action Type', hue='Player', col='Player', col_wrap=4, height=3, aspect=1.5, palette='Set1')
plt.xlabel('Type d\'action')
plt.ylabel('Type de tir')
plt.suptitle("Relation entre le type de tir et le type d'action pour chaque joueur (Top 5)")
plt.tight_layout()
plt.show()

shot_loc_filtered_players = shot_loc[shot_loc['Player'].isin(top_players[:5])]
top_action_types = shot_loc_filtered_players['Action Type'].value_counts().nlargest(5).index

for player_name in top_players[:5]:
    shot_loc_filtered = shot_loc_filtered_players[(shot_loc_filtered_players['Player'] == player_name) & (shot_loc_filtered_players['Action Type'].isin(top_action_types))]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=shot_loc_filtered, x='Shot Distance', y='Action Type', hue='Player', palette='Set1')
    plt.title(f"Type de tir par rapport à la distance de tir pour {player_name}")
    plt.xlabel('Distance de tir')
    plt.ylabel('Type d\'action')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

for player_name in top_players[:5]:
    shot_loc_filtered = shot_loc_filtered_players[(shot_loc_filtered_players['Player'] == player_name) ]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=shot_loc_filtered, x='Shot Distance', y='Shot Zone Basic', hue='Player', palette='Set1')
    plt.title(f"Type de tir par rapport à la distance de tir pour {player_name}")
    plt.xlabel('Distance de tir')
    plt.ylabel('Type d\'action')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

class_counts = shot_loc['Shot Made Flag'].value_counts()
print(class_counts)
total_samples = len(shot_loc)
class_proportions = class_counts / total_samples
print(class_proportions)

sns.countplot(x='Shot Made Flag', data=shot_loc)
plt.title("Répartition des classes")
plt.xlabel("Shot Made Flag")
plt.ylabel("Nombre d'occurrences")
plt.show()


#Modélisation 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

shot_loc_active_players = shot_loc[shot_loc['Player'].isin(top_players)]

shot_loc_regular_season = shot_loc_active_players[shot_loc_active_players['Season Type'] == 'Regular Season']
shot_loc_playoffs = shot_loc_active_players[shot_loc_active_players['Season Type'] == 'Playoffs']

variables_selection = ['Game Date', 'Action Type', 'Shot Made Flag', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 'X Location', 'Y Location', 'Season Type', 'Player']
shot_loc_regular_season = shot_loc_regular_season[variables_selection]
shot_loc_playoffs = shot_loc_playoffs[variables_selection]

numeric_columns = shot_loc_active_players[['Shot Distance', 'X Location', 'Y Location']]
scaler = StandardScaler()
scaled_numeric_columns = scaler.fit_transform(numeric_columns)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_numeric_columns)
shot_loc_active_players['PCA Component 1'] = pca_result[:, 0]
shot_loc_active_players['PCA Component 2'] = pca_result[:, 1]

categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_regular_season_encoded = pd.get_dummies(shot_loc_regular_season, columns=categorical_variables)
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_reg_season = shot_loc_regular_season_encoded.drop(columns=['Shot Made Flag'])
y_reg_season = shot_loc_regular_season_encoded['Shot Made Flag']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_season, y_reg_season, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)
dump(scaler,"scalerSR")
categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_regular_season_encoded = pd.get_dummies(shot_loc_regular_season, columns=categorical_variables)
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_reg_season = shot_loc_regular_season_encoded.drop(columns=['Shot Made Flag'])
y_reg_season = shot_loc_regular_season_encoded['Shot Made Flag']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_season, y_reg_season, test_size=0.2, random_state=42)


print(X_train_reg.columns.tolist())
columns_prediction = ['Action Type_Driving Jump shot', 'Action Type_Jump Hook Shot', 'Action Type_Cutting Layup Shot', 'Action Type_Driving Finger Roll Shot', 'Shot Type_2PT Field Goal', 'Action Type_Running Pull-Up Jump Shot', 'Player_Kevin Durant', 'Action Type_Putback Dunk Shot', 'Season Type_Playoffs', 'Action Type_Jump Bank Hook Shot', 'Action Type_Cutting Dunk Shot', 'Shot Zone Basic_Right Corner 3', 'Action Type_Driving Floating Jump Shot', 'Shot Distance', 'Action Type_Running Reverse Dunk Shot', 'Action Type_Hook Bank Shot', 'Action Type_No Shot', 'Year', 'Shot Zone Area_Right Side(R)', 'Y Location', 'Shot Zone Range_24+ ft.', 'Player_Tony Parker', 'Action Type_Running Alley Oop Layup Shot', 'Action Type_Turnaround Finger Roll Shot', 'Action Type_Turnaround Bank shot', 'Action Type_Running Hook Shot', 'Shot Zone Basic_Backcourt', 'Player_Stephen Curry', 'Action Type_Floating Jump shot', 'Action Type_Follow Up Dunk Shot', 'Action Type_Turnaround Fadeaway shot', 'Player_Dwyane Wade', 'Shot Zone Basic_Mid-Range', 'Shot Zone Area_Right Side Center(RC)', 'Shot Type_3PT Field Goal', 'Action Type_Driving Bank Hook Shot', 'Shot Zone Area_Center(C)', 'Action Type_Driving Floating Bank Jump Shot', 'Action Type_Driving Layup Shot', 'Action Type_Alley Oop Dunk Shot', 'Player_James Harden', 'Action Type_Running Slam Dunk Shot', 'Player_Tim Duncan', 'Action Type_Turnaround Hook Shot', 'Action Type_Putback Reverse Dunk Shot', 'Player_Kevin Garnett', 'Player_Giannis Antetokounmpo', 'Shot Zone Area_Left Side(L)', 'Player_Dirk Nowitzki', 'Action Type_Running Bank shot', 'Player_Jason Kidd', 'Action Type_Turnaround Jump Shot', 'Action Type_Hook Shot', 'Action Type_Turnaround Bank Hook Shot', 'Action Type_Pullup Bank shot', 'Shot Zone Basic_In The Paint (Non-RA)', 'Action Type_Dunk Shot', 'Action Type_Slam Dunk Shot', 'Shot Zone Area_Left Side Center(LC)', 'Action Type_Jump Bank Shot', 'Action Type_Running Dunk Shot', 'Player_Damian Lillard', 'Action Type_Cutting Finger Roll Layup Shot', 'Shot Zone Range_16-24 ft.', 'Action Type_Step Back Bank Jump Shot', 'Action Type_Running Tip Shot', 'Player_Allen Iverson', "Player_Shaquille O'Neal", 'X Location', 'Action Type_Finger Roll Layup Shot', 'Shot Zone Basic_Above the Break 3', 'Action Type_Driving Finger Roll Layup Shot', 'Action Type_Running Jump Shot', 'Action Type_Tip Dunk Shot', 'Action Type_Tip Layup Shot', 'Action Type_Putback Slam Dunk Shot', 'Action Type_Driving Dunk Shot', 'Action Type_Tip Shot', 'Action Type_Driving Reverse Layup Shot', 'Player_LeBron James', 'Action Type_Reverse Dunk Shot', 'Action Type_Reverse Slam Dunk Shot', 'Player_Kawhi Leonard', 'Action Type_Finger Roll Shot', 'Player_Paul Pierce', 'Shot Zone Range_8-16 ft.', 'Season Type_Regular Season', 'Action Type_Running Finger Roll Layup Shot', 'Shot Zone Range_Less Than 8 ft.', 'Action Type_Step Back Jump shot', 'Action Type_Turnaround Fadeaway Bank Jump Shot', 'Player_Ray Allen', 'Shot Zone Range_Back Court Shot', 'Game Date', 'Action Type_Running Bank Hook Shot', 'Action Type_Running Layup Shot', 'Action Type_Running Finger Roll Shot', 'Shot Zone Basic_Left Corner 3', 'Action Type_Pullup Jump shot', 'Action Type_Driving Hook Shot', 'Action Type_Driving Slam Dunk Shot', 'Player_Steve Nash', 'Action Type_Putback Layup Shot', 'Action Type_Fadeaway Bank shot', 'Action Type_Running Reverse Layup Shot', 'Player_Kobe Bryant', 'Action Type_Driving Reverse Dunk Shot', 'Action Type_Running Alley Oop Dunk Shot', 'Action Type_Alley Oop Layup shot', 'Action Type_Layup Shot', 'Action Type_Jump Shot', 'Action Type_Reverse Layup Shot', 'Shot Zone Basic_Restricted Area', 'Player_Russell Westbrook', 'Action Type_Driving Bank shot', 'Shot Zone Area_Back Court(BC)', 'Action Type_Fadeaway Jump Shot']   

# Liste des colonnes de vos données d'entraînement
columns_training = X_train_reg.columns.tolist()

# Recherche de la caractéristique en trop
extra_feature = None
for column in columns_prediction:
    if column not in columns_training:
        extra_feature = column
        break

if extra_feature is not None:
    print("Caractéristique en trop :", extra_feature)
else:
    print("Aucune caractéristique en trop trouvée.")
#Algorithme regression logistique SR
logistic_model_reg = LogisticRegression(max_iter=1000) 

logistic_model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_logistic = logistic_model_reg.predict(X_test_reg_scaled)

print("Metrics for Regular Season - Logistic Regression:")
print("-----------------------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_logistic))
print("Recall:", recall_score(y_test_reg, y_pred_reg_logistic))
print("Precision:", precision_score(y_test_reg, y_pred_reg_logistic))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_logistic))
print(classification_report(y_test_reg, y_pred_reg_logistic))

confusion_mat_reg = confusion_matrix(y_test_reg, y_pred_reg_logistic)

print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_mat_reg)

#Algorithme RandomForest SR

rf_model_reg = RandomForestClassifier()
rf_model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_rf = rf_model_reg.predict(X_test_reg_scaled)
dump(rf_model_reg,'random-forest')

print("Metrics for Regular Season - Random Forest:")
print("-----------------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_rf))
print("Recall:", recall_score(y_test_reg, y_pred_reg_rf))
print("Precision:", precision_score(y_test_reg, y_pred_reg_rf))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_rf))
print(classification_report(y_test_reg, y_pred_reg_rf))
y_pred_reg_rf
confusion_reg_rf = confusion_matrix(y_test_reg, y_pred_reg_rf)
print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_reg_rf)

#Algorithme KNN SR

knn_model_reg = KNeighborsClassifier()
knn_model_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg_knn = knn_model_reg.predict(X_test_reg_scaled)

print("Metrics for Regular Season - KNN:")
print("--------------------------------")
print("Accuracy:", accuracy_score(y_test_reg, y_pred_reg_knn))
print("Recall:", recall_score(y_test_reg, y_pred_reg_knn))
print("Precision:", precision_score(y_test_reg, y_pred_reg_knn))
print("F1-score:", f1_score(y_test_reg, y_pred_reg_knn))
print(classification_report(y_test_reg, y_pred_reg_knn))

confusion_mat_knn = confusion_matrix(y_test_reg, y_pred_reg_knn)
print("Matrice de confusion - Saison régulière")
print("--------------------------------------")
print(confusion_mat_knn)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}
player_scores_global = []

for player in top_players:
    player_data = shot_loc_regular_season_encoded[shot_loc_regular_season_encoded['Player_' + player] == 1]
    X_player = player_data.drop(columns=['Shot Made Flag'])
    y_player = player_data['Shot Made Flag']

    scaler = StandardScaler()
    X_player_scaled = scaler.fit_transform(X_player)

    model_scores_list = []
    for model_name, model in [('Logistic Regression', logistic_model_reg), ('Random Forest', rf_model_reg), ('KNN', knn_model_reg)]:
        model.fit(X_train_reg_scaled, y_train_reg)  # Adjust the model with the regular season data
        y_pred_player = model.predict(X_player_scaled)
        accuracy = accuracy_score(y_player, y_pred_player)
        model_scores_list.append({'Player': player, 'Model': model_name, 'Accuracy': accuracy})

    player_scores_global.extend(model_scores_list)

# Création du DataFrame contenant les scores
scores_df_global = pd.DataFrame(player_scores_global)

# Création du graphique en barres verticales pour les scores de chaque modèle par joueur
plt.figure(figsize=(12, 8))
sns.barplot(data=scores_df_global, x='Player', y='Accuracy', hue='Model')
plt.xlabel('Joueurs')
plt.ylabel('Score de précision')
plt.title('Scores par joueur et par modèle - Tous les joueurs')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Modèle')
plt.tight_layout()
plt.show()



specific_players = ['Allen Iverson', 'Kobe Bryant']

player_scores = []

for player_name in specific_players:
    player_data = shot_loc_regular_season_encoded[shot_loc_regular_season_encoded['Player_' + player_name] == 1]
    X_player = player_data.drop(columns=['Shot Made Flag'])
    y_player = player_data['Shot Made Flag']

    scaler = StandardScaler()
    X_player_scaled = scaler.fit_transform(X_player)

    model_scores_list = []
    for model_name, model in [('Logistic Regression', logistic_model_reg), ('Random Forest', rf_model_reg), ('KNN', knn_model_reg)]:
        model.fit(X_train_reg_scaled, y_train_reg)  # Adjust the model with the regular season data
        y_pred_player = model.predict(X_player_scaled)
        accuracy = accuracy_score(y_player, y_pred_player)
        model_scores_list.append({'Player': player_name, 'Model': model_name, 'Accuracy': accuracy})

    player_scores.extend(model_scores_list)


scores_df_specific = pd.DataFrame(player_scores)

# Création du graphique en barres verticales pour les scores de chaque modèle par joueur
plt.figure(figsize=(12, 8))
sns.barplot(data=scores_df_specific, x='Player', y='Accuracy', hue='Model')
plt.xlabel('Joueurs')
plt.ylabel('Score de précision')
plt.title('Scores par joueur et par modèle - Joueurs spécifiques')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Modèle')
plt.tight_layout()
plt.show()


#Playoffs
numeric_columns1 = shot_loc_playoffs[['Shot Distance', 'X Location', 'Y Location']]
scaler = StandardScaler()
scaled_numeric_columns = scaler.fit_transform(numeric_columns1)
pca = PCA()
pca_result = pca.fit_transform(scaled_numeric_columns)
explained_variances = pca.explained_variance_ratio_

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_numeric_columns)
shot_loc_playoffs['PCA Component 1'] = pca_result[:, 0]
shot_loc_playoffs['PCA Component 2'] = pca_result[:, 1]
dump(pca,'PCA')
categorical_variables = ['Action Type', 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type', 'Player']
shot_loc_playoffs_encoded = pd.get_dummies(shot_loc_playoffs, columns=categorical_variables)

X_playoffs = shot_loc_playoffs_encoded.drop(columns=['Shot Made Flag'])
y_playoffs = shot_loc_playoffs_encoded['Shot Made Flag']
X_train_playoffs, X_test_playoffs, y_train_playoffs, y_test_playoffs = train_test_split(X_playoffs, y_playoffs, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_playoffs_scaled = scaler.fit_transform(X_train_playoffs)
X_test_playoffs_scaled = scaler.transform(X_test_playoffs)
dump(scaler,'scalerPO')
logistic_model_playoffs = LogisticRegression()
logistic_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_logistic = logistic_model_playoffs.predict(X_test_playoffs_scaled)

print("Metrics for Playoffs - Logistic Regression:")
print("-------------------------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_logistic))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_logistic))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_logistic))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_logistic))
print(classification_report(y_test_playoffs, y_pred_playoffs_logistic))

confusion_mat_playoffs = confusion_matrix(y_test_playoffs, y_pred_playoffs_logistic)

print("Matrice de confusion - Playoffs")
print("---------------------------------")
print(confusion_mat_playoffs)

# Algorithme RandomForest Playoffs
rf_model_playoffs = RandomForestClassifier()
rf_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_rf = rf_model_playoffs.predict(X_test_playoffs_scaled)
dump(rf_model_playoffs,'random-forestPO')
print("Metrics for Playoffs - Random Forest:")
print("-------------------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_rf))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_rf))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_rf))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_rf))
print(classification_report(y_test_playoffs, y_pred_playoffs_rf))


# Algorithme KNN Playoffs
knn_model_playoffs = KNeighborsClassifier()
knn_model_playoffs.fit(X_train_playoffs_scaled, y_train_playoffs)
y_pred_playoffs_knn = knn_model_playoffs.predict(X_test_playoffs_scaled)

print("Metrics for Playoffs - KNN:")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test_playoffs, y_pred_playoffs_knn))
print("Recall:", recall_score(y_test_playoffs, y_pred_playoffs_knn))
print("Precision:", precision_score(y_test_playoffs, y_pred_playoffs_knn))
print("F1-score:", f1_score(y_test_playoffs, y_pred_playoffs_knn))
print(classification_report(y_test_playoffs, y_pred_playoffs_knn))



# Modèles pour les playoffs
models = {
    "Régression logistique": LogisticRegression(),
    "Forêt aléatoire": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

player_scores_global = []

for player in top_players:
    player_data = shot_loc_playoffs_encoded[shot_loc_playoffs_encoded['Player_' + player] == 1]
    X_player = player_data.drop(columns=['Shot Made Flag'])
    y_player = player_data['Shot Made Flag']

    scaler = StandardScaler()
    X_player_scaled = scaler.fit_transform(X_player)

    model_scores_list = []
    for model_name, model in models.items():
        model.fit(X_train_playoffs_scaled, y_train_playoffs)
        y_pred_player = model.predict(X_player_scaled)
        accuracy = accuracy_score(y_player, y_pred_player)
        model_scores_list.append({'Player': player, 'Model': model_name, 'Accuracy': accuracy})

    player_scores_global.extend(model_scores_list)

# Création du DataFrame contenant les scores
scores_df_global = pd.DataFrame(player_scores_global)

plt.figure(figsize=(12, 8))
sns.barplot(data=scores_df_global, x='Player', y='Accuracy', hue='Model')
plt.xlabel('Joueurs')
plt.ylabel('Score de précision')
plt.title('Scores par joueur et par modèle - Tous les joueurs')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Modèle')
plt.tight_layout()
plt.show()

specific_players = ['Allen Iverson', 'Kobe Bryant']

player_scores = []

for player_name in specific_players:
    player_data = shot_loc_playoffs_encoded[shot_loc_playoffs_encoded['Player_' + player_name] == 1]
    X_player = player_data.drop(columns=['Shot Made Flag'])
    y_player = player_data['Shot Made Flag']

    scaler = StandardScaler()
    X_player_scaled = scaler.fit_transform(X_player)

    model_scores_list = []
    for model_name, model in models.items():
        model.fit(X_train_playoffs_scaled, y_train_playoffs)  
        y_pred_player = model.predict(X_player_scaled)
        accuracy = accuracy_score(y_player, y_pred_player)
        model_scores_list.append({'Player': player_name, 'Model': model_name, 'Accuracy': accuracy})

    player_scores.extend(model_scores_list)

scores_df_specific = pd.DataFrame(player_scores)

plt.figure(figsize=(12, 8))
sns.barplot(data=scores_df_specific, x='Player', y='Accuracy', hue='Model')
plt.xlabel('Joueurs')
plt.ylabel('Score de précision')
plt.title('Scores par joueur et par modèle - Joueurs spécifiques')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Modèle')
plt.tight_layout()
plt.show()



