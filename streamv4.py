import streamlit as st
import pandas as pd
import requests
#import json
import plotly.graph_objects as go
import pickle
import shap
import matplotlib.pyplot as plt
#from xgboost import XGBClassifier
import seaborn as sns





st.set_page_config(page_title="Application Crédit")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Définir l'URL de l'API
url = "https://ludo-test-may-123.azurewebsites.net/"
#url = "http://127.0.0.1:5000/"

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Chemins vers les fichiers CSV
base = 'p8_base_ok_v2.csv'  

# Charger les données d'entraînement
df_base = load_data(base)
df_base = df_base.drop(columns=['SK_ID_CURR'])

def main():
    # Afficher un titre dans l'interface utilisateur
    
    st.title("Application Crédit")

    # Affichage du formulaire de téléchargement du fichier CSV
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        # Lire le fichier CSV dans un DataFrame Pandas
        df = load_data(uploaded_file)
        df_ok = df.drop(columns=['SK_ID_CURR'])

        # On charge les ID dans une liste
        id_values = df['SK_ID_CURR'].unique().tolist()

        # Creation liste déroulante
        selected_id = st.selectbox("Sélectionnez un client :", id_values)

        # Affichage des données associées à l'ID sélectionné
        st.write(f"Vous avez sélectionné l'ID client : {selected_id}")
        selected_data = df[df['SK_ID_CURR'] == selected_id] # On enregistre les data de la ligne selec
        
        # On supprime la colonne ID pour envoyer à notre modèle
        selected_data2 = selected_data.drop(columns=['SK_ID_CURR'])
        selected_df = selected_data.drop(columns=['SK_ID_CURR'])

        # Sélectionner les colonnes à afficher
        list_col = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'APPROVED_CNT_PAYMENT_MEAN', 'DAYS_EMPLOYED',
                    'PAYMENT_RATE', 'PREV_CNT_PAYMENT_MEAN', 'INSTAL_DPD_MEAN', 'DAYS_BIRTH', 'POS_SK_DPD_DEF_MEAN']  
        echantillon = selected_data2[list_col]

        # Afficher l'échantillon sélectionné avec les colonnes spécifiques
        st.write("Échantillon sélectionné (colonnes spécifiques) :")
        st.write(echantillon)

        # Créer un bouton dans l'interface utilisateur pour effectuer la requête à l'API
        if st.button("Predire") or "pred_result" not in st.session_state:
            # Convertir les données sélectionnées en JSON
            selected_data2 = selected_data2.values.tolist()

            # Effectuer la requête à l'API
            response = requests.post(url, json=selected_data2)
            # Vérifier si la requête a réussi (code de statut 200)
            if response.status_code == 200:
                # Convertir la réponse JSON en un dictionnaire Python
                pred = response.json()
                # Stocker les résultats dans st.session_state
                st.session_state.pred_result = pred
                st.session_state.selected_df = selected_df
                st.session_state.df_ok = df_ok
                st.session_state.selected_data2 = selected_data2

                # Mise à jour de la cible (TARGET) dans les données sélectionnées
                prediction_class = pred['prediction_class'][0]
                if prediction_class == 0:
                    st.session_state.selected_df['TARGET'] = 0
                else:
                    st.session_state.selected_df['TARGET'] = 1
            else:
                st.error(f"La requête a échoué avec le code de statut : {response.status_code}")

        # Vérifier si les résultats de la prédiction sont disponibles dans st.session_state
        if "pred_result" in st.session_state:
            pred = st.session_state.pred_result
            selected_df = st.session_state.selected_df
            df_ok = st.session_state.df_ok
            selected_data2 = st.session_state.selected_data2

            # Afficher la mention "Crédit accordé" ou "Crédit refusé"
            prediction_class = pred['prediction_class'][0]
            if prediction_class == 0:
                st.success("Crédit accordé")
            else:
                st.error("Crédit refusé")

            # Afficher la probabilité de prédiction et le seuil
            prediction_proba = pred['prediction_proba'][0]
            st.write("Le seuil d'acceptation est 0.14")

            # JAUGE
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba * 100,
                title={'text': "Scoring client"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 14], 'color': "#009E73"},
                        {'range': [14, 100], 'color': "#D55E00"}
                    ],
                }
            ))
            st.plotly_chart(fig)

            # FEATURE IMPORTANCE avec SHAP
            model = pickle.load(open('boostv4_df.pkl', 'rb'))
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            shap_values = explainer.shap_values(df_ok)

            st.header("Importances globales")
            fig_shap_summary = plt.figure()
            shap.summary_plot(shap_values, df_ok, show=False)
            st.pyplot(fig_shap_summary)

            # Importance locale
            st.header("Importance locale des caractéristiques (SHAP)")
            shap_values_single = explainer.shap_values(pd.DataFrame(selected_data2, columns=df_ok.columns))
            fig_shap_force = shap.force_plot(explainer.expected_value, shap_values_single[0], pd.DataFrame(selected_data2, columns=df_ok.columns), matplotlib=True)
            st.pyplot(fig_shap_force)

            # Box des 5 variables spécifiées
            st.header("Boxplot des 5 variables spécifiées")
            selected_column = st.selectbox("Sélectionnez une variable à afficher :", options=list_col)

            st.header(f"Boxplot de {selected_column} par TARGET")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_base, x='TARGET', y=selected_column, palette=['#009E73', '#D55E00'], ax=ax)
            # Ajout du point représentant le client sélectionné
            sns.swarmplot(data=selected_df, y=selected_column, color='#FFD700', size=10, ax=ax)
            ax.set_title(f'Boxplot de {selected_column} par TARGET')
            st.pyplot(fig)

            # Scatter plot des variables sélectionnées
            st.header("Scatter plot des variables sélectionnées")
            x_column = st.selectbox("Sélectionnez la variable pour l'axe X :", options=list_col, key="x_axis")
            y_column = st.selectbox("Sélectionnez la variable pour l'axe Y :", options=list_col, key="y_axis")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_base, x=x_column, y=y_column, hue='TARGET', palette=['#009E73', '#D55E00'], ax=ax)
            # Ajout du point représentant le client sélectionné
            ax.scatter(x=selected_df[x_column], y=selected_df[y_column], color='#FFD700', s=100, label='Client sélectionné')
            ax.set_title(f'Scatter plot de {x_column} vs {y_column}')
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("Veuillez prédire les résultats pour afficher les graphes.")
        
if __name__ == "__main__":
    main()
