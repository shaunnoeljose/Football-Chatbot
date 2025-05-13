import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import requests
import io

# loading the data
url = 'https://github.com/shaunnoeljose/Football-Chatbot/releases/download/data/final_football.csv'
response = requests.get(url)
csv_data = io.BytesIO(response.content)
df = pd.read_csv(csv_data)

xgb_model = XGBClassifier()
xgb_model.load_model("Model/xgb_model_player.json")
explainer = shap.TreeExplainer(xgb_model)

# Scaling numerical features
expected_features = xgb_model.get_booster().feature_names
numeric_cols = df[expected_features].select_dtypes(include=['int32','int64', 'float64']).columns.tolist()
scaler = StandardScaler()
scaler.fit(df[numeric_cols])

# creating mapper function - The mapping converts categorical values to integers 
def create_mapping(lst):
    return {k: i for i, k in enumerate(lst.dropna().unique())}

player_order = df['Player'].dropna().unique().tolist()
player_mapping = {k: i for i, k in enumerate(player_order)}

stadium_mapping = create_mapping(df['Stadium'])
pos_mapping = create_mapping(df['Pos'])
squad_mapping = create_mapping(df['Squad'])

# prediction logic
# 
def predict_match(input_dict):
    # converting player, stadium, pos and Squad to numeric values
    df_input = pd.DataFrame([input_dict])
    player_name = df_input['Player'].values[0]
    df_input['Player'] = player_mapping.get(player_name, 0)
    stadium_name = df_input['Stadium'].values[0]
    df_input['Stadium'] = stadium_mapping.get(stadium_name, 0)
    position = df_input['Pos'].values[0]
    df_input['Pos'] = pos_mapping.get(position, 0)
    squad_name = df_input['Squad'].values[0]
    df_input['Squad'] = squad_mapping.get(squad_name, 0)
    
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    df_input = df_input[expected_features]
    pred = xgb_model.predict_proba(df_input)[:, 1]
    shap_vals = explainer.shap_values(df_input)
    return pred[0], df_input, shap_vals

# Streamlit interface
st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("⚽ Player-Aware Football Win Predictor")

match_selector = st.selectbox("Select Match", df[['HomeTeam', 'AwayTeam', 'Season']].drop_duplicates()
    .apply(lambda row: f"{row['HomeTeam']} vs {row['AwayTeam']} - {row['Season']}", axis=1).tolist())
home_match = match_selector.split(" vs ")[0].strip()
away_match, match_season = match_selector.split(" vs ")[1].rsplit(" - ", 1)
away_match = away_match.strip()
match_season = int(match_season.strip())

match_row = df[(df['HomeTeam'] == home_match) & (df['AwayTeam'] == away_match) & (df['Season'] == match_season)].iloc[0]

template = st.selectbox("Pick a scenario", ["None", "Clear day", "Rainy winter", "Windy derby"])
template_values = {
    "Clear day": {"rainy": 0, "high_wind": 0, "weather_code": 0, "precipitation_sum": 0.0, "weather_severity": 1},
    "Rainy winter": {"rainy": 1, "high_wind": 0, "weather_code": 5, "precipitation_sum": 10.0, "weather_severity": 4},
    "Windy derby": {"rainy": 0, "high_wind": 1, "weather_code": 2, "precipitation_sum": 2.0, "weather_severity": 3}
}

user_input = match_row[expected_features].to_dict()
if template != "None":
    user_input.update(template_values[template])

st.subheader("Best-Suited Players for This Match Scenario")
valid_teams = [home_match, away_match]
selected_team = st.selectbox("Filter by Team", valid_teams)
filtered_df = df[(df['Squad'] == selected_team) & (df['Season'] == match_season)]
player_pool = filtered_df['Player'].dropna().unique().tolist()

if st.button("Find Best Players for This Scenario"):
    with st.spinner("Evaluating player suitability..."):
        scenario_features = template_values.get(template, {})
        player_scores = []
        progress_bar = st.progress(0)
        total = len(player_pool)

        for idx, player in enumerate(player_pool):
            player_row = filtered_df[filtered_df['Player'] == player].iloc[0]
            player_features = player_row[expected_features].copy().to_dict()
            player_features.update(scenario_features)
            
            player_features['Player'] = int(player_mapping.get(player, 0))
            player_features['Stadium'] = int(stadium_mapping.get(player_features['Stadium'], 0))
            player_features['Pos'] = int(pos_mapping.get(player_features['Pos'], 0))
            player_features['Squad'] = int(squad_mapping.get(player_features['Squad'], 0))
            
            player_df = pd.DataFrame([player_features])
            player_df[numeric_cols] = scaler.transform(player_df[numeric_cols])
            player_df = player_df[expected_features]
            prob = xgb_model.predict_proba(player_df)[:, 1][0]
            player_scores.append((player, prob))

            progress_bar.progress((idx + 1) / total)

        top_players = sorted(player_scores, key=lambda x: x[1], reverse=True)
        st.markdown("###🏅 Top 5 Players for This Scenario")
        for i, (player, _) in enumerate(top_players[:5], 1):
            st.markdown(f"{i}. **{player}**")

# checking derby
home_team = match_row['HomeTeam']
away_team = match_row['AwayTeam']
derby_pairs = [("Real Madrid", "Barcelona"), ("Real Madrid", "Ath Madrid"), ("Barcelona", "Ath Madrid")]

def check_derby(home_team, away_team):
    if (home_team, away_team) in derby_pairs or (away_team, home_team) in derby_pairs:
        return True
    return False
    
is_derby = check_derby(home_team, away_team)   

# predicting
if st.button("Predict"):

    default_player = player_pool[0] if player_pool else "Unknown"
    user_input['Player'] = default_player
    user_input['Squad'] = selected_team
    user_input['Stadium'] = match_row['Stadium']

    player_pos_data = df[df['Player'] == default_player]['Pos'].dropna()
    if not player_pos_data.empty:
        user_input['Pos'] = player_pos_data.values[0]
    else:
        user_input['Pos'] = "Unknown"

    result, input_df, shap_values = predict_match(user_input)
    st.info(f"Home Team: **{home_team}**")
    if is_derby:
        st.markdown("""
        <div style="background-color:#ffcccb;padding:15px;border-radius:10px;text-align:center">
        <strong>Derby Day!</strong> This is a high-stakes rivalry match. Expect intensity and unpredictability!
        </div>
        """, unsafe_allow_html=True)
    st.success(f"Predicted Home Win Probability: **{result:.2%}**")
    st.subheader("Feature Impact (SHAP)")
    plt.gcf().set_size_inches(8, 4)
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()
