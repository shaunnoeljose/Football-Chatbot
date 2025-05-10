import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain         
from difflib import get_close_matches
from PIL import Image
import json
import re
import requests
import io

# Loading the dataset
url = 'https://github.com/shaunnoeljose/Football-Chatbot/releases/download/data/final_football.csv'
response = requests.get(url)
csv_data = io.BytesIO(response.content)
df = pd.read_csv(csv_data)

xgb_model = XGBClassifier()
xgb_model.load_model("Model/xgb_model.json")

# Scaling numerical features
expected_features = xgb_model.get_booster().feature_names
numeric_cols = df[expected_features].select_dtypes(include=['int32','int64', 'float64']).columns.tolist()
scaler = StandardScaler()
scaler.fit(df[numeric_cols])

# creating mapper function
result_map = {'H': 0, 'A': 1, 'D': 2}
if 'FTR' in df.columns:
    df['FTR_label'] = df['FTR'].map(result_map)

def create_mapping(lst):
    return {k: i for i, k in enumerate(lst.dropna().unique())}

player_order = df['Player'].dropna().unique().tolist()
player_mapping = {k: i for i, k in enumerate(player_order)}
    
stadium_mapping = create_mapping(df['Stadium'])
pos_mapping = create_mapping(df['Pos'])
squad_mapping = create_mapping(df['Squad'])

# Using fuzzy logic for matching different team names
def standardize_team_name(name):
    all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    match = get_close_matches(name, all_teams, n=1, cutoff=0.6)
    return match[0] if match else name
    
# prediction logic
def predict_match_prob(home, away, season, scenario_features):
    home = standardize_team_name(home)
    away = standardize_team_name(away)
    match = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away) & (df['Season'] == season)]
    if match.empty:
        return "Match data not found. Please the team or the season entered"

    input_features  = match.iloc[0][expected_features].to_dict()
    input_features .update(scenario_features)

    # default to first player if no player is selected
    if not input_features.get('Player'):
        fallback_players = df[(df['Season'] == season) & (df['HomeTeam'] == home)]['Player'].dropna().unique().tolist()
        if fallback_players:
            input_features['Player'] = fallback_players[0]
        else:
            input_features['Player'] = df['Player'].dropna().iloc[0]

    input_features['Player'] = player_mapping.get(input_features['Player'], 0)
    input_features['Stadium'] = stadium_mapping.get(input_features['Stadium'], 0)
    input_features['Pos'] = pos_mapping.get(input_features['Pos'], 0)
    input_features['Squad'] = squad_mapping.get(input_features['Squad'], 0)

    df_input = pd.DataFrame([input_features])
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    df_input = df_input[expected_features]

    win_prob = xgb_model.predict_proba(df_input)[:, 1][0]

    return f"Home Win Probability: {win_prob:.2%}"

def best_players_for_team(team, season, scenario_features):
    candidates = df[(df['Squad'] == team) & (df['Season'] == season)]
    scores = []
    
    for _, row in candidates.iterrows():
        player_features  = row[expected_features].to_dict()
        player_features .update(scenario_features)
        
        player_features ['Player'] = player_mapping.get(row['Player'], 0)
        player_features ['Stadium'] = stadium_mapping.get(player_features['Stadium'], 0)
        player_features ['Pos'] = pos_mapping.get(player_features['Pos'], 0)
        player_features ['Squad'] = squad_mapping.get(player_features['Squad'], 0)

        player_df  = pd.DataFrame([player_features])
        player_df [numeric_cols] = scaler.transform(player_df [numeric_cols])
        player_df  = player_df[expected_features]

        score = xgb_model.predict_proba(player_df )[:, 1][0]
        scores.append((row['Player'], score))
    
    sorted_scores = sorted(scores, key=lambda score: score[1], reverse=True)
    top_players = sorted_scores[:5]
    return [p[0] for p in top_players]

#LangChain setup

llm = Ollama(model= "mistral:7b-instruct") 
prompt = PromptTemplate.from_template("""
You are a football assistant. Extract this from the user query:
- home_team
- away_team
- season
- scenario (e.g. clear, rainy, windy)
- intent: "predict_win", "best_players", or "team_performance_comparison"
- team (only for best_players)

Return as JSON in this format:
{{
  "intent": "predict_win",
  "home_team": "Barcelona",
  "away_team": "Real Madrid",
  "season": 2022,
  "scenario": "windy",
  "team": "Barcelona"
}}

Question: {question}
""")

chain = LLMChain(prompt=prompt, llm=llm)

def compare_team_performance(season, scenario_features):
    matches = df[df['Season'] == season]

    # Apply scenario filters
    for key, val in scenario_dict.items():
        if key in filtered.columns:
            filtered = filtered[filtered[key] == val]

    if matches.empty:
        return "No matches found for the given scenario."

    win_counts = {}
    total_counts = {}

    for _, row in filtered.iterrows():
        result = row.get('FTR_label')
        home = row['HomeTeam']
        away = row['AwayTeam']

        total_counts[home] = total_counts.get(home, 0) + 1
        total_counts[away] = total_counts.get(away, 0) + 1

        if result == 0:
            win_counts[home] = win_counts.get(home, 0) + 1
        elif result == 1:
            win_counts[away] = win_counts.get(away, 0) + 1

    win_rates = {}
    for team in total_counts:
        wins = win_counts.get(team, 0)
        total = total_counts[team]
        win_rates[team] = wins / total

    sorted_teams = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

    top_teams_output = "📊 Top teams in {season} under specified conditions:\n".format(season=season)
    for team, win_rate in sorted_teams[:5]:
        top_teams_output += f"- {team}: {win_rate:.2%} win rate\n"

    return top_teams_output

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)

# Streamlit interface
st.set_page_config(page_title="⚽ Weather-Aware Football Chatbot", layout="centered")

st.markdown("""
<style>
body, .stApp {
    background-color: #f8f9fa;
    color: #212529;
}
.block-container {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
h1, h2, h3, label, p, div, span {
    color: #212529 !important;
}
button {
    background-color: #1f77b4 !important;
    color: white !important;
}
.sticky-launch-button button {
    background-color: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)
    
# Adding launch button
st.markdown("""
<style>
.sticky-launch-button {
    position: fixed;
    bottom: 60px;
    right: 20px;
    z-index: 9999;
}
.sticky-launch-button a {
    text-decoration: none;
}
.sticky-launch-button button {
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
</style>

<div class="sticky-launch-button">
    <a href="http://localhost:8502/advanced_predictor_app" target="_blank">
        <button>🚀 Launch Advanced Predictor App</button>
    </a>
</div>
""", unsafe_allow_html=True)

logo_path = "C:/Users/shaun/Downloads/Football_Logo.png"
logo_image = Image.open(logo_path)

# Setting the streamlit layout
col1, col2 = st.columns([1, 4])

with col1:
    st.markdown("<div style='margin-top: 55px;'>", unsafe_allow_html=True)
    st.image(logo_image, width=250)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("## Weather-Aware Football Chatbot")
    st.markdown("""
    Welcome to the **Weather-Aware Football Insights Chatbot**, your personal assistant for analyzing LaLiga football matches based on weather conditions.
    This tool leverages weather features(clear, windy & rainy) to give data-backed insights on team and player performance. Sample queries could be:
    * Who plays best for Barcelona when it rains in 2019?
    * Winning chance for Real Madrid vs Atlético Madrid in 2020 on a windy day?
    * Which teams performed best in rainy weather in 2021?
    """)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

user_query = st.chat_input("Ask something like 'What’s the win chance for Barcelona vs Real Madrid in 2022 on a rainy day?'")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    response_json = chain.run(question=user_query)
    st.code(response_json, language="json")  # Show parsed JSON for debugging

    parsed = extract_json(response_json)
    if not parsed:
        response = "Could not parse response from the model. Please try rephrasing your question."
    scenario_dict = {}
    scenario = parsed.get("scenario", "").lower()
    if scenario == "clear":
        scenario_dict = {"rainy": 0, "high_wind": 0, "precipitation_sum": 0.0, "weather_severity": 1}
    elif scenario == "rainy":
        scenario_dict = {"rainy": 1, "high_wind": 0, "precipitation_sum": 10.0, "weather_severity": 4}
    elif scenario == "windy":
        scenario_dict = {"rainy": 0, "high_wind": 1, "precipitation_sum": 2.0, "weather_severity": 3}

    if parsed.get("intent") == "predict_win":
        response = predict_match_prob(
            parsed.get("home_team"),
            parsed.get("away_team"),
            parsed.get("season"),
            scenario_dict)
   
    elif parsed.get("intent") == "best_players":
        team = parsed.get("team")
        season = parsed.get("season")
        if not team or not season:
            response = "Missing 'team' or 'season' for best player query."
        else:
            players = best_players_for_team(team, season, scenario_dict)
            response = f"🏅 Top players for {scenario} conditions: {', '.join(players)}"
    
    elif parsed.get("intent") == "team_performance_comparison":
        season = parsed.get("season")
        scenario = parsed.get("scenario", "").lower()
        scenario_dict = {}
        if scenario == "rainy":
            scenario_dict = {"rainy": 1}
        elif scenario == "windy":
            scenario_dict = {"high_wind": 1}
        elif scenario == "clear":
            scenario_dict = {"rainy": 0, "high_wind": 0}
        response = compare_team_performance(season, scenario_dict)
        
    else:
        response = "I couldn't understand the intent. Try rephrasing."

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
