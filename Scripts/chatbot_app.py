import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from difflib import get_close_matches
from PIL import Image
import json
import re
import requests
import io
from huggingface_hub import InferenceClient
import streamlit as st

# Set Streamlit page
st.set_page_config(page_title="⚽ Weather-Aware Football Chatbot", layout="centered")

# Optional advanced app launch
query_params = st.query_params
if "advanced_predictor_app" in query_params:
    exec(open("Scripts/advanced_predictor_app.py").read())
    st.stop()

# Load football dataset
url = 'https://github.com/shaunnoeljose/Football-Chatbot/releases/download/data/final_football.csv'
df = pd.read_csv(io.BytesIO(requests.get(url).content))

# Load model and prepare preprocessing
xgb_model = XGBClassifier()
xgb_model.load_model("Model/xgb_model.json")
expected_features = xgb_model.get_booster().feature_names
numeric_cols = df[expected_features].select_dtypes(include=['int', 'float']).columns.tolist()
scaler = StandardScaler().fit(df[numeric_cols])

# Result mapping
result_map = {'H': 0, 'A': 1, 'D': 2}
if 'FTR' in df.columns:
    df['FTR_label'] = df['FTR'].map(result_map)

# Encode mapping
def create_mapping(lst): return {k: i for i, k in enumerate(lst.dropna().unique())}
player_mapping = create_mapping(df['Player'])
stadium_mapping = create_mapping(df['Stadium'])
pos_mapping = create_mapping(df['Pos'])
squad_mapping = create_mapping(df['Squad'])

def standardize_team_name(name):
    all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    match = get_close_matches(name, all_teams, n=1, cutoff=0.6)
    return match[0] if match else name

def predict_match_prob(home, away, season, scenario_features):
    home, away = map(standardize_team_name, [home, away])
    match = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away) & (df['Season'] == season)]
    if match.empty:
        return "Match data not found. Check the team names or season."

    input_features = match.iloc[0][expected_features].to_dict()
    input_features.update(scenario_features)
    if not input_features.get('Player'):
        fallback = df[(df['Season'] == season) & (df['HomeTeam'] == home)]['Player'].dropna().tolist()
        input_features['Player'] = fallback[0] if fallback else df['Player'].dropna().iloc[0]

    input_features['Player'] = player_mapping.get(input_features['Player'], 0)
    input_features['Stadium'] = stadium_mapping.get(input_features['Stadium'], 0)
    input_features['Pos'] = pos_mapping.get(input_features['Pos'], 0)
    input_features['Squad'] = squad_mapping.get(input_features['Squad'], 0)

    df_input = pd.DataFrame([input_features])
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    win_prob = xgb_model.predict_proba(df_input[expected_features])[:, 1][0]
    return f"Home Win Probability: {win_prob:.2%}"

def best_players_for_team(team, season, scenario_features):
    candidates = df[(df['Squad'] == team) & (df['Season'] == season)]
    scores = []
    for _, row in candidates.iterrows():
        player_features = row[expected_features].to_dict()
        player_features.update(scenario_features)
        player_features['Player'] = player_mapping.get(row['Player'], 0)
        player_features['Stadium'] = stadium_mapping.get(player_features['Stadium'], 0)
        player_features['Pos'] = pos_mapping.get(player_features['Pos'], 0)
        player_features['Squad'] = squad_mapping.get(player_features['Squad'], 0)
        df_player = pd.DataFrame([player_features])
        df_player[numeric_cols] = scaler.transform(df_player[numeric_cols])
        score = xgb_model.predict_proba(df_player[expected_features])[:, 1][0]
        scores.append((row['Player'], score))
    return [p[0] for p in sorted(scores, key=lambda x: x[1], reverse=True)[:5]]

hf_client = InferenceClient(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

def call_llm(prompt: str) -> str:
    response = hf_client.text_generation(
        prompt,
        model="google/flan-t5-base",  # ✅ Provide model here, not in constructor
        max_new_tokens=512,
        do_sample=False
    )
    return response.strip()

def extract_json(text): 
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else json.loads(text)

def compare_team_performance(season, scenario_features):
    matches = df[df['Season'] == season]
    filtered = matches.copy()
    for key, val in scenario_features.items():
        if key in filtered.columns:
            filtered = filtered[filtered[key] == val]
    if filtered.empty: return "No matches found for the given scenario."

    win_counts, total_counts = {}, {}
    for _, row in filtered.iterrows():
        result = row.get('FTR_label')
        home, away = row['HomeTeam'], row['AwayTeam']
        total_counts[home] = total_counts.get(home, 0) + 1
        total_counts[away] = total_counts.get(away, 0) + 1
        if result == 0: win_counts[home] = win_counts.get(home, 0) + 1
        elif result == 1: win_counts[away] = win_counts.get(away, 0) + 1

    win_rates = {team: win_counts.get(team, 0)/total for team, total in total_counts.items()}
    sorted_teams = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
    return "\n".join([f"- {team}: {rate:.2%} win rate" for team, rate in sorted_teams[:5]])

# Prompt Template
def build_prompt(query: str) -> str:
    return f"""
You are a football assistant. Extract from the question:
- home_team
- away_team
- season
- scenario
- intent
- team (if best_players intent)

Return in JSON:
{{
  "intent": "predict_win",
  "home_team": "Barcelona",
  "away_team": "Real Madrid",
  "season": 2022,
  "scenario": "windy",
  "team": "Barcelona"
}}

Question: {query}
"""

# UI setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask: 'Win chance for Barcelona vs Real Madrid in 2022 on a rainy day?'")
if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    prompt_text = build_prompt(user_query)
    response_text = call_llm(prompt_text)
    st.code(response_text, language="json")

    try:
        parsed = extract_json(response_text)
    except:
        st.chat_message("assistant").write("❌ Could not parse model response. Try again.")
        st.stop()

    scenario = parsed.get("scenario", "").lower()
    scenario_dict = {
        "clear": {"rainy": 0, "high_wind": 0, "precipitation_sum": 0.0, "weather_severity": 1},
        "rainy": {"rainy": 1, "high_wind": 0, "precipitation_sum": 10.0, "weather_severity": 4},
        "windy": {"rainy": 0, "high_wind": 1, "precipitation_sum": 2.0, "weather_severity": 3}
    }.get(scenario, {})

    intent = parsed.get("intent")
    if intent == "predict_win":
        response = predict_match_prob(parsed.get("home_team"), parsed.get("away_team"), parsed.get("season"), scenario_dict)
    elif intent == "best_players":
        team, season = parsed.get("team"), parsed.get("season")
        response = f"🏅 Top players for {scenario} conditions: {', '.join(best_players_for_team(team, season, scenario_dict))}" if team and season else "Missing 'team' or 'season'."
    elif intent == "team_performance_comparison":
        response = compare_team_performance(parsed.get("season"), scenario_dict)
    else:
        response = "I couldn't understand the intent. Try rephrasing."

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
