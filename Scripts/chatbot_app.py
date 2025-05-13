import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from difflib import get_close_matches
from huggingface_hub import InferenceClient
import streamlit as st
import json
import re
import io
import requests

# Streamlit setup
st.set_page_config(page_title="⚽ Weather-Aware Football Chatbot", layout="centered")

# Load dataset
url = 'https://github.com/shaunnoeljose/Football-Chatbot/releases/download/data/final_football.csv'
df = pd.read_csv(io.BytesIO(requests.get(url).content))

# Load model
xgb_model = XGBClassifier()
xgb_model.load_model("Model/xgb_model.json")
expected_features = xgb_model.get_booster().feature_names
numeric_cols = df[expected_features].select_dtypes(include=['int', 'float']).columns.tolist()
scaler = StandardScaler().fit(df[numeric_cols])

# Encoders
result_map = {'H': 0, 'A': 1, 'D': 2}
if 'FTR' in df.columns:
    df['FTR_label'] = df['FTR'].map(result_map)

def create_mapping(series):
    return {v: i for i, v in enumerate(series.dropna().unique())}

player_mapping = create_mapping(df['Player'])
stadium_mapping = create_mapping(df['Stadium'])
pos_mapping = create_mapping(df['Pos'])
squad_mapping = create_mapping(df['Squad'])

def standardize_team_name(name):
    all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    match = get_close_matches(name, all_teams, n=1, cutoff=0.6)
    return match[0] if match else name

# LLM setup (FLAN-T5-Base)
hf_client = InferenceClient(model="google/flan-t5-base", token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

def call_llm(prompt: str) -> str:
    return hf_client.text_to_text(prompt).strip()

def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else json.loads(text)

def build_prompt(query):
    return f"""
You are a football assistant. Extract from the question:
- home_team
- away_team
- season
- scenario
- intent
- team (if intent is best_players)

Return JSON like:
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

def predict_match_prob(home, away, season, scenario_features):
    home, away = map(standardize_team_name, [home, away])
    match = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away) & (df['Season'] == season)]
    if match.empty:
        return "Match data not found."

    row = match.iloc[0]
    features = row[expected_features].to_dict()
    features.update(scenario_features)

    if not features.get("Player"):
        fallback = df[(df['Season'] == season) & (df['HomeTeam'] == home)]['Player'].dropna().tolist()
        features['Player'] = fallback[0] if fallback else df['Player'].dropna().iloc[0]

    features['Player'] = player_mapping.get(features['Player'], 0)
    features['Stadium'] = stadium_mapping.get(features['Stadium'], 0)
    features['Pos'] = pos_mapping.get(features['Pos'], 0)
    features['Squad'] = squad_mapping.get(features['Squad'], 0)

    df_input = pd.DataFrame([features])
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    prob = xgb_model.predict_proba(df_input[expected_features])[:, 1][0]
    return f"Home Win Probability: {prob:.2%}"

def best_players_for_team(team, season, scenario_features):
    candidates = df[(df['Squad'] == team) & (df['Season'] == season)]
    scores = []
    for _, row in candidates.iterrows():
        features = row[expected_features].to_dict()
        features.update(scenario_features)
        features['Player'] = player_mapping.get(row['Player'], 0)
        features['Stadium'] = stadium_mapping.get(features['Stadium'], 0)
        features['Pos'] = pos_mapping.get(features['Pos'], 0)
        features['Squad'] = squad_mapping.get(features['Squad'], 0)
        df_player = pd.DataFrame([features])
        df_player[numeric_cols] = scaler.transform(df_player[numeric_cols])
        score = xgb_model.predict_proba(df_player[expected_features])[:, 1][0]
        scores.append((row['Player'], score))
    return [p[0] for p in sorted(scores, key=lambda x: x[1], reverse=True)[:5]]

def compare_team_performance(season, scenario_features):
    filtered = df[df['Season'] == season].copy()
    for k, v in scenario_features.items():
        if k in filtered.columns:
            filtered = filtered[filtered[k] == v]
    if filtered.empty:
        return "No matches found."
    win_counts, total_counts = {}, {}
    for _, row in filtered.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        total_counts[home] = total_counts.get(home, 0) + 1
        total_counts[away] = total_counts.get(away, 0) + 1
        result = row['FTR_label']
        if result == 0:
            win_counts[home] = win_counts.get(home, 0) + 1
        elif result == 1:
            win_counts[away] = win_counts.get(away, 0) + 1
    win_rates = {team: win_counts.get(team, 0)/total for team, total in total_counts.items()}
    return "\n".join([f"- {team}: {rate:.2%}" for team, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True)[:5]])

# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask a football/weather question...")
if query:
    st.chat_message("user").write(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    prompt_text = build_prompt(query)
    llm_response = call_llm(prompt_text)
    st.code(llm_response, language="json")

    try:
        parsed = extract_json(llm_response)
    except:
        st.chat_message("assistant").write("Could not parse the response. Try again.")
        st.stop()

    scenario_dict = {
        "clear": {"rainy": 0, "high_wind": 0, "precipitation_sum": 0.0, "weather_severity": 1},
        "rainy": {"rainy": 1, "high_wind": 0, "precipitation_sum": 10.0, "weather_severity": 4},
        "windy": {"rainy": 0, "high_wind": 1, "precipitation_sum": 2.0, "weather_severity": 3}
    }.get(parsed.get("scenario", "").lower(), {})

    intent = parsed.get("intent")
    if intent == "predict_win":
        result = predict_match_prob(parsed["home_team"], parsed["away_team"], parsed["season"], scenario_dict)
    elif intent == "best_players":
        result = "\n".join(best_players_for_team(parsed["team"], parsed["season"], scenario_dict))
    elif intent == "team_performance_comparison":
        result = compare_team_performance(parsed["season"], scenario_dict)
    else:
        result = "Intent not recognized."

    st.chat_message("assistant").write(result)
    st.session_state.chat_history.append({"role": "assistant", "content": result})
