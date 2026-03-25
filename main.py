import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# =========================================================
# 1. CARICAMENTO DATASET
# =========================================================
df = pd.read_csv("20-21_plEventsData.csv", low_memory=False)

# =========================================================
# 2. RIMOZIONE COLONNA INDICE TECNICA
# =========================================================
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

## =========================================================
# 3. MAPPATURA teamId -> team_name
# =========================================================

team_map = {
    13: "Arsenal",
    14: "Leicester City",
    15: "Chelsea",
    18: "Southampton",
    19: "Leeds United",
    23: "Newcastle United",
    24: "Aston Villa",
    26: "Liverpool",
    29: "West Ham United",
    30: "Tottenham Hotspur",
    31: "Everton",
    32: "Manchester United",
    161: "Wolverhampton Wanderers",
    162: "Crystal Palace",
    163: "Sheffield United",
    167: "Manchester City",
    170: "Fulham",
    175: "West Bromwich Albion",
    184: "Burnley",
    211: "Brighton & Hove Albion"
}

# Associazione dell'identificativo numerico al nome della squadra
df["team_name"] = df["teamId"].map(team_map)

# Dopo la mappatura, teamId non è più necessario
df = df.drop(columns=["teamId"])

print("\nNUMERO SQUADRE:")
print(df["team_name"].nunique())

print("\nELENCO SQUADRE:")
for squadra in sorted(df["team_name"].dropna().unique()):
    print("-", squadra)

# =========================================================
# 4. INTERPRETAZIONE E DESCRIZIONE DATASET
# =========================================================
print("\n================ INTERPRETAZIONE E DESCRIZIONE ================\n")
print("Il dataset racchiude gli eventi della Premier League 2020-2021.")
print("Ogni riga rappresenta un singolo evento di gioco associato a una squadra e a un giocatore,")
print("annotato con informazioni temporali e spaziali sul campo.")
print("Poiché il dataset è strutturato a livello di evento, è necessario ricostruire le partite")
print("e aggregare gli eventi per ottenere feature utili all'analisi esplorativa e alla classificazione.\n")

print("DIMENSIONE DATASET:")
print(df.shape)

print("\nNUMERO COLONNE:")
print(len(df.columns))

print("\nNUMERO SQUADRE:")
print(df["team_name"].nunique())

print("\nELENCO SQUADRE:")
for squadra in sorted(df["team_name"].dropna().unique()):
    print("-", squadra)

# =========================================================
# 5. VERIFICA NUMERO PARTITE
# =========================================================
start_events = df[df["type_displayName"] == "Start"].copy()
first_half_start = start_events[start_events["period_displayName"] == "FirstHalf"].copy()

numero_partite = int(len(first_half_start) / 2)

print("\nNUMERO EVENTI START TOTALI:")
print(len(start_events))

print("\nSTART PER PERIODO:")
print(start_events["period_displayName"].value_counts())

print("\nEVENTI START FIRST HALF:")
print(len(first_half_start))

print("\nNUMERO PARTITE TOTALI:")
print(int(numero_partite))

# =========================================================
# 6. RICOSTRUZIONE TABELLA PARTITE
# =========================================================
start_first_half = df[
    (df["type_displayName"] == "Start") &
    (df["period_displayName"] == "FirstHalf")
].copy().reset_index(drop=True)

matches = []

for i in range(0, len(start_first_half), 2):
    team1 = start_first_half.loc[i, "team_name"]
    team2 = start_first_half.loc[i + 1, "team_name"]

    matches.append({
        "match_id": i // 2 + 1,
        "team1": team1,
        "team2": team2
    })

matches_df = pd.DataFrame(matches)

print("\nNUMERO PARTITE RICOSTRUITE:")
print(len(matches_df))

print("\nPRIME PARTITE:")
print(matches_df.head())

print("\nNumero squadre in team1:")
print(matches_df["team1"].nunique())

print("\nNumero squadre in team2:")
print(matches_df["team2"].nunique())

matches_df.to_csv("matches_table.csv", index=False)
print("\nFile salvato: matches_table.csv")

# =========================================================
# 7. ASSEGNAZIONE match_id A OGNI EVENTO
# =========================================================
df = df.reset_index(drop=True)

start_mask = (
    (df["type_displayName"] == "Start") &
    (df["period_displayName"] == "FirstHalf")
)

start_indices = df[start_mask].index.tolist()

match_ids = []
current_match = 0
next_start_pointer = 0

for i in range(len(df)):
    if next_start_pointer < len(start_indices) and i == start_indices[next_start_pointer]:
        current_match += 1
        next_start_pointer += 2

    match_ids.append(current_match)

df["match_id"] = match_ids

print("\nNumero match_id trovati negli eventi:")
print(df["match_id"].nunique())

# =========================================================
# 8. FEATURE EXTRACTION PER PARTITA E SQUADRA
# =========================================================
event_features = [
    "Pass",
    "Goal",
    "MissedShots",
    "SavedShot",
    "ShotOnPost",
    "Foul",
    "Tackle",
    "Interception",
    "Clearance",
    "BallRecovery",
    "Aerial",
    "TakeOn",
    "Card"
]

all_teams_matches = []

for _, row in matches_df.iterrows():
    all_teams_matches.append({"match_id": row["match_id"], "team_name": row["team1"]})
    all_teams_matches.append({"match_id": row["match_id"], "team_name": row["team2"]})

all_teams_matches_df = pd.DataFrame(all_teams_matches)


features_df = (
    df[df["type_displayName"].isin(event_features)]
    .groupby(["match_id", "team_name", "type_displayName"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Garantiamo 2 squadre per match
features_df = all_teams_matches_df.merge(
    features_df,
    on=["match_id", "team_name"],
    how="left"
).fillna(0)
features_df.head(20)

# =========================================================
# 9. FEATURE SPAZIALI
# =========================================================
spatial_features = (
    df.groupby(["match_id", "team_name"])[["x", "y", "endX", "endY"]]
    .mean()
    .reset_index()
    .rename(columns={
        "x": "mean_x",
        "y": "mean_y",
        "endX": "mean_endX",
        "endY": "mean_endY"
    })
)

features_df = features_df.merge(
    spatial_features,
    on=["match_id", "team_name"],
    how="left"
)

# =========================================================
# 10. AGGIUNTA AVVERSARIO
# =========================================================
opponent_rows = []

for _, row in matches_df.iterrows():
    opponent_rows.append({
        "match_id": row["match_id"],
        "team_name": row["team1"],
        "opponent": row["team2"]
    })
    opponent_rows.append({
        "match_id": row["match_id"],
        "team_name": row["team2"],
        "opponent": row["team1"]
    })

opponent_df = pd.DataFrame(opponent_rows)

features_df = features_df.merge(
    opponent_df,
    on=["match_id", "team_name"],
    how="left"
)

# =========================================================
# 11. FEATURE DERIVATE
# =========================================================
features_df["total_shots"] = (
    features_df["Goal"] +
    features_df["MissedShots"] +
    features_df["SavedShot"] +
    features_df["ShotOnPost"]
)

features_df["shots_on_target"] = (
    features_df["Goal"] +
    features_df["SavedShot"] +
    features_df["ShotOnPost"]
)

# =========================================================
# 12. GOAL CORRETTI (GESTIONE AUTORETI)
# =========================================================
goal_events = df[df["type_displayName"] == "Goal"].copy()

goal_events = goal_events.merge(
    opponent_df,
    on=["match_id", "team_name"],
    how="left"
)

goal_events["scoring_team"] = np.where(
    goal_events["isOwnGoal"] == True,
    goal_events["opponent"],
    goal_events["team_name"]
)

goals_corrected = (
    goal_events.groupby(["match_id", "scoring_team"])
    .size()
    .reset_index(name="goals_corrected")
    .rename(columns={"scoring_team": "team_name"})
)

features_df = features_df.drop(
    columns=["goals", "goals_conceded", "points", "result", "result_x", "result_y"],
    errors="ignore"
)

features_df = features_df.merge(
    goals_corrected,
    on=["match_id", "team_name"],
    how="left"
)

features_df["goals"] = features_df["goals_corrected"].fillna(0).astype(int)
features_df = features_df.drop(columns=["goals_corrected"])

# =========================================================
# 13. RICOSTRUZIONE RISULTATO PARTITA
# =========================================================
results = []

for match_id in features_df["match_id"].unique():
    match_rows = features_df[features_df["match_id"] == match_id][["team_name", "goals"]].reset_index(drop=True)

    if len(match_rows) != 2:
        print(f"Errore: match_id {match_id} ha {len(match_rows)} squadre")
        continue

    team_a = match_rows.loc[0, "team_name"]
    goals_a = match_rows.loc[0, "goals"]

    team_b = match_rows.loc[1, "team_name"]
    goals_b = match_rows.loc[1, "goals"]

    if goals_a > goals_b:
        results.append({"match_id": match_id, "team_name": team_a, "result": "Vittoria"})
        results.append({"match_id": match_id, "team_name": team_b, "result": "Sconfitta"})
    elif goals_a < goals_b:
        results.append({"match_id": match_id, "team_name": team_a, "result": "Sconfitta"})
        results.append({"match_id": match_id, "team_name": team_b, "result": "Vittoria"})
    else:
        results.append({"match_id": match_id, "team_name": team_a, "result": "Pareggio"})
        results.append({"match_id": match_id, "team_name": team_b, "result": "Pareggio"})

results_df = pd.DataFrame(results)

features_df = features_df.drop(columns=["result", "result_x", "result_y"], errors="ignore")

features_df = features_df.merge(
    results_df,
    on=["match_id", "team_name"],
    how="left"
)
# =========================================================
# 14. GOAL SUBITI E PUNTI
# =========================================================
features_df["goals_conceded"] = (
    features_df.groupby("match_id")["goals"].transform("sum") - features_df["goals"]
)

features_df["points"] = features_df["result"].map({
    "Vittoria": 3,
    "Pareggio": 1,
    "Sconfitta": 0
})

# =========================================================
# 14bis. AGGIUNTA SAVE (PORTIERI)
# =========================================================
team_saves = (
    df[df["type_displayName"] == "Save"]
    .groupby(["match_id", "team_name"])
    .size()
    .reset_index(name="Saves")
)

features_df = features_df.merge(
    team_saves,
    on=["match_id", "team_name"],
    how="left"
)

features_df["Saves"] = features_df["Saves"].fillna(0).astype(int)

# =========================================================
# 14.3 POSSESSO PALLA (PROXY)
# =========================================================

# Proxy del possesso basata su eventi che riflettono controllo del gioco
features_df["possession_proxy"] = (
    features_df["Pass"] +
    features_df["BallRecovery"] +
    features_df["TakeOn"] +
    features_df["Aerial"]
)

# Differenza di possesso tra le due squadre nella stessa partita
features_df["possession_diff"] = features_df.groupby("match_id")["possession_proxy"].transform(
    lambda x: x - x.iloc[::-1].values
)

# Possesso relativo all'interno della partita
total_possession = features_df.groupby("match_id")["possession_proxy"].transform("sum")
features_df["possession_ratio"] = features_df["possession_proxy"] / (total_possession + 1)

# Variabile binaria di vantaggio nel possesso
features_df["possession_advantage"] = (features_df["possession_diff"] > 0).astype(int)



# =========================================================
# 15. CONTROLLI FINALI DI COERENZA
# =========================================================
print("\n================ CONTROLLI DI COERENZA ================\n")

print("Numero righe features_df:", len(features_df))
print("Numero match_id unici:", features_df["match_id"].nunique())

teams_per_match = features_df.groupby("match_id")["team_name"].nunique()
print("\nDistribuzione numero squadre per partita:")
print(teams_per_match.value_counts())

bad_matches = teams_per_match[teams_per_match != 2]
print("\nPartite con numero squadre diverso da 2:")
print(bad_matches)

matches_per_team = features_df.groupby("team_name")["match_id"].nunique().sort_values()
print("\nNumero partite per squadra:")
print(matches_per_team)

print("\nDistribuzione risultati:")
print(features_df["result"].value_counts())

print("\nPrime 20 righe con team, opponent, goals, goals_conceded, result:")
print(features_df[["match_id", "team_name", "opponent", "goals", "goals_conceded", "result"]].head(20))

print("\nPrime 10 partite complete:")
for m in sorted(features_df["match_id"].unique())[:10]:
    print(f"\nMatch {m}")
    print(features_df[features_df["match_id"] == m][
        ["team_name", "opponent", "goals", "goals_conceded", "result",
         "Pass", "total_shots", "Foul", "Tackle", "Interception"]
    ])

dup_check = features_df.duplicated(subset=["match_id", "team_name"]).sum()
print("\nNumero duplicati su (match_id, team_name):", dup_check)

# =========================================================
# 16. CONTROLLO DIRETTO SUGLI EVENTI ORIGINALI DI UNA PARTITA
# =========================================================
match_to_check = 1

print(f"\n================ EVENTI ORIGINALI MATCH {match_to_check} ================\n")

original_match_events = df[df["match_id"] == match_to_check]

print("Squadre presenti negli eventi originali:")
print(original_match_events["team_name"].value_counts())

print("\nTipi di evento per squadra nella partita:")
print(
    original_match_events.groupby(["team_name", "type_displayName"]).size()
    .sort_values(ascending=False)
    .head(30)
)

print("\nRighe aggregate nel dataset finale:")
print(
    features_df[features_df["match_id"] == match_to_check][
        ["team_name", "opponent", "goals", "goals_conceded", "result",
         "Pass", "Goal", "MissedShots", "SavedShot", "ShotOnPost",
         "Foul", "Tackle", "Interception", "Clearance", "BallRecovery"]
    ]
)

# =========================================================
# 17. SALVATAGGIO FILE FINALI
# =========================================================
features_df.to_csv("match_team_features_final.csv", index=False)
print("\nFile salvato: match_team_features_final.csv")

df.to_csv("pl_events_enriched_v2.csv", index=False)
print("File salvato: pl_events_enriched_v2.csv")

# =========================================================
# 18. CONTROLLO AUTOMATICO COMPLETO DELLE FEATURE EVENTO
# =========================================================
print("\n================ CONTROLLO COMPLETO FEATURE EVENTO ================\n")

check_df = (
    df[df["type_displayName"].isin(event_features)]
    .groupby(["match_id", "team_name", "type_displayName"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

all_pairs = features_df[["match_id", "team_name"]].drop_duplicates()

check_df = all_pairs.merge(check_df, on=["match_id", "team_name"], how="left").fillna(0)

compare_cols = ["match_id", "team_name"] + event_features

check_df = check_df[compare_cols].copy()
features_check = features_df[compare_cols].copy()

for col in event_features:
    check_df[col] = check_df[col].astype(int)
    features_check[col] = features_check[col].astype(int)

comparison = features_check.merge(
    check_df,
    on=["match_id", "team_name"],
    suffixes=("_features", "_original")
)

mismatch_rows = []

for _, row in comparison.iterrows():
    for col in event_features:
        val_features = row[f"{col}_features"]
        val_original = row[f"{col}_original"]

        if val_features != val_original:
            mismatch_rows.append({
                "match_id": row["match_id"],
                "team_name": row["team_name"],
                "feature": col,
                "value_features_df": val_features,
                "value_original_df": val_original
            })

mismatch_df = pd.DataFrame(mismatch_rows)

if mismatch_df.empty:
    print("✅ Tutte le feature evento coincidono perfettamente con i conteggi degli eventi originali.")
else:
    print("⚠️ Sono state trovate discrepanze.")
    print("\nNumero discrepanze trovate:")
    print(len(mismatch_df))

    print("\nPrime discrepanze:")
    print(mismatch_df.head(20))

    mismatch_df.to_csv("feature_mismatch_report.csv", index=False)
    print("\nFile salvato: feature_mismatch_report.csv")

# =========================================================
# 19. CLASSIFICA FINALE DEL CAMPIONATO
# =========================================================
print("\n================ CLASSIFICA FINALE STAGIONE ================\n")

classifica = (
    features_df.groupby("team_name")
    .agg(
        points=("points", "sum"),
        played=("match_id", "nunique"),
        wins=("result", lambda x: (x == "Vittoria").sum()),
        draws=("result", lambda x: (x == "Pareggio").sum()),
        losses=("result", lambda x: (x == "Sconfitta").sum()),
        goals=("goals", "sum"),
        goals_conceded=("goals_conceded", "sum")
    )
    .reset_index()
)

classifica["goal_difference"] = classifica["goals"] - classifica["goals_conceded"]

classifica = classifica.sort_values(
    by=["points", "goal_difference", "goals"],
    ascending=False
).reset_index(drop=True)

print(tabulate(classifica, headers="keys", tablefmt="fancy_grid", showindex=False))

classifica.to_csv("league_table.csv", index=False)
print("\nFile salvato: league_table.csv")

# =========================================================
# 20. RIASSUNTO STAGIONALE PER SQUADRA
# =========================================================
print("\n================ RIASSUNTO STAGIONALE PER SQUADRA ================\n")

season_summary = features_df.groupby("team_name").agg({
    "Pass": "sum",
    "Goal": "sum",
    "MissedShots": "sum",
    "SavedShot": "sum",
    "ShotOnPost": "sum",
    "Foul": "sum",
    "Tackle": "sum",
    "Interception": "sum",
    "Clearance": "sum",
    "BallRecovery": "sum",
    "Aerial": "sum",
    "TakeOn": "sum",
    "Card": "sum",
    "total_shots": "sum",
    "shots_on_target": "sum",
    "goals": "sum",
    "goals_conceded": "sum",
    "mean_x": "mean",
    "mean_y": "mean",
    "mean_endX": "mean",
    "mean_endY": "mean",
    "Saves": "sum"
}).reset_index()

wins = features_df[features_df["result"] == "Vittoria"].groupby("team_name").size()
draws = features_df[features_df["result"] == "Pareggio"].groupby("team_name").size()
losses = features_df[features_df["result"] == "Sconfitta"].groupby("team_name").size()

season_summary["wins"] = season_summary["team_name"].map(wins).fillna(0).astype(int)
season_summary["draws"] = season_summary["team_name"].map(draws).fillna(0).astype(int)
season_summary["losses"] = season_summary["team_name"].map(losses).fillna(0).astype(int)
season_summary["points"] = season_summary["wins"] * 3 + season_summary["draws"]
season_summary["goal_difference"] = season_summary["goals"] - season_summary["goals_conceded"]

season_summary = season_summary.sort_values(
    by=["points", "goal_difference", "goals"],
    ascending=False
).reset_index(drop=True)

print("Prime righe del riassunto stagionale:")
print(season_summary.head(20))

season_summary.to_csv("season_summary_by_team.csv", index=False)
print("\nFile salvato: season_summary_by_team.csv")

# =========================================================
# 21. STATISTICHE SQUADRE
# =========================================================
print("\n================ STATISTICHE SQUADRE ================\n")

top_scoring_team = season_summary.sort_values("goals", ascending=False).head(1)
best_defense = season_summary.sort_values("goals_conceded").head(1)
most_passes = season_summary.sort_values("Pass", ascending=False).head(1)
most_shots = season_summary.sort_values("total_shots", ascending=False).head(1)

print("Squadra con più goal segnati:")
print(tabulate(top_scoring_team[["team_name", "goals"]], headers="keys", tablefmt="fancy_grid", showindex=False))

print("\nSquadra con meno goal subiti:")
print(tabulate(best_defense[["team_name", "goals_conceded"]], headers="keys", tablefmt="fancy_grid", showindex=False))

print("\nSquadra con più passaggi:")
print(tabulate(most_passes[["team_name", "Pass"]], headers="keys", tablefmt="fancy_grid", showindex=False))

print("\nSquadra con più tiri:")
print(tabulate(most_shots[["team_name", "total_shots"]], headers="keys", tablefmt="fancy_grid", showindex=False))

# =========================================================
# 22. STATISTICHE GIOCATORI
# =========================================================
print("\n================ STATISTICHE GIOCATORI ================\n")

# MIGLIORI MARCATORI
goals_players = df[df["isGoal"] == True]

top_scorers = (
    goals_players
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="goals")
    .sort_values("goals", ascending=False)
)

print("\nMIGLIORI MARCATORI:")
print(tabulate(top_scorers.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
top_scorers.to_csv("top_scorers.csv", index=False)

# GIOCATORI CON PIÙ PASSAGGI
passes_players = df[df["type_displayName"] == "Pass"]

top_passers = (
    passes_players
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="passes")
    .sort_values("passes", ascending=False)
)

print("\nGIOCATORI CON PIÙ PASSAGGI:")
print(tabulate(top_passers.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
top_passers.to_csv("top_passers.csv", index=False)

# GIOCATORI CON PIÙ TIRI
shots_events = df[
    df["type_displayName"].isin([
        "MissedShots",
        "SavedShot",
        "ShotOnPost",
        "Goal"
    ])
].copy()

top_shooters = (
    shots_events
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="shots")
    .sort_values("shots", ascending=False)
)

print("\nGIOCATORI CON PIÙ TIRI:")
print(tabulate(top_shooters.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
top_shooters.to_csv("top_shooters.csv", index=False)

# ASSIST
assists_df = df[
    (df["isGoal"] == True) &
    (df["relatedPlayerId"].notna())
][["relatedPlayerId"]].copy()

player_lookup = df[["playerId", "name", "team_name"]].drop_duplicates()

assists_df = assists_df.merge(
    player_lookup,
    left_on="relatedPlayerId",
    right_on="playerId",
    how="left"
)

top_assists = (
    assists_df.groupby(["name", "team_name"])
    .size()
    .reset_index(name="assist")
    .sort_values("assist", ascending=False)
)

print("\nGIOCATORI CON PIÙ ASSIST:")
print(tabulate(top_assists.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
top_assists.to_csv("top_assists.csv", index=False)

# EFFICIENZA REALIZZATIVA
goals_players_eff = (
    df[df["type_displayName"] == "Goal"]
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Goal")
)

shots_players_eff = (
    shots_events
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Tiri")
)

efficiency_df = shots_players_eff.merge(
    goals_players_eff,
    on=["name", "team_name"],
    how="left"
)

efficiency_df["Goal"] = efficiency_df["Goal"].fillna(0).astype(int)
efficiency_df["Efficienza"] = efficiency_df["Goal"] / efficiency_df["Tiri"]
efficiency_df = efficiency_df[efficiency_df["Tiri"] >= 50].copy()
efficiency_df["Efficienza %"] = (efficiency_df["Efficienza"] * 100).round(2)

efficiency_df = efficiency_df.sort_values(
    by="Efficienza",
    ascending=False
).reset_index(drop=True)

print("\nGIOCATORI CON MIGLIORE EFFICIENZA REALIZZATIVA:")
print(tabulate(
    efficiency_df.head(15)[["name", "team_name", "Tiri", "Goal", "Efficienza %"]],
    headers=["Giocatore", "Squadra", "Tiri", "Goal", "Efficienza %"],
    tablefmt="fancy_grid",
    showindex=False
))
efficiency_df.to_csv("shooting_efficiency.csv", index=False)

# AUTOGOAL
own_goals_df = df[df["isOwnGoal"] == True].copy()

top_own_goals = (
    own_goals_df
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Autogol")
    .sort_values(by="Autogol", ascending=False)
)

print("\nGIOCATORI CON PIÙ AUTOGOAL:")
print(tabulate(top_own_goals.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
top_own_goals.to_csv("top_own_goals.csv", index=False)

# CARTELLINI
cards_df = df[df["cardType_displayName"].notna()].copy()

yellow_cards = (
    cards_df[cards_df["cardType_displayName"] == "Yellow"]
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Gialli")
    .sort_values(by="Gialli", ascending=False)
)

red_cards = (
    cards_df[cards_df["cardType_displayName"].isin(["Red", "SecondYellow"])]
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Rossi")
    .sort_values(by="Rossi", ascending=False)
)

print("\nGIOCATORI CON PIÙ CARTELLINI GIALLI:")
print(tabulate(yellow_cards.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
yellow_cards.to_csv("top_yellow_cards.csv", index=False)

print("\nGIOCATORI CON PIÙ CARTELLINI ROSSI:")
print(tabulate(red_cards.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
red_cards.to_csv("top_red_cards.csv", index=False)

# DIFESA GIOCATORI
tackles = df[
    (df["type_displayName"] == "Tackle") &
    (df["outcomeType_displayName"] == "Successful")
]

tackles_players = (
    tackles.groupby(["name", "team_name"])
    .size()
    .reset_index(name="Tackle riusciti")
    .sort_values(by="Tackle riusciti", ascending=False)
)

interceptions = df[df["type_displayName"] == "Interception"]

interceptions_players = (
    interceptions.groupby(["name", "team_name"])
    .size()
    .reset_index(name="Intercetti")
    .sort_values(by="Intercetti", ascending=False)
)

clearances = df[df["type_displayName"] == "Clearance"]

clearances_players = (
    clearances.groupby(["name", "team_name"])
    .size()
    .reset_index(name="Clearance")
    .sort_values(by="Clearance", ascending=False)
)

defensive_index = tackles_players.merge(
    interceptions_players,
    on=["name", "team_name"],
    how="outer"
).merge(
    clearances_players,
    on=["name", "team_name"],
    how="outer"
).fillna(0)

for col in ["Tackle riusciti", "Intercetti", "Clearance"]:
    defensive_index[col] = defensive_index[col].astype(int)

defensive_index["Indice difensivo"] = (
    defensive_index["Tackle riusciti"] +
    defensive_index["Intercetti"] +
    defensive_index["Clearance"]
)

defensive_index = defensive_index.sort_values(
    by="Indice difensivo",
    ascending=False
)

print("\nTOP DIFENSORI PER INDICE DIFENSIVO:")
print(tabulate(defensive_index.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
defensive_index.to_csv("defensive_index_players.csv", index=False)

# =========================================================
# 22bis. ANALISI PORTIERI
# =========================================================
print("\n================ ANALISI PORTIERI ================\n")

# Portieri con più parate
gk_saves_players = (
    df[
        (df["type_displayName"] == "Save") &
        (df["position"] == "GK")
    ]
    .groupby(["name", "team_name"])
    .size()
    .reset_index(name="Parate")
    .sort_values(by="Parate", ascending=False)
)

print("\nTOP 10 PORTIERI PER PARATE:")
print(tabulate(gk_saves_players.head(10), headers="keys", tablefmt="fancy_grid", showindex=False))
gk_saves_players.to_csv("top_goalkeepers_saves.csv", index=False)

# Clean sheet per squadra
clean_sheets = (
    features_df[features_df["goals_conceded"] == 0]
    .groupby("team_name")
    .size()
    .reset_index(name="Clean Sheet")
    .sort_values(by="Clean Sheet", ascending=False)
)

# Portiere principale per squadra
gk_main = (
    df[df["position"] == "GK"]
    .groupby(["team_name", "name"])
    .size()
    .reset_index(name="presenze")
    .sort_values(["team_name", "presenze"], ascending=[True, False])
    .drop_duplicates("team_name")
)

clean_sheets_gk = clean_sheets.merge(
    gk_main[["team_name", "name"]],
    on="team_name",
    how="left"
)

clean_sheets_gk.columns = ["Squadra", "Clean Sheet", "Portiere"]

print("\nCLEAN SHEET PER SQUADRA:")
print(tabulate(clean_sheets_gk, headers="keys", tablefmt="fancy_grid", showindex=False))
clean_sheets_gk.to_csv("clean_sheets_goalkeepers.csv", index=False)



# =========================================================
# 22.3. ANALISI COORDINATE SPAZIALI
# =========================================================
print("\n================ ANALISI COORDINATE SPAZIALI ================\n")

xy_analysis = features_df.groupby("result")[["mean_x", "mean_y"]].mean()

print(xy_analysis)
xy_analysis.to_csv("xy_analysis_by_result.csv")


# =========================================================
# 23. EDA QUANTITATIVA
# =========================================================
print("\n================ STEP 1: SIGNIFICATIVITÀ DELLE FEATURE ================\n")

feature_numeriche = [
    "Pass",
    "total_shots",
    "shots_on_target",
    "Foul",
    "Tackle",
    "Interception",
    "BallRecovery",
    "Aerial",
    "TakeOn",
    "mean_x",
    "mean_y",
    "Saves",
]

risultati_stat = []

for feature in feature_numeriche:
    medie_classi = features_df.groupby("result")[feature].mean()

    media_vittoria = medie_classi.get("Vittoria", 0)
    media_pareggio = medie_classi.get("Pareggio", 0)
    media_sconfitta = medie_classi.get("Sconfitta", 0)

    std_totale = features_df[feature].std()
    diff_v_s = abs(media_vittoria - media_sconfitta)
    is_significativa = diff_v_s > (0.5 * std_totale)

    risultati_stat.append({
        "Feature": feature,
        "Media Vittoria": round(media_vittoria, 2),
        "Media Pareggio": round(media_pareggio, 2),
        "Media Sconfitta": round(media_sconfitta, 2),
        "Dev Std": round(std_totale, 2),
        "Diff V-S": round(diff_v_s, 2),
        "Discriminante": "SI" if is_significativa else "NO"
    })

df_statistiche = pd.DataFrame(risultati_stat)
df_statistiche["Score"] = df_statistiche["Diff V-S"] / df_statistiche["Dev Std"]
df_statistiche = df_statistiche.sort_values(by="Score", ascending=False)

print(df_statistiche.to_string(index=False))

feature_discriminanti = df_statistiche[df_statistiche["Discriminante"] == "SI"]["Feature"].tolist()
print(f"\nLe feature più discriminanti risultano: {feature_discriminanti}\n")

eda_mean = features_df.groupby("result")[feature_numeriche].mean()
global_std = features_df[feature_numeriche].std()
eda_mean.to_csv("eda_feature_means.csv")
global_std.to_csv("eda_global_std.csv")

# =========================================================
# 24. EDA GRAFICA
# =========================================================
sns.set_theme(style="whitegrid")

# 1. Distribuzione risultati
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=features_df,
    x="result",
    hue="result",
    palette="pastel",
    legend=False
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center",
        va="bottom"
    )
plt.title("Distribuzione dei risultati")
plt.xlabel("Risultato")
plt.ylabel("Numero di osservazioni")
plt.tight_layout()
plt.savefig("eda_distribuzione_risultati.png")
plt.close()

# 2. Tiri totali
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=features_df,
    x="result",
    y="total_shots",
    hue="result",
    palette="pastel",
    legend=False
)
plt.title("Distribuzione dei tiri totali per risultato")
plt.xlabel("Risultato")
plt.ylabel("Tiri totali")
plt.tight_layout()
plt.savefig("eda_boxplot_total_shots.png")
plt.close()

# 3. Tiri in porta
plt.figure(figsize=(8, 5))
sns.barplot(
    data=features_df,
    x="result",
    y="shots_on_target",
    estimator="mean",
    errorbar="sd",
    hue="result",
    palette="Set2",
    legend=False
)
plt.title("Media dei tiri in porta per risultato")
plt.xlabel("Risultato")
plt.ylabel("Tiri in porta")
plt.tight_layout()
plt.savefig("eda_shots_on_target_barplot.png")
plt.close()

# 4. Passaggi
plt.figure(figsize=(8, 5))
sns.barplot(
    data=features_df,
    x="result",
    y="Pass",
    estimator="mean",
    errorbar=None,
    hue="result",
    palette=["red", "gold", "green"],
    legend=False
)
plt.title("Media dei passaggi per risultato")
plt.xlabel("Risultato")
plt.ylabel("Passaggi")
plt.tight_layout()
plt.savefig("eda_passaggi_barplot.png")
plt.close()

# 5. Matrice di correlazione
plt.figure(figsize=(10, 8))
corr = features_df[feature_numeriche + ["goals"]].corr()
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Matrice di correlazione tra le feature")
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png")
plt.close()

# 6. Relazione tiri in porta-goal
plt.figure(figsize=(8, 6))
sns.regplot(
    data=features_df,
    x="shots_on_target",
    y="goals",
    scatter_kws={"alpha": 0.35},
    line_kws={"color": "red"}
)
plt.title("Relazione tra tiri in porta e goal segnati")
plt.xlabel("Tiri in porta")
plt.ylabel("Goal segnati")
plt.tight_layout()
plt.savefig("eda_regplot_shots_on_target_goals.png")
plt.close()

# 7. Top 10 squadre per punti
plt.figure(figsize=(10, 6))
sns.barplot(
    data=classifica.head(10),
    x="points",
    y="team_name",
    hue="team_name",
    palette="viridis",
    legend=False
)
plt.title("Top 10 squadre per punti")
plt.xlabel("Punti")
plt.ylabel("Squadra")
plt.tight_layout()
plt.savefig("league_top10_points.png")
plt.close()

# 8. Top 10 assistman
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_assists.head(10),
    x="assist",
    y="name",
    hue="name",
    palette="viridis",
    legend=False
)
plt.title("Top 10 assistman Premier League 2020-21")
plt.xlabel("Numero di assist")
plt.ylabel("Giocatore")
plt.tight_layout()
plt.savefig("players_top_assists.png")
plt.close()

# 9. Efficienza realizzativa top 15
top15_eff = efficiency_df.head(15).copy()
top15_eff["player_team"] = top15_eff["name"] + "\n(" + top15_eff["team_name"] + ")"

plt.figure(figsize=(12, 6))
colors = sns.color_palette("Spectral", len(top15_eff))

plt.bar(
    top15_eff["player_team"],
    top15_eff["Efficienza %"],
    color=colors
)

plt.title("Top 15 giocatori per efficienza realizzativa", fontsize=14)
plt.xlabel("Giocatore")
plt.ylabel("Efficienza realizzativa (%)")
plt.xticks(rotation=45, ha="right")

for i, (eff, tiri) in enumerate(zip(top15_eff["Efficienza %"], top15_eff["Tiri"])):
    plt.text(i, eff + 0.3, f"{eff:.1f}%\n({tiri} tiri)", ha="center", fontsize=9)

plt.ylim(0, top15_eff["Efficienza %"].max() + 5)
plt.tight_layout()
plt.savefig("players_shooting_efficiency.png")
plt.close()

# 10. Top 10 indice difensivo
plt.figure(figsize=(10, 6))
sns.barplot(
    data=defensive_index.head(10),
    x="Indice difensivo",
    y="name",
    hue="name",
    palette="rainbow",
    legend=False
)
plt.title("Top 10 giocatori per indice difensivo")
plt.xlabel("Indice difensivo totale")
plt.ylabel("Giocatore")
plt.tight_layout()
plt.savefig("players_defensive_index.png")
plt.close()


# =========================================================
# CLASSIFICAZIONE - TWO STAGE XGBOOST
# Versione per file main.py
# =========================================================



# =========================================================
# FUNZIONE DI TUNING
# =========================================================
def tune_xgb(X_train, y_train, scale_pos_weight=None):
    param_grid = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5, 7],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_lambda": [1, 2, 5, 10]
    }

    model = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=25,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    search.fit(X_train, y_train)
    return search.best_estimator_


# =========================================================
# CLASSIFICAZIONE TWO-STAGE
# =========================================================
def run_two_stage_classification(features_df):
    print("\n" + "=" * 70)
    print("CLASSIFICAZIONE - TWO STAGE XGBOOST")
    print("=" * 70)

    # =========================================================
    # 1. DATASET BASE
    # =========================================================
    clf_df = features_df.copy()

    # =========================================================
    # 2. FEATURE PRIMITIVE PER SQUADRA
    # =========================================================
    base_cols = [
        "BallRecovery",
        "Clearance",
        "Pass",
        "Tackle",
        "Interception",
        "Foul",
        "Card",
        "MissedShots",
        "ShotOnPost",
        "SavedShot",
        "Aerial",
        "TakeOn",
        "Saves",
        "mean_x",
        "mean_y",
        "mean_endX",
        "mean_endY",
        "total_shots",
        "shots_on_target"
    ]
    base_cols = [c for c in base_cols if c in clf_df.columns]

    # =========================================================
    # 3. MERGE CON L'AVVERSARIO
    # =========================================================
    opp_df = clf_df[["match_id", "team_name"] + base_cols].copy()
    opp_df = opp_df.rename(columns={"team_name": "opponent"})
    opp_df = opp_df.rename(columns={c: f"{c}_opp" for c in base_cols})

    clf_df = clf_df.merge(opp_df, on=["match_id", "opponent"], how="left")
    clf_df = clf_df.rename(columns={c: f"{c}_team" for c in base_cols})

    # =========================================================
    # 4. FEATURE COMPARATIVE
    # =========================================================
    for col in base_cols:
        clf_df[f"{col}_diff"] = clf_df[f"{col}_team"] - clf_df[f"{col}_opp"]
        clf_df[f"{col}_diff_abs"] = clf_df[f"{col}_diff"].abs()
        clf_df[f"{col}_ratio"] = clf_df[f"{col}_team"] / (clf_df[f"{col}_opp"] + 1)

    # =========================================================
    # 5. FEATURE AGGREGATE SENSATE
    # =========================================================
    clf_df["attack_strength_team"] = (
        clf_df.get("total_shots_team", 0)
        + 2 * clf_df.get("shots_on_target_team", 0)
        + clf_df.get("ShotOnPost_team", 0)
        + clf_df.get("MissedShots_team", 0)
    )

    clf_df["attack_strength_opp"] = (
        clf_df.get("total_shots_opp", 0)
        + 2 * clf_df.get("shots_on_target_opp", 0)
        + clf_df.get("ShotOnPost_opp", 0)
        + clf_df.get("MissedShots_opp", 0)
    )

    clf_df["attack_strength_diff"] = clf_df["attack_strength_team"] - clf_df["attack_strength_opp"]
    clf_df["attack_strength_diff_abs"] = clf_df["attack_strength_diff"].abs()

    clf_df["defense_strength_team"] = (
        clf_df.get("Clearance_team", 0)
        + clf_df.get("Interception_team", 0)
        + clf_df.get("Tackle_team", 0)
        + clf_df.get("BallRecovery_team", 0)
    )

    clf_df["defense_strength_opp"] = (
        clf_df.get("Clearance_opp", 0)
        + clf_df.get("Interception_opp", 0)
        + clf_df.get("Tackle_opp", 0)
        + clf_df.get("BallRecovery_opp", 0)
    )

    clf_df["defense_strength_diff"] = clf_df["defense_strength_team"] - clf_df["defense_strength_opp"]
    clf_df["defense_strength_diff_abs"] = clf_df["defense_strength_diff"].abs()

    clf_df["shot_accuracy_team"] = clf_df.get("shots_on_target_team", 0) / (clf_df.get("total_shots_team", 0) + 1)
    clf_df["shot_accuracy_opp"] = clf_df.get("shots_on_target_opp", 0) / (clf_df.get("total_shots_opp", 0) + 1)
    clf_df["shot_accuracy_diff"] = clf_df["shot_accuracy_team"] - clf_df["shot_accuracy_opp"]
    clf_df["shot_accuracy_diff_abs"] = clf_df["shot_accuracy_diff"].abs()

    clf_df["pass_per_shot_team"] = clf_df.get("Pass_team", 0) / (clf_df.get("total_shots_team", 0) + 1)
    clf_df["pass_per_shot_opp"] = clf_df.get("Pass_opp", 0) / (clf_df.get("total_shots_opp", 0) + 1)
    clf_df["pass_per_shot_diff"] = clf_df["pass_per_shot_team"] - clf_df["pass_per_shot_opp"]

    # =========================================================
    # 6. FEATURE DI EQUILIBRIO PARTITA
    # =========================================================
    clf_df["total_shots_match"] = clf_df.get("total_shots_team", 0) + clf_df.get("total_shots_opp", 0)
    clf_df["shots_on_target_match"] = clf_df.get("shots_on_target_team", 0) + clf_df.get("shots_on_target_opp", 0)
    clf_df["passes_match"] = clf_df.get("Pass_team", 0) + clf_df.get("Pass_opp", 0)

    clf_df["def_actions_match"] = (
        clf_df.get("Clearance_team", 0) + clf_df.get("Interception_team", 0) +
        clf_df.get("Tackle_team", 0) + clf_df.get("BallRecovery_team", 0) +
        clf_df.get("Clearance_opp", 0) + clf_df.get("Interception_opp", 0) +
        clf_df.get("Tackle_opp", 0) + clf_df.get("BallRecovery_opp", 0)
    )

    clf_df["shot_balance"] = clf_df["shots_on_target_diff_abs"] / (clf_df["shots_on_target_match"] + 1)
    clf_df["territorial_balance"] = clf_df["mean_x_diff_abs"] + clf_df["mean_endX_diff_abs"]
    clf_df["defensive_balance"] = clf_df["defense_strength_diff_abs"] / (clf_df["def_actions_match"] + 1)

    # =========================================================
    # 7. COSTRUZIONE X E y
    # =========================================================
    drop_cols = [
        "match_id", "team_name", "opponent",
        "result", "Goal", "goals", "goals_conceded", "points"
    ]
    drop_cols = [c for c in drop_cols if c in clf_df.columns]

    X_full = clf_df.drop(columns=drop_cols).copy()
    X_full = X_full.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_full = clf_df["result"].copy()

    print("\nOVERVIEW DATASET")
    print(f"- Numero osservazioni: {X_full.shape[0]}")
    print(f"- Numero feature disponibili: {X_full.shape[1]}")
    print("- Distribuzione classi:")
    print(y_full.value_counts())

    # =========================================================
    # 8. TRAIN / VALIDATION / TEST
    # =========================================================
    X_temp, X_test_full, y_temp, y_test_full = train_test_split(
        X_full, y_full,
        test_size=0.20,
        random_state=42,
        stratify=y_full
    )

    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_temp, y_temp,
        test_size=0.125,
        random_state=42,
        stratify=y_temp
    )

    # =========================================================
    # 9. FEATURE FINALI
    # =========================================================
    stage1_features = [
        "total_shots_diff_abs",
        "shots_on_target_diff_abs",
        "Pass_diff_abs",
        "Tackle_diff_abs",
        "mean_x_diff_abs",
        "mean_endX_diff_abs",
        "shot_balance",
        "territorial_balance"
    ]
    stage1_features = [c for c in stage1_features if c in X_train_full.columns]

    stage2_features = [
        "attack_strength_diff",
        "defense_strength_diff",
        "shots_on_target_diff",
        "shot_accuracy_diff",
        "pass_per_shot_diff",
        "shots_on_target_team",
        "shots_on_target_opp",
        "Clearance_diff"
    ]
    stage2_features = [c for c in stage2_features if c in X_train_full.columns]

    print("\nFEATURE FINALI UTILIZZATE")
    print("- Stadio 1 (Pareggio vs Non Pareggio):")
    for f in stage1_features:
        print(f"  - {f}")

    print("- Stadio 2 (Vittoria vs Sconfitta):")
    for f in stage2_features:
        print(f"  - {f}")

    # =========================================================
    # 10. DATASET PRIMO STADIO
    # =========================================================
    X_train_s1 = X_train_full[stage1_features].copy()
    X_val_s1 = X_val_full[stage1_features].copy()
    X_test_s1 = X_test_full[stage1_features].copy()

    y_train_s1 = (y_train_full == "Pareggio").astype(int)
    y_val_s1 = (y_val_full == "Pareggio").astype(int)
    y_test_s1 = (y_test_full == "Pareggio").astype(int)

    # =========================================================
    # 11. DATASET SECONDO STADIO
    # =========================================================
    mask_train_s2 = y_train_full != "Pareggio"
    mask_val_s2 = y_val_full != "Pareggio"
    mask_test_s2 = y_test_full != "Pareggio"

    X_train_s2 = X_train_full.loc[mask_train_s2, stage2_features].copy()
    X_val_s2 = X_val_full.loc[mask_val_s2, stage2_features].copy()
    X_test_s2 = X_test_full.loc[mask_test_s2, stage2_features].copy()

    y_train_s2 = y_train_full[mask_train_s2].map({"Sconfitta": 0, "Vittoria": 1})
    y_val_s2 = y_val_full[mask_val_s2].map({"Sconfitta": 0, "Vittoria": 1})
    y_test_s2 = y_test_full[mask_test_s2].map({"Sconfitta": 0, "Vittoria": 1})

    # =========================================================
    # 12. ADDESTRAMENTO MODELLI
    # =========================================================
    spw_s1 = (len(y_train_s1) - y_train_s1.sum()) / y_train_s1.sum()

    best_s1 = tune_xgb(X_train_s1, y_train_s1, scale_pos_weight=spw_s1)
    best_s2 = tune_xgb(X_train_s2, y_train_s2)

    # =========================================================
    # 13. RISULTATI DEL SECONDO MODELLO
    # =========================================================
    pred_val_s2 = best_s2.predict(X_val_s2)

    print("\n" + "=" * 70)
    print("RISULTATI DEL SECONDO MODELLO (VITTORIA VS SCONFITTA)")
    print("=" * 70)
    print(f"Accuracy: {accuracy_score(y_val_s2, pred_val_s2):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_val_s2, pred_val_s2):.4f}")
    print(f"F1: {f1_score(y_val_s2, pred_val_s2):.4f}")
    print(classification_report(y_val_s2, pred_val_s2, zero_division=0))

    # =========================================================
    # 14. OTTIMIZZAZIONE SOGLIA DEL PRIMO MODELLO SU VALIDATION
    # =========================================================
    proba_s1_val = best_s1.predict_proba(X_val_s1)[:, 1]
    candidate_thresholds = np.arange(0.25, 0.65, 0.01)

    best_threshold = None
    best_macro_val = -1

    for thr in candidate_thresholds:
        preds_val = []

        for i in range(len(X_val_full)):
            proba_draw = proba_s1_val[i]

            if proba_draw >= thr:
                preds_val.append("Pareggio")
            elif proba_draw < (thr - 0.10):
                row_s2 = X_val_full.iloc[[i]][stage2_features]
                pred_s2 = best_s2.predict(row_s2)[0]
                preds_val.append("Vittoria" if pred_s2 == 1 else "Sconfitta")
            else:
                preds_val.append("Pareggio")

        macro_val = f1_score(y_val_full, preds_val, average="macro", zero_division=0)

        if macro_val > best_macro_val:
            best_macro_val = macro_val
            best_threshold = thr

    print("\nSOGLIA OTTIMALE MODELLO 1")
    print(f"- Threshold ottimale: {best_threshold:.4f}")
    print(f"- Miglior F1 Macro su validation: {best_macro_val:.4f}")

    # =========================================================
    # 15. RISULTATI DEL PRIMO MODELLO
    # =========================================================
    pred_val_s1 = (proba_s1_val >= best_threshold).astype(int)

    print("\n" + "=" * 70)
    print("RISULTATI DEL PRIMO MODELLO (PAREGGIO VS NON PAREGGIO)")
    print("=" * 70)
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_val_s1, pred_val_s1):.4f}")
    print(classification_report(
        y_val_s1,
        pred_val_s1,
        target_names=["Non Pareggio", "Pareggio"],
        zero_division=0
    ))

    # =========================================================
    # 16. RISULTATI FINALI DEL MODELLO TWO-STAGE SU TEST
    # =========================================================
    proba_s1_test = best_s1.predict_proba(X_test_s1)[:, 1]
    best_pred_final = []

    for i in range(len(X_test_full)):
        if proba_s1_test[i] >= best_threshold:
            best_pred_final.append("Pareggio")
        else:
            row_s2 = X_test_full.iloc[[i]][stage2_features]
            pred_s2 = best_s2.predict(row_s2)[0]
            best_pred_final.append("Vittoria" if pred_s2 == 1 else "Sconfitta")

    print("\n" + "=" * 70)
    print("RISULTATI FINALI DEL MODELLO TWO-STAGE")
    print("=" * 70)
    print(f"Accuracy: {accuracy_score(y_test_full, best_pred_final):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test_full, best_pred_final):.4f}")
    print(f"Precision Macro: {precision_score(y_test_full, best_pred_final, average='macro', zero_division=0):.4f}")
    print(f"Recall Macro: {recall_score(y_test_full, best_pred_final, average='macro', zero_division=0):.4f}")
    print(f"F1 Macro: {f1_score(y_test_full, best_pred_final, average='macro', zero_division=0):.4f}")
    print(f"F1 Weighted: {f1_score(y_test_full, best_pred_final, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_test_full, best_pred_final, zero_division=0))

    cm = confusion_matrix(
        y_test_full,
        best_pred_final,
        labels=["Vittoria", "Pareggio", "Sconfitta"]
    )
    cm_df = pd.DataFrame(
        cm,
        index=["Reale_Vittoria", "Reale_Pareggio", "Reale_Sconfitta"],
        columns=["Pred_Vittoria", "Pred_Pareggio", "Pred_Sconfitta"]
    )

    print("\nMATRICE DI CONFUSIONE FINALE")
    print(cm_df)

    return {
        "best_s1": best_s1,
        "best_s2": best_s2,
        "best_threshold": best_threshold,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "final_predictions": best_pred_final,
        "y_test_full": y_test_full,
        "confusion_matrix": cm_df
    }


# =========================================================
# CHIAMATA DA MAIN
# =========================================================
# Da eseguire dopo aver già costruito features_df
results_two_stage = run_two_stage_classification(features_df)