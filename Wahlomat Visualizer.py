import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

# st.set_page_config(layout="wide")
st.title("Wahlomat Visualizer")
st.markdown(
    "Dieses Dashboard dient der Visualisierung von Aussagen der Parteien zum Wahl-O-Mat der Bundestagswahl 2025. "
    "Der Datensatz ist verfügbar unter: https://www.bpb.de/themen/wahl-o-mat/bundestagswahl-2025/558463/download/")
st.info("Es ist mir leider nicht gestattet, Funktionalitäten zu implementieren, die "
        "die eigene Position visualisieren. Das darf leider nur der Wahl-O-Mat. "
        "Macht er allerdings nicht wirklich naja.")
excel = pd.ExcelFile("Wahl-O-Mat Bundestagswahl 2025_Datensatz_v1.01.xlsx")
df = pd.read_excel(excel, "Datensatz BTW 2025")

st.header("Rohdaten")
st.markdown("Im folgenden könnt ihr den rohen Datensatz von BPB anschauen.")
st.dataframe(df)

st.header("Übereinstimmung zwischen Parteien")
st.markdown("Im folgenden könnt ihr die Übereinstimmung zwischen Parteien visualisieren, wobei Übereinstimmung als der "
            "prozentuale Anteil gleicher Antworten definiert ist.")
st.info("Wenn du gegenteilige Ansichten als negativen Wert werten willst, kannst du diesen Switch drücken."
        "   Beispiel: Wenn Partei A zustimmt und Partei B nicht "
        "zustimmt, dann wird die Übereinstimmung für diese Frage normalerweise als 0 gewertet. In der erweiterten "
        "Übereinstimmung würde diese Frage als -1 gewertet werden.")
advanced_agreement = st.toggle("Erweiterte Übereinstimmung")
parties_unique = df["Partei: Kurzbezeichnung"].unique()
parties = st.multiselect("Wähle Parteien zum Vergleichen der Übereinstimmung aus", parties_unique,
                         default=parties_unique)
agreement = np.zeros((len(parties), len(parties)))
for i, p1 in enumerate(parties):
    for j, p2 in enumerate(parties):
        party1 = df[df["Partei: Kurzbezeichnung"] == p1]
        party2 = df[df["Partei: Kurzbezeichnung"] == p2]
        if advanced_agreement:
            agreement_count = np.count_nonzero(
                party1["Position: Position"].values == party2["Position: Position"].values)
            disagreement_count = np.count_nonzero(
                np.logical_or(
                    np.logical_and(party1["Position: Position"].values == "stimme zu",
                                   party2["Position: Position"].values == "stimme nicht zu"),
                    np.logical_and(party1["Position: Position"].values == "stimme nicht zu",
                                   party2["Position: Position"].values == "stimme zu"),
                )
            )
            print(p1, p2, agreement_count, disagreement_count)
            agreement[i, j] = (agreement_count - disagreement_count) / 38
        else:
            agreement[i, j] = (
                    np.count_nonzero(party1["Position: Position"].values == party2["Position: Position"].values)
                    / 38)
agreement_df = pd.DataFrame(agreement, columns=parties, index=parties)


# Function to color cells based on percentage
def color_percentage_normal(val):
    color = f"rgb({255 - int(val * 255)}, {int(val * 255)}, 100)"  # Red to green
    return f"background-color: {color}; color: black;"

def color_percentage_advanced(val):
    new_val = (val + 1) / 2
    color = f"rgb({255 - int(new_val * 255)}, {int(new_val * 255)}, 100)"  # Red to green
    return f"background-color: {color}; color: black;"


# Apply styling
if advanced_agreement:
    styled_agreement_df = agreement_df.style.format("{:.1%}").applymap(
        color_percentage_advanced)
else:
    styled_agreement_df = agreement_df.style.format("{:.1%}").applymap(
        color_percentage_normal)
st.dataframe(styled_agreement_df, use_container_width=True)

st.header("Clustering")
st.markdown("Im folgenden könnt ihr die Parteien anhand ihrer Übereinstimmung clustern.")
st.info
