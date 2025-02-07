import streamlit as st
import pandas as pd
#st.set_page_config(layout="wide")
st.title("Wahlomat Visualizer")
st.markdown("Dieses Dashboard dient der Visualisierung von Aussagen der Parteien zum Wahl-O-Mat der Bundestagswahl 2025. "
        "Der Datensatz ist verfügbar unter: https://www.bpb.de/themen/wahl-o-mat/bundestagswahl-2025/558463/download/")
st.info("Es ist mir leider nicht gestattet, Funktionalitäten zu implementieren, die "
        "die eigene Position visualisieren. Das darf leider nur der Wahl-O-Mat. "
        "Macht er allerdings nicht wirklich naja.")
excel = pd.ExcelFile("Wahl-O-Mat Bundestagswahl 2025_Datensatz_v1.01.xlsx")
df = pd.read_excel(excel, "Datensatz BTW 2025")

st.header("Rohdaten")
st.markdown("Im folgenden könnt ihr den Datensatz von BPB anschauen. Keine Sorge.")
st.dataframe(df)

st.header("Übereinstimmung zwischen Parteien")
parteien = df["Partei: Kurzbezeichnung"].unique()
st.markdown("")

