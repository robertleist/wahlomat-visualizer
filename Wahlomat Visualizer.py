import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# st.set_page_config(layout="wide")
st.title("Wahl-O-Mat Visualizer")
st.markdown(
    "Aufgrund eines interessanten "
    "[Reddit Posts in r/de](https://www.reddit.com/r/de/comments/1ij54lu/wahlomat_%C3%BCberschneidung_analyse/) "
    "kam ich darauf dieses Dashboard zu erstellen. "
    "Es dient der Visualisierung von Aussagen der Parteien zum "
    "[Wahl-O-Mat zur Bundestagswahl 2025](https://www.wahl-o-mat.de/bundestagswahl2025/app/main_app.html). "
    "Der Datensatz ist verfügbar unter: https://www.bpb.de/themen/wahl-o-mat/bundestagswahl-2025/558463/download/")
st.warning("Es ist mir leider nicht gestattet, Funktionalitäten zu implementieren, die "
           "die eigene Position visualisieren. Das darf leider nur der Wahl-O-Mat selbst.")
st.info("Wenn du diese Seite auf dem PC öffnest, kannst du das Layout zu 'wide' ändern, um mehr Platz zu haben. Geh "
        "dafür auf die drei Punkte oben rechts und wähle 'Settings -> Appearance -> Wide Mode'.")
st.info("Wenn du Vorschläge oder Verbesserungen hast, kannst du diese gerne auf GitHub als "
        "[Issue](https://github.com/robertleist/wahlomat-visualizer/issues) oder als "
        "[Pull Request](https://github.com/robertleist/wahlomat-visualizer/pulls) einreichen.")
excel = pd.ExcelFile("Wahl-O-Mat Bundestagswahl 2025_Datensatz_v1.01.xlsx")
df = pd.read_excel(excel, "Datensatz BTW 2025")
with st.expander("Rohdaten"):
    st.markdown("Im folgenden könnt ihr den rohen Datensatz von BPB anschauen.")
    st.dataframe(df)

with st.expander("Fragen und Antworten"):
    st.error("Diese Funktionalität bietet der Wahl-O-Mat schon an. Daher werde ich sie hier erstmal nicht erneut "
             "implementieren.")

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

st.header("Dimensionality Reduction und Clustering")
st.markdown("Im folgenden könnt ihr die Parteien anhand ihrer Übereinstimmung clustern. Clustering Algorithmen sind "
            "etwas technischer und können nicht immer intuitiv interpretiert werden. Ich versuche es trotzdem so "
            "einfach wie möglich zu halten.")
st.info("Um Parteien zu clustern muss man erst die Dimensionality reduzieren. Das bedeutet man projiziert die "
        "Daten in einen niedriger dimensionalen Raum. In diesem Fall wird die Dimensionalität von 38 auf 2 reduziert. "
        "Dabei verliert man etwas Information, zeigt jedoch Parteien, die ähnliche Antworten haben, näher zusammen.")
with st.form("Dimensionality Reduction und Clustering"):
    with st.expander("Inkludierte Fragen (Standardmäßig alle)"):
        questions_to_include = st.multiselect(
            "Wähle Fragen aus, die für die Dimensionality Reduction genutzt werden sollen.",
            df["These: Titel"].unique(), default=df["These: Titel"].unique())
    with st.expander("Inkludierte Parteien (Standardmäßig alle)"):
        parties_to_include = st.multiselect(
            "Wähle Parteien aus, die für die Dimensionality Reduction genutzt werden sollen.",
            df["Partei: Kurzbezeichnung"].unique(), default=df["Partei: Kurzbezeichnung"].unique())
    col1, col2 = st.columns(2)
    with col1:
        dim_algo = st.radio("Wähle den Algorithmus zur Dimensionality Reduction", ["PCA", "t-SNE"], index=0)
    with col2:
        clu_algo = st.radio("Wähle den Algorithmus zum Clustern", ["KMeans", "Hierarchical Clustering"], index=0)
    clu_num = st.slider("Wähle die Anzahl der Cluster", 2, 10, 2)
    st.form_submit_button("Neu berechnen")

# Filter questions
df_filtered = df[df["These: Titel"].isin(questions_to_include)]
df_filtered = df_filtered[df_filtered["Partei: Kurzbezeichnung"].isin(parties_to_include)]
df_to_reduce = df_filtered.pivot(index="Partei: Kurzbezeichnung", columns="These: Titel", values="Position: Position")

def map_answers(x):
    if x == "stimme zu":
        return 1
    elif x == "neutral":
        return 0
    else:
        return -1

df_to_reduce = df_to_reduce.map(map_answers)

# Dimensionality Reduction
if dim_algo == "PCA":
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df_to_reduce)
else:
    st.warning("t-SNE kann unterschiedliche Ergebnisse lieferen pro Durchlauf. Für reproduzierbare Ergebnisse "
               "empfehle ich PCA.")
    tsne = TSNE(n_components=2, perplexity=5, n_iter=5000)
    reduced = tsne.fit_transform(df_to_reduce)


# Clustering
if clu_algo == "KMeans":
    clusterer = KMeans(n_clusters=clu_num, random_state=42, n_init=10)
elif clu_algo == "Hierarchical Clustering":
    clusterer = AgglomerativeClustering(n_clusters=clu_num)

clusters = clusterer.fit_predict(reduced)
reduced_df = pd.DataFrame(reduced, columns=["x", "y"], index=df_to_reduce.index)
reduced_df["Cluster"] = clusters
if clu_algo == "Hierarchical Clustering":
    with st.expander("Hierarchical Dendrogram"):
        # Hierarchische Cluster-Darstellung
        if clu_algo == "Hierarchical Clustering":
            plt.figure(figsize=(10, 5))
            linkage_matrix = sch.linkage(reduced, method='ward')
            sch.dendrogram(linkage_matrix, labels=reduced_df.index, leaf_rotation=90, leaf_font_size=10)
            st.pyplot(plt)

# Plot
fig = px.scatter(reduced_df, x="x", y="y", text=reduced_df.index, color=reduced_df["Cluster"].astype(str),
                 title="Dimensionality Reduction und Clustering", color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig)
