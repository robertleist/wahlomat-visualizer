import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def party_colors(party):
    colors = {
        "CDU / CSU": "black",
        "SPD": "red",
        "AfD": "blue",
        "FDP": "yellow",
        "Die Linke": "purple",
        "GRÜNE": "green",
    }
    if party in colors:
        return colors[party]
    else:
        return "grey"


# st.set_page_config(layout="wide")
st.title("Wahl-O-Mat Visualizer")
st.markdown(
    "Aufgrund eines interessanten "
    "[Reddit Posts in r/de](https://www.reddit.com/r/de/comments/1ij54lu/wahlomat_%C3%BCberschneidung_analyse/) "
    "kam ich darauf dieses Dashboard zu erstellen. "
    "Es dient der Visualisierung von Aussagen der Parteien zum "
    "[Wahl-O-Mat zur Bundestagswahl 2025](https://www.wahl-o-mat.de/bundestagswahl2025/app/main_app.html). "
    "Der Datensatz ist verfügbar unter: https://www.bpb.de/themen/wahl-o-mat/bundestagswahl-2025/558463/download/")
st.error("Dieses Dashboard/Analyse ist nicht von der Bundeszentrale für politische Bildung (bpb) oder dem "
         "Wahl-O-Mat erstellt.")
st.error("Ich garantiere nicht für die Richtigkeit der Daten oder des Codes. Bitte überprüfe die Daten und den Code"
         " selbst, bevor du sie "
         "verwendest.")
st.warning("Es ist mir leider nicht gestattet, Funktionalitäten zu implementieren, die "
           "die eigene Position visualisieren. Das darf leider nur der Wahl-O-Mat selbst.")
st.toast("Wenn du diese Seite auf dem PC öffnest, kannst du das Layout zu 'wide' ändern, um mehr Platz zu haben. Geh "
         "dafür auf die drei Punkte oben rechts und wähle 'Settings -> Appearance -> Wide Mode'.")
st.subheader("Verbesserungsvorschläge")
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
st.markdown("""
Im folgenden kannst du die Übereinstimmung zwischen Parteien visualisieren, wobei du mehrere Metriken auswählen kannst:

- **Prozentuale Übereinstimmung**: Wird als Anteil der gleichen Antworten berechnet.  
- **Erweiterte Übereinstimmung**: Wird als Anteil der gleichen Antworten minus der gegenteiligen Antworten berechnet.  
- **Korrelationskoeffizienten**: Zeigen die Korrelation zwischen den Übereinstimmungen der Parteien an. Kann auf Basis
    einer der anderen Metriken berechnet werden. 
""")

metric = st.radio("Wähle die Metrik zur Berechnung der Übereinstimmung",
                  ["Prozentuale Übereinstimmung",
                   "Erweiterte Übereinstimmung",
                   "Korrelationskoeffizienten (Für prozentuale Ü.)",
                   "Korrelationskoeffizienten (Für erweiterte Ü.)"])
st.info("Wenn du gegenteilige Ansichten als negativen Wert werten willst, kannst du diesen Switch drücken."
        "   Beispiel: Wenn Partei A zustimmt und Partei B nicht "
        "zustimmt, dann wird die Übereinstimmung für diese Frage normalerweise als 0 gewertet. In der erweiterten "
        "Übereinstimmung würde diese Frage als -1 gewertet werden.")
parties_unique = df["Partei: Kurzbezeichnung"].unique()
with st.expander("Parteinauswahl (Standardmäßig alle)"):
    parties = st.multiselect("Wähle Parteien zum Vergleichen der Übereinstimmung aus", parties_unique,
                             default=parties_unique)
with st.expander("Fragenauswahl (Standardmäßig alle)"):
    questions = st.multiselect("Wähle Fragen zum Vergleichen der Übereinstimmung aus", df["These: Titel"].unique(),
                               default=df["These: Titel"].unique())
agreement = np.zeros((len(parties), len(parties)))
overlap_df = df[df["These: Titel"].isin(questions)]
advanced_agreement = metric == "Erweiterte Übereinstimmung" or metric == "Korrelationskoeffizienten (Für erweiterte Ü.)"
for i, p1 in enumerate(parties):
    for j, p2 in enumerate(parties):
        party1 = overlap_df[overlap_df["Partei: Kurzbezeichnung"] == p1]
        party2 = overlap_df[overlap_df["Partei: Kurzbezeichnung"] == p2]
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
            agreement[i, j] = (agreement_count - disagreement_count) / len(questions)
        else:
            agreement[i, j] = (
                    np.count_nonzero(party1["Position: Position"].values == party2["Position: Position"].values)
                    / len(questions))

agreement_df = pd.DataFrame(agreement, columns=parties, index=parties)
if "Korrelationskoeffizient" in metric:
    agreement_df = agreement_df.corr()


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
st.subheader("Venn Diagram der Übereinstimmung")
st.info("Im folgenden kannst du die Übereinstimmung zwischen drei Parteien visualisieren. "
        "Das Venn Diagram zeigt die Anzahl der gleichen Antworten und der unterschiedlichen Antworten an."
        "Die Farbe der Punkte gibt an, wie die Parteien geantwortet haben. Grün bedeutet 'stimme zu', blau bedeutet "
        "'neutral' und rot bedeutet 'stimme nicht zu'. Durch Hovern über die Punkte kannst du die Fragen sehen.")
parties_venn = st.multiselect("Wähle Parteien für das Venn Diagram aus", parties, max_selections=3)


def jitter_points(x, y, num_points, scale=0.1):
    return np.column_stack((np.random.normal(x, scale, num_points), np.random.normal(y, scale, num_points)))


if len(parties_venn) != 3:
    st.warning("Bitte wähle drei Parteien aus.")
else:
    party1_df = overlap_df[overlap_df["Partei: Kurzbezeichnung"] == parties_venn[0]]
    party2_df = overlap_df[overlap_df["Partei: Kurzbezeichnung"] == parties_venn[1]]
    party3_df = overlap_df[overlap_df["Partei: Kurzbezeichnung"] == parties_venn[2]]
    party1_venn = party1_df["Position: Position"].values
    party2_venn = party2_df["Position: Position"].values
    party3_venn = party3_df["Position: Position"].values
    all_same = np.logical_and(np.logical_and(party1_venn == party2_venn, party2_venn == party3_venn),
                              party1_venn == party3_venn)
    p1p2 = np.logical_and(party1_venn == party2_venn, party1_venn != party3_venn)
    p1p3 = np.logical_and(party1_venn == party3_venn, party1_venn != party2_venn)
    p2p3 = np.logical_and(party2_venn == party3_venn, party2_venn != party1_venn)
    p1 = np.logical_not(np.logical_or(all_same, np.logical_or(p1p2, p1p3)))
    p2 = np.logical_not(np.logical_or(all_same, np.logical_or(p1p2, p2p3)))
    p3 = np.logical_not(np.logical_or(all_same, np.logical_or(p1p3, p2p3)))

    # Generate jittered points
    points_all_same = jitter_points(0, 0, np.sum(all_same))
    points_p1p2 = jitter_points(0, 1, np.sum(p1p2))
    points_p1p3 = jitter_points(-1, -0.5, np.sum(p1p3))
    points_p2p3 = jitter_points(1, -0.5, np.sum(p2p3))
    points_p1 = jitter_points(-1, 0.5, np.sum(p1))
    points_p2 = jitter_points(1, 0.5, np.sum(p2))
    points_p3 = jitter_points(0, -1, np.sum(p3))


    def convex_hull_points(points):
        points = np.array(points)  # Ensure input is a NumPy array
        if len(points) < 3:
            return points  # Return the same points without modification

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]  # Extract actual hull points
        centroid = np.mean(hull_points, axis=0)
        expanded_points = centroid + (hull_points - centroid) * 1.2  # Scale outward
        return np.vstack([expanded_points, expanded_points[0]])


    hull_p1 = convex_hull_points(np.concatenate([points_p1, points_p1p2, points_p1p3, points_all_same], axis=0))
    hull_p2 = convex_hull_points(np.concatenate([points_p2, points_p1p2, points_p2p3, points_all_same], axis=0))
    hull_p3 = convex_hull_points(np.concatenate([points_p3, points_p1p3, points_p2p3, points_all_same], axis=0))

    all_same_text = party1_df["These: Titel"][all_same]
    p1p2_text = party1_df["These: Titel"][p1p2]
    p1p3_text = party1_df["These: Titel"][p1p3]
    p2p3_text = party2_df["These: Titel"][p2p3]
    p1_text = party1_df["These: Titel"][p1]
    p2_text = party2_df["These: Titel"][p2]
    p3_text = party3_df["These: Titel"][p3]


    def answer_to_color(answer):
        if answer == "stimme zu":
            return "green"
        elif answer == "neutral":
            return "blue"
        else:
            return "red"


    all_same_answer = party1_df["Position: Position"][all_same].map(answer_to_color)
    p1p2_answer = party1_df["Position: Position"][p1p2].map(answer_to_color)
    p1p3_answer = party1_df["Position: Position"][p1p3].map(answer_to_color)
    p2p3_answer = party2_df["Position: Position"][p2p3].map(answer_to_color)
    p1_answer = party1_df["Position: Position"][p1].map(answer_to_color)
    p2_answer = party2_df["Position: Position"][p2].map(answer_to_color)
    p3_answer = party3_df["Position: Position"][p3].map(answer_to_color)

    fig = go.Figure(layout=dict(height=900))
    # Add the hulls
    fig.add_trace(go.Scatter(x=hull_p1[:, 0], y=hull_p1[:, 1], mode='lines', name=f"{parties_venn[0]}",
                             line=dict(color=party_colors(parties_venn[0]), width=2), fill='toself'))
    fig.add_trace(go.Scatter(x=hull_p2[:, 0], y=hull_p2[:, 1], mode='lines', name=f"{parties_venn[1]}",
                             line=dict(color=party_colors(parties_venn[1]), width=2), fill='toself'))
    fig.add_trace(go.Scatter(x=hull_p3[:, 0], y=hull_p3[:, 1], mode='lines', name=f"{parties_venn[2]}",
                             line=dict(color=party_colors(parties_venn[2]), width=2), fill='toself'))

    # Add the scatter points
    fig.add_trace(go.Scatter(x=points_all_same[:, 0], y=points_all_same[:, 1], mode='markers', name="Alle gleich",
                             marker=dict(color=all_same_answer, size=10), text=all_same_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p1p2[:, 0], y=points_p1p2[:, 1], mode='markers',
                             name=f"{parties_venn[0]} und {parties_venn[1]}",
                             marker=dict(color=p1p2_answer, size=10), text=p1p2_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p1p3[:, 0], y=points_p1p3[:, 1], mode='markers',
                             name=f"{parties_venn[0]} und {parties_venn[2]}",
                             marker=dict(color=p1p3_answer, size=10), text=p1p3_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p2p3[:, 0], y=points_p2p3[:, 1], mode='markers',
                             name=f"{parties_venn[1]} und {parties_venn[2]}",
                             marker=dict(color=p2p3_answer, size=10), text=p2p3_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p1[:, 0], y=points_p1[:, 1], mode='markers',
                             name=f"{parties_venn[0]}",
                             marker=dict(color=p1_answer, size=10), text=p1_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p2[:, 0], y=points_p2[:, 1], mode='markers',
                             name=f"{parties_venn[1]}",
                             marker=dict(color=p2_answer, size=10), text=p2_text, showlegend=False))
    fig.add_trace(go.Scatter(x=points_p3[:, 0], y=points_p3[:, 1], mode='markers',
                             name=f"{parties_venn[2]}",
                             marker=dict(color=p3_answer, size=10), text=p3_text, showlegend=False))

    st.plotly_chart(fig, height=900)
st.subheader("Line Graph der Übereinstimmung")
party = st.selectbox("Wähle eine Partei aus", parties)
sorted_df = agreement_df.sort_values(by=[party], ascending=False) * 100
pos_df = sorted_df[sorted_df[party] > 0]
neg_df = sorted_df[sorted_df[party] < 0]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pos_df.index,
    y=pos_df[party],
    mode='lines+markers',
    marker=dict(color='green', size=10),
    line=dict(color='green'),
    fill='tozeroy',
    fillcolor='RGBA(0, 255, 0, 0.25)',
    opacity=0.5
))
fig.add_trace(go.Scatter(
    x=neg_df.index,
    y=neg_df[party],
    mode='lines+markers',
    marker=dict(color='red', size=10),
    line=dict(color='red'),
    fill='tozeroy',
    fillcolor='RGBA(255, 0, 0, 0.25)',
))

fig.update_layout(
    title=f"Übereinstimmung von {party} mit anderen Parteien",
    xaxis_title="Partei",
    yaxis_title="Übereinstimmung",
    height=900
)
st.plotly_chart(fig, height=900)

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
                 title="Dimensionality Reduction und Clustering", color_discrete_sequence=px.colors.qualitative.Set1,
                 height=900)
fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig)
