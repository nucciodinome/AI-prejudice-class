import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations
from collections import Counter
import spacy
import os

# ------------------------------------------------------
# LOAD ITALIAN SPACY MODEL
# ------------------------------------------------------
nlp = spacy.load("it_core_news_sm")

# ------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Ultra-Advanced Text Analysis Suite", layout="wide")
st.title("📊 Advanced Text Analysis Suite – Gender-Focused Version")

# ------------------------------------------------------
# AUTOLOAD EXCEL
# ------------------------------------------------------
DATA_FILE = "llm_perception_study.xlsx"

if not os.path.exists(DATA_FILE):
    st.error(f"❌ Il file '{DATA_FILE}' non è presente.")
    st.stop()

df = pd.read_excel(DATA_FILE)

# ------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------
st.sidebar.header("Settings")

# Auto-select text column
if "document" in df.columns:
    default_index = df.columns.get_loc("document")
else:
    default_index = 0

text_col = st.sidebar.selectbox("Select text column", df.columns, index=default_index)
model_col = st.sidebar.selectbox("Select model column", df.columns)

gender_col = "type"
if gender_col not in df.columns:
    st.error("La colonna 'type' (gender) non esiste.")
    st.stop()

    # --------------------------------------------------
    # STOPWORDS
    # --------------------------------------------------
default_stop = {
        "a", "abbia", "abbiamo", "abbiano", "abbiate", "ad", "adesso", "affatto", "agl", "agli", "ai", "al", "alcuna", "alcune", "alcuni", 
        "alcuno", "all", "alla", "alle", "allo", "altri", "altrimenti", "altro", "altra", "altre", "anche", "ancora", "anzi", "assai", 
        "attualmente", "avendo", "avete", "aveva", "avevano", "avevo", "avrai", "avranno", "avrebbe", "avrebbero", "avrei", "avremo", 
        "avrete", "avrete", "avrò", "avere", "aver", "avete", "avuto", "basta", "ben", "benchè", "bene", "bensì", "breve", "c", "casa", 
        "c’è", "c’erano", "c’era", "certo", "certa", "certe", "certi", "che", "chi", "chicchessia", "chiunque", "ci", "ciascuna", 
        "ciascuno", "ciò", "cioè", "circa", "ciro", "coi", "col", "come", "cominci", "comincia", "cominciando", "comunque", "con", "contro", 
        "cosa", "cose", "cui", "da", "dai", "dal", "dall", "dalla", "dalle", "dallo", "davanti", "de", "degli", "dei", "del", "della", "delle", 
        "dello", "dentro", "di", "dice", "dicendo", "dietro", "dire", "dirò", "dirai", "diranno", "direbbe", "direbbero", "dite", "diventa", 
        "diventare", "divenire", "dopo", "dov", "dove", "dovrei", "dovremmo", "dovrete", "dovrò", "dovrà", "dovranno", "dovrebbero", "dovrebbe", "due", 
        "dunque", "durante", "e", "ebbene", "ecc", "ecco", "ed", "egli", "ella", "entrambi", "entro", "erano", "era", "eravate", "eravamo", "erei", "ero", 
        "essendo", "essere", "essi", "esse", "est", "etc", "facendo", "facile", "fai", "fanno", "fare", "farò", "farai", "faranno", "farebbe", "farebbero", 
        "farei", "farò", "fatto", "felice", "fin", "finalmente", "finchè", "fino", "forse", "fra", "fu", "fui", "fummo", "furono", "già", "giacché", "giacche", 
        "giusto", "gli", "gliene", "glieli", "glielo", "gliela", "gliele", "grazie", "guarda", "ha", "hai", "hanno", "ho", "i", "il", "in", "infatti", "inoltre", 
        "insieme", "intanto", "intorno", "invece", "io", "là", "la", "lei", "le", "li", "lo", "loro", "là", "lì", "ma", "macché", "magari", "mai", "malgrado", 
        "mancanza", "me", "mediante", "mentre", "meno", "mese", "mi", "mia", "mie", "miei", "mio", "molta", "molte", "molti", "molto", "ne", "negl", "negli", 
        "nei", "nel", "nell", "nella", "nelle", "nello", "nemmeno", "neppure", "nessun", "nessuna", "nessuno", "niente", "no", "noi", "non", "nostra", "nostre", 
        "nostri", "nostro", "nullo", "nulla", "o", "od", "oggi", "ognuno", "ogni", "oltre", "oppure", "ora", "osserva", "ossia", "ovvero", "ove", "per", "perché", 
        "perciò", "perfino", "persino", "più", "poiché", "poi", "poiché", "possa", "possiamo", "possono", "posso", "potere", "potrà", "potrei", "potremmo", 
        "potrebbero", "pp", "prendendo", "prima", "primo", "proprio", "può", "pure", "qual", "quale", "quali", "qualcosa", "qualcuno", "qualche", "qualsiasi", 
        "quando", "quanta", "quante", "quanti", "quanto", "quasi", "questa", "queste", "questi", "questo", "qui", "quindi", "quinto", "se", "sé", "senza", "sembra", 
        "sempre", "senza", "si", "sia", "siamo", "siano", "siate", "siete", "sino", "sinistra", "solito", "solo", "soltanto", "sono", "sopra", "sotto", "spesso", 
        "sta", "stai", "stando", "stanno", "stare", "starà", "starei", "stavamo", "stava", "stavi", "stavo", "stessi", "stesso", "stessa", "stesse", "su", "sua", 
        "sue", "sui", "sul", "sull", "sulla", "sulle", "sullo", "suo", "suoi", "svariati", "svariato", "tal", "tale", "tali", "tanta", "tante", "tanti", "tanto", 
        "te", "tempo", "ti", "tra", "tranne", "tre", "troppo", "tu", "tua", "tue", "tuo", "tuoi", "tutta", "tutte", "tutti", "tutto", "u", "uguali", "un", "una", 
        "un’altra", "un’altro", "uno", "uno", "unica", "unici", "unico", "uniche", "uno", "uomo", "va", "vai", "varie", "vario", "verso", "vi", "via", "voi", 
        "volta", "volte", "vostra", "vostre", "vostri", "vostro", "vuole", "vuoi"

        # punctuation artifacts
        ".", ",", ";", "!", "?", "_", "(", ")", "[", "]", "{", "}", "'", "\""
}

sociological_stop = {
        "people","think","say","study","student","analysis","text","use","like",
        "beijing","washington","dc","world","post","post-decarbonized","decarbonized","&"
}

custom_stop = st.sidebar.text_area("Add custom stopwords (comma-separated)")
custom_stop = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

stopwords_final = sorted(default_stop.union(sociological_stop).union(custom_stop))

# ------------------------------------------------------
# TOPIC MODELING CONTROLS
# ------------------------------------------------------
n_topics = st.sidebar.slider("Number of topic clusters", 3, 20, 6)
min_df = st.sidebar.slider("Min document frequency", 1, 10, 2)
max_df = st.sidebar.slider("Max document frequency", 0.1, 1.0, 0.9)
top_n_words = st.sidebar.slider("Top terms per topic", 5, 30, 10)

# ------------------------------------------------------
# CLEAN & TOKENIZE
# ------------------------------------------------------
def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9àèéìòù]+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t)>2 and t not in stopwords_final]
    return tokens

df = df.dropna(subset=[text_col])
df["_clean_tokens"] = df[text_col].astype(str).apply(clean_tokenize)
docs = df["_clean_tokens"].apply(lambda tok: " ".join(tok)).tolist()

# ------------------------------------------------------
# TF-IDF
# ------------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words=None,
    max_features=6000,
    min_df=min_df,
    max_df=max_df,
    ngram_range=(1,1)
)

X = tfidf.fit_transform(docs)
feature_names = np.array(tfidf.get_feature_names_out())

# ------------------------------------------------------
# NMF
# ------------------------------------------------------
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_
df["topic"] = W.argmax(axis=1)

def top_words(topic_idx, n=top_n_words):
    idx = H[topic_idx].argsort()[-n:][::-1]
    return feature_names[idx], H[topic_idx][idx]

# ------------------------------------------------------
# TABS
# ------------------------------------------------------
tabs = st.tabs([
    "1️⃣ Topic Modeling",
    "2️⃣ STM-style Word Differences + Dual Semantic Nets",
    "3️⃣ Topic Distance Map",
    "4️⃣ Semantic Network",
    "5️⃣ Sentiment (Italiano)",
    "6️⃣ Wordclouds (Gender × Model)"
])

# ======================================================
# TAB 1
# ======================================================
with tabs[0]:

    st.subheader("📌 Extracted Topics")

    topic_rows = []
    for t in range(n_topics):
        words, _ = top_words(t)
        topic_rows.append({"Topic": t, "Top Terms": ", ".join(words)})

    st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)

    st.subheader("Topic Distribution by Gender")
    fig = px.histogram(
        df, x="topic", color=gender_col, barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Topic–Term Heatmap")
    heat = np.vstack([top_words(t, top_n_words)[1] for t in range(n_topics)])
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=heat,
            x=[f"Term {i+1}" for i in range(top_n_words)],
            y=[f"Topic {i}" for i in range(n_topics)],
            colorscale="Viridis"
        )
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ======================================================
# TAB 2 — WORD DIFFERENCE + DUAL SEMANTIC NETS
# ======================================================
with tabs[1]:

    st.header("STM-Style Difference-in-Word-Use Analysis")

    genders = df[gender_col].unique()
    if len(genders) != 2:
        st.warning("Il dataset deve avere 2 categorie gender.")
        st.stop()

    g1, g2 = genders

    tok1 = df[df[gender_col] == g1]["_clean_tokens"].sum()
    tok2 = df[df[gender_col] == g2]["_clean_tokens"].sum()

    w1 = Counter(tok1)
    w2 = Counter(tok2)
    vocab = list(set(w1.keys()).union(w2.keys()))

    diff = [{"word": w, "diff": w1[w] - w2[w]} for w in vocab]
    diff_df = pd.DataFrame(diff)
    diff_df["abs"] = diff_df["diff"].abs()
    diff_df = diff_df.sort_values("abs", ascending=False).head(40)

    fig = px.bar(
        diff_df,
        x="word",
        y="diff",
        color="diff",
        color_continuous_scale=px.colors.diverging.RdBu[::-1],
        title=f"{g1} vs {g2}: Word Usage Differences"
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------
    # DUAL SEMANTIC NETWORK (with percentile filtering)
    # ------------------------------------------------------
    st.subheader("🔵🔴 Dual Semantic Networks by Gender")

    def build_semantic_network(tokens, max_terms=30):
        # top words
        top_terms = Counter(tokens).most_common(max_terms)
        words = [w for w,_ in top_terms]

        # co-occurrence
        G = nx.Graph()
        for w1, w2 in combinations(words, 2):
            G.add_edge(w1, w2, weight=1)

        # edge weights vector
        weights = [d["weight"] for _,_,d in G.edges(data=True)]
        if len(weights) == 0:
            return G

        # percentile threshold
        thresh = np.percentile(weights, 75)  # keep top 25%
        rem = [(u,v) for u,v,d in G.edges(data=True) if d["weight"] < thresh]
        G.remove_edges_from(rem)
        G.remove_nodes_from(list(nx.isolates(G)))

        return G

    col1, col2 = st.columns(2)

    # ---- NETWORK 1 ----
    with col1:
        st.markdown(f"### 🔴 Semantic Network – {g1}")
        G1 = build_semantic_network(tok1)
        pos = nx.spring_layout(G1, seed=1, k=0.7)

        fig = go.Figure()

        # edges
        for u,v,d in G1.edges(data=True):
            x0,y0 = pos[u]; x1,y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0,x1], y=[y0,y1],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.55)", width=1.4),
                hoverinfo="none"
            ))

        # nodes
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G1.nodes()],
            y=[pos[n][1] for n in G1.nodes()],
            mode="markers+text",
            text=list(G1.nodes()),
            textposition="top center",
            marker=dict(
                size=12,
                color="rgba(255,60,60,0.75)",
                line=dict(color="rgba(0,0,0,0.25)", width=1)
            )
        ))
        fig.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- NETWORK 2 ----
    with col2:
        st.markdown(f"### 🔵 Semantic Network – {g2}")
        G2 = build_semantic_network(tok2)
        pos = nx.spring_layout(G2, seed=1, k=0.7)

        fig = go.Figure()

        # edges
        for u,v,d in G2.edges(data=True):
            x0,y0 = pos[u]; x1,y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0,x1], y=[y0,y1],
                mode="lines",
                line=dict(color="rgba(70,70,70,0.55)", width=1.4),
                hoverinfo="none"
            ))

        # nodes
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G2.nodes()],
            y=[pos[n][1] for n in G2.nodes()],
            mode="markers+text",
            text=list(G2.nodes()),
            textposition="top center",
            marker=dict(
                size=12,
                color="rgba(60,100,255,0.75)",
                line=dict(color="rgba(0,0,0,0.25)", width=1)
            )
        ))
        fig.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3 — TOPIC DISTANCE MAP
# ======================================================
with tabs[2]:

    st.header("Topic Distance Map (MDS)")
    dist = pairwise_distances(H)
    coords = MDS(n_components=2, random_state=42, dissimilarity="precomputed").fit_transform(dist)

    fig = px.scatter(
        x=coords[:,0], y=coords[:,1],
        text=[f"T{i}" for i in range(n_topics)],
        color=list(range(n_topics)),
        color_continuous_scale="Viridis"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 4 — ORIGINAL SEMANTIC NETWORK
# ======================================================
with tabs[3]:
    st.header("Semantic Network (Weighted Strength Centrality)")

    min_w = st.slider("Minimum co-occurrence weight", 1, 6, 1)
    min_cent = st.slider("Minimum node strength (centrality filter)", 0.0, 1.0, 0.0, 0.01)

    G = nx.Graph()
    for t in range(n_topics):
        words,_ = top_words(t, 12)
        for w1,w2 in combinations(words, 2):
            if G.has_edge(w1,w2):
                G[w1][w2]["weight"] += 1
            else:
                G.add_edge(w1,w2, weight=1)

    rem = [(u,v) for u,v,d in G.edges(data=True) if d["weight"] < min_w]
    G.remove_edges_from(rem)
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) == 0:
        st.warning("No nodes remain.")
        st.stop()

    strength = {n: sum(d["weight"] for _,_,d in G.edges(n, data=True)) for n in G.nodes()}
    cent_vals = np.array(list(strength.values()))
    min_v, max_v = cent_vals.min(), cent_vals.max()
    cent_norm = (cent_vals - min_v) / (max_v - min_v + 1e-12)

    keep_nodes = [node for node,c in zip(G.nodes(),cent_norm) if c>=min_cent]
    G = G.subgraph(keep_nodes).copy()

    strength = {n: sum(d["weight"] for _,_,d in G.edges(n, data=True)) for n in G.nodes()}
    cent_vals = np.array(list(strength.values()))
    min_v, max_v = cent_vals.min(), cent_vals.max()
    cent_norm = (cent_vals - min_v) / (max_v - min_v + 1e-12)

    colors = px.colors.diverging.Portland
    idx = (cent_norm*(len(colors)-1)).astype(int)
    node_colors = [colors[i] for i in idx]
    node_sizes = [8+80*c for c in cent_norm]

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=100)

    edge_x, edge_y = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]
        edge_y += [y0,y1,None]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.3, color="dimgray"), opacity=0.55
    ))

    fig_net.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(size=node_sizes, color=node_colors, opacity=0.97)
    ))

    fig_net.update_layout(height=830, dragmode="pan")
    st.plotly_chart(fig_net, use_container_width=True)

# ======================================================
# TAB 5 — SENTIMENT (ITALIANO SPA-CY)
# ======================================================
with tabs[4]:

    st.header("Sentiment Analysis (Italiano – spaCy)")

    sentiment_lexicon = {
        "buono": 1.5, "ottimo": 2.0, "positivo": 2.2, "favorevole": 1.8,
        "eccellente": 2.5, "felice": 2.0, "contento": 1.8,
        "cattivo": -1.8, "negativo": -2.0, "sfavorevole": -1.8,
        "terribile": -2.5, "orrendo": -2.8, "pessimo": -2.2
    }

    def sentiment_it_spacy(text):
        doc = nlp(text)
        score = 0
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in sentiment_lexicon:
                score += sentiment_lexicon[lemma]
        if score > 0.5:
            return "Positivo", score
        elif score < -0.5:
            return "Negativo", score
        else:
            return "Neutro", score

    df["sentiment_label"], df["sentiment_score"] = zip(*df[text_col].astype(str).apply(sentiment_it_spacy))

    color_map = {"Positivo": "#4DA6FF", "Negativo": "#FF6666", "Neutro": "#BFBFBF"}

    st.subheader("Distribuzione generale")
    sent_counts = df["sentiment_label"].value_counts(normalize=True).reset_index()
    sent_counts.columns = ["sentiment_label","percent"]
    sent_counts["percent_display"] = (sent_counts["percent"]*100).round(1)

    fig = px.bar(
        sent_counts, x="sentiment_label", y="percent",
        color="sentiment_label", color_discrete_map=color_map,
        text="percent_display"
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment per Gender")
    fig = px.histogram(
        df,
        x=gender_col,
        color="sentiment_label",
        barnorm="percent",
        color_discrete_map=color_map
    )
    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Percentuale",
        legend_title="Sentiment"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment per Model")
    fig = px.histogram(
        df,
        x=model_col,
        color="sentiment_label",
        barnorm="percent",
        color_discrete_map=color_map
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Percentuale",
        legend_title="Sentiment"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 6 — WORDCLOUDS
# ======================================================
with tabs[5]:

    st.header("Wordclouds (Gender × Model)")

    groups = df.groupby([gender_col, model_col])
    for (gen, mod), subset in groups:
        st.subheader(f"{gen} – {mod}")
        text = " ".join(subset[text_col].astype(str))
        if len(text) < 20:
            st.write("Not enough text.")
            continue
        wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
        st.image(wc.to_array(), use_container_width=True)
