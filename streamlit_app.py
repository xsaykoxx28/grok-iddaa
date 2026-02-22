import streamlit as st
import requests
import pandas as pd
from datetime import date
import numpy as np
import math

# Manuel Poisson PMF (scipy olmadan)
def poisson_pmf(k, lam):
    if k < 0 or not isinstance(k, int):
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

# MOBİL OPTİMİZASYON
st.set_page_config(
    page_title="Grok İddaa",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("⚽ Grok İddaa Tahmin - TÜM LİGLER 🌍")
st.caption("Telefon için optimize • Gerçek zamanlı • Poisson Modeli")

# API KEY
api_key = st.secrets["api_key"]

# Sidebar Filtreler
with st.sidebar:
    st.header("🎛️ Filtreler")
    selected_date = st.date_input("Maç Tarihi", value=date.today())

if st.sidebar.button("🌍 Tüm Liglerden Maçları Çek", use_container_width=True):
    with st.spinner("Dünya maçları yükleniyor..."):
        url = f"https://v3.football.api-sports.io/fixtures?date={selected_date.isoformat()}"
        headers = {"x-apisports-key": api_key}
        r = requests.get(url, headers=headers)
        
        if r.status_code != 200:
            st.error("API Hatası (kota dolduysa yarın dene)")
            st.stop()
        
        fixtures = r.json().get("response", [])
        data = [{
            "fixture_id": f["fixture"]["id"],
            "lig": f["league"]["name"],
            "ülke": f["league"].get("country", "Uluslararası"),
            "saat": f["fixture"]["date"][11:16],
            "ev": f["teams"]["home"]["name"],
            "deplasman": f["teams"]["away"]["name"],
            "durum": f["fixture"]["status"]["short"]
        } for f in fixtures]
        
        df = pd.DataFrame(data)
        popular = ["Süper Lig", "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Champions League"]
        df["popüler"] = df["lig"].isin(popular)
        df = df.sort_values(["popüler", "ülke", "lig", "saat"], ascending=[False, True, True, True])
        
        st.session_state.df = df
        st.success(f"✅ {len(df)} maç yüklendi!")

# Veri varsa göster
if "df" in st.session_state:
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    with col1:
        secili_ulke = st.multiselect("Ülke", sorted(df["ülke"].unique()), default=["Türkiye"])
    with col2:
        filtered = df[df["ülke"].isin(secili_ulke)] if secili_ulke else df
        secili_lig = st.multiselect("Lig", sorted(filtered["lig"].unique()), default=filtered["lig"].unique()[:8])
    
    if secili_lig:
        filtered = filtered[filtered["lig"].isin(secili_lig)]
    
    st.dataframe(filtered[["saat", "lig", "ev", "deplasman", "durum"]], use_container_width=True, hide_index=True)
    
    st.subheader("🏟️ Liglere Göre Maçlar")
    for lig in sorted(filtered["lig"].unique()):
        lig_df = filtered[filtered["lig"] == lig]
        with st.expander(f"{lig} ({len(lig_df)} maç)", expanded=False):
            for _, row in lig_df.iterrows():
                cols = st.columns([4, 2, 1])
                with cols[0]:
                    st.write(f"**{row['saat']}** {row['ev']} - {row['deplasman']}")
                with cols[1]:
                    st.code(row['fixture_id'], language=None)
                with cols[2]:
                    if st.button("🎯 Tahmin", key=f"btn_{row['fixture_id']}", use_container_width=True):
                        st.session_state.selected = row['fixture_id']
                        st.rerun()

    # Tahmin ekranı
    if "selected" in st.session_state:
        fid = st.session_state.selected
        st.divider()
        st.subheader(f"🔮 Maç ID: {fid}")
        
        p_resp = requests.get(f"https://v3.football.api-sports.io/predictions?fixture={fid}", headers={"x-apisports-key": api_key})
        if p_resp.json().get("response"):
            p = p_resp.json()["response"][0]["predictions"]
            c1, c2 = st.columns(2)
            c1.metric("Maç Sonucu", p["winner"]["name"] or "Beraberlik")
            c2.metric("Öneri", p["advice"])
        
        st.subheader("📊 Grok Poisson Tahmini")
        home_l, away_l = 1.6, 1.3
        max_g = 8
        home_probs = np.array([poisson_pmf(i, home_l) for i in range(max_g)])
        away_probs = np.array([poisson_pmf(i, away_l) for i in range(max_g)])
        probs = np.outer(home_probs, away_probs)
        
        ml = np.unravel_index(probs.argmax(), probs.shape)
        st.success(f"**En olası skor: {ml[0]} - {ml[1]}**")
        
        st.write("**İY/MS Top 5**")
        ht_l = 0.45
        ht_home_probs = np.array([poisson_pmf(i, home_l*ht_l) for i in range(4)])
        ht_away_probs = np.array([poisson_pmf(i, away_l*ht_l) for i in range(4)])
        ht_probs = np.outer(ht_home_probs, ht_away_probs)
        
        top5 = sorted([(f"{h}-{a} / {ml[0]}-{ml[1]}", ht_probs[h,a] * probs[ml[0], ml[1]]) 
                       for h in range(4) for a in range(4)], key=lambda x: x[1], reverse=True)[:5]
        for combo, p in top5:
            st.write(f"**{combo}** → %{p*100:.1f}")
        
        if st.button("Başka maç seç", use_container_width=True):
            del st.session_state.selected
            st.rerun()

st.caption("© Grok 2026 • Mobil için optimize • Sorumlu oyna!")
