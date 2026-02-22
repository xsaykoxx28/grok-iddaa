import streamlit as st
import requests
import pandas as pd
from datetime import date
import numpy as np
import math

def poisson_pmf(k, lam):
    if k < 0 or not isinstance(k, int):
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

st.set_page_config(page_title="Grok İddaa", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

st.title("⚽ Grok İddaa Tahmin - Football-Data.org 🌍")
st.caption("Telefon optimize • Gerçek zamanlı maçlar • Poisson Modeli")

api_key = st.secrets["football_data_key"]

# Hızlı butonlar
col1, col2 = st.columns(2)
with col1:
    if st.button("🔴 Canlı Maçlar", use_container_width=True):
        st.session_state.mode = "live"
        st.rerun()
with col2:
    if st.button("📅 Bugünkü Maçları Getir", use_container_width=True):
        st.session_state.mode = "today"
        st.rerun()

with st.sidebar:
    st.header("🎛️ Filtreler")
    selected_date = st.date_input("Tarih Seç", value=date.today())

if st.sidebar.button("🌍 Seçili Tarihten Maç Çek", use_container_width=True):
    st.session_state.mode = "date"
    st.session_state.date = selected_date
    st.rerun()

# Maç çekme
if "mode" in st.session_state:
    mode = st.session_state.mode
    with st.spinner("Maçlar Football-Data.org'dan yükleniyor..."):
        if mode == "live":
            url = "https://api.football-data.org/v4/matches?status=LIVE"
        elif mode == "today":
            url = f"https://api.football-data.org/v4/matches?date={date.today().isoformat()}"
        else:
            url = f"https://api.football-data.org/v4/matches?date={st.session_state.date.isoformat()}"
        
        headers = {"X-Auth-Token": api_key}
        r = requests.get(url, headers=headers)
        
        if r.status_code != 200:
            st.error("API Hatası → Key'i doğru girdiğinden emin ol")
            st.stop()
        
        data = r.json().get("matches", [])
        
        if not data:
            st.warning("❌ Şu anda maç yok. 🔴 Canlı veya 📅 Bugünkü butonunu dene.")
            st.stop()
        
        matches_list = []
        for m in data:
            matches_list.append({
                "fixture_id": m["id"],
                "lig": m["competition"]["name"],
                "country": m["competition"].get("area", {}).get("name", "International"),
                "saat": m["utcDate"][11:16],
                "ev": m["homeTeam"]["name"],
                "deplasman": m["awayTeam"]["name"],
                "durum": m["status"]
            })
        
        df = pd.DataFrame(matches_list)
        df = df.sort_values(["country", "lig", "saat"])
        
        st.session_state.df = df
        st.success(f"✅ {len(df)} maç yüklendi!")

# Veri varsa göster (DÜZELTİLMİŞ KISIM)
if "df" in st.session_state:
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    with col1:
        countries = sorted(df["country"].unique())
        default_countries = []
        for pref in ["Turkey", "Türkiye", "Turkiye"]:
            if pref in countries:
                default_countries = [pref]
                break
        if not default_countries and countries:
            default_countries = [countries[0]]
        
        secili_country = st.multiselect("Ülke", countries, default=default_countries)
    
    with col2:
        filtered = df[df["country"].isin(secili_country)] if secili_country else df
        secili_lig = st.multiselect("Lig", sorted(filtered["lig"].unique()), default=filtered["lig"].unique()[:10])
    
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

    if "selected" in st.session_state:
        fid = st.session_state.selected
        st.divider()
        st.subheader(f"🔮 Maç ID: {fid}")
        
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

st.caption("© Grok 2026 • Football-Data.org API • Sorumlu oyna!")
