"""NBA Predictor test UI: pick 2 teams -> player predictions + totals + win prob.
Run: streamlit run app.py   (then open http://localhost:8501)
"""
import numpy as np
import streamlit as st
import pandas as pd
from nba_api.stats.static import teams as static_teams
from src.predict import predict_matchup
from src.progress import get_stages

st.set_page_config(page_title="NBA Predictor", layout="wide")
st.title("NBA Predictor — test it out")

with st.expander("Pipeline status", expanded=False):
    for label, frac, detail in get_stages():
        c1, c2 = st.columns([3, 1])
        c1.progress(min(max(frac, 0.0), 1.0), text=label)
        c2.caption(detail)

all_teams = sorted(static_teams.get_teams(), key=lambda t: t["abbreviation"])
abbrevs = [t["abbreviation"] for t in all_teams]
names = {t["abbreviation"]: f"{t['abbreviation']} — {t['full_name']}" for t in all_teams}

c1, c2, c3 = st.columns(3)
with c1:
    team_a = st.selectbox("Team A", abbrevs, index=abbrevs.index("LAL") if "LAL" in abbrevs else 0,
                          format_func=lambda x: names[x])
with c2:
    default_b = abbrevs.index("BOS") if "BOS" in abbrevs else 1
    team_b = st.selectbox("Team B", abbrevs, index=default_b, format_func=lambda x: names[x])
with c3:
    home = st.selectbox("Home team", [team_a, team_b], index=0)

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

if st.button("Predict", type="primary"):
    with st.spinner("Fetching rosters + predicting..."):
        try:
            out = predict_matchup(team_a, team_b, home_team=home)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    st.caption(f"Mode: {'trained model' if out['used_model'] else 'baseline (model not trained yet)'} — {out['date']}")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{out['team_a']} total", out["totals"][out["team_a"]])
    m2.metric(f"{out['team_b']} total", out["totals"][out["team_b"]])
    spread_txt = f"{out['team_a']} -{abs(out['spread'])}" if out["spread"] >= 0 else f"{out['team_b']} -{abs(out['spread'])}"
    m3.metric("Winner", f"{out['winner']} ({out['win_prob'][out['winner']] * 100:.1f}%)", spread_txt)
    st.progress(out["win_prob"][out["team_a"]],
                text=f"P({out['team_a']}) = {out['win_prob'][out['team_a']] * 100:.1f}%")

    with st.expander("Simulated total distribution (10k sims)", expanded=False):
        sim_df = pd.DataFrame(out["sim_totals"])
        bins = np.histogram_bin_edges(
            pd.concat([sim_df[out["team_a"]], sim_df[out["team_b"]]]), bins=30)
        hist = pd.DataFrame({
            out["team_a"]: np.histogram(sim_df[out["team_a"]], bins=bins)[0],
            out["team_b"]: np.histogram(sim_df[out["team_b"]], bins=bins)[0],
        }, index=[f"{bins[i]:.0f}-{bins[i + 1]:.0f}" for i in range(len(bins) - 1)])
        st.bar_chart(hist)

    p1, p2 = st.columns(2)
    for col, t in [(p1, out["team_a"]), (p2, out["team_b"])]:
        with col:
            st.subheader(t)
            df = out["players"][t].head(12).copy()
            df["pred_pts"] = df["pred_pts"].round(1)
            df["sigma"] = df["sigma"].round(1)
            df["base_avg10"] = df["base_avg10"].round(1)
            st.dataframe(df[["player", "pred_pts", "sigma", "base_avg10"]],
                         use_container_width=True)
