"""NBA Predictor — pick any matchup or check today's slate.
Run: streamlit run app.py   (then open http://localhost:8501)
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from nba_api.stats.static import teams as static_teams
from src.predict import predict_matchup
from src.progress import get_stages
from src.today import predict_all_games, summary_table, get_todays_games
from src.grade import grade as grade_slate

TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#444444", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#C89A3C", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#8A8D8F", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

# ---------- header ----------
c_title, c_badges = st.columns([3, 2])
with c_title:
    st.title("🏀 NBA Predictor")
    st.caption("Per-player scores, team totals and win probability — every game, every day.")
with c_badges:
    try:
        m = json.loads(Path("models/metrics.json").read_text())
        st.metric("Model test MAE", f"{m['player_pts_mae']} pts",
                  f"{m['n_train']:,} games trained".replace(",", ","))
    except Exception:
        st.metric("Model", "baseline")


LOGO_OVERRIDES = {"NOP": "no", "UTA": "utah"}


def logo(abbr, px=36):
    slug = LOGO_OVERRIDES.get(abbr, abbr.lower())
    return (f"<img src='https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png' "
            f"height='{px}' style='vertical-align:middle;margin-right:4px'>")


def team_chip(abbr):
    return (f"<span style='color:{TEAM_COLORS.get(abbr, '#333')};"
            f"font-weight:700'>{abbr}</span>")


def live_badge(state):
    if state == "Final":
        return "⬛ Final"
    if state == "Scheduled":
        return ""
    return "🔴 LIVE"


# ---------- today's slate ----------
st.header("Today's games")
if st.button("Predict today's slate", type="primary"):
    with st.status("Predicting every game...", expanded=True) as status:
        def cb(i, n, g):
            status.update(label=f"[{i + 1}/{n}] {g['away']} @ {g['home']}...")
        try:
            st.session_state["slate"] = predict_all_games(progress_cb=cb)
        except Exception as e:
            st.error(f"Failed to load today's schedule: {e}")
            st.stop()
        status.update(label="Slate complete", state="complete")

slate = st.session_state.get("slate")
if slate:
    df = summary_table(slate)
    show = df[[c for c in ["Away", "Home", "Tip", "Live", "Away total",
                           "Home total", "Winner", "P(winner)"] if c in df.columns]]
    st.dataframe(show, width="stretch", hide_index=True)

    for r in slate:
        if r["prediction"] is None:
            st.warning(f"{r['game']['away']} @ {r['game']['home']}: {r['error']}")
            continue
        p, g = r["prediction"], r["game"]
        with st.container(border=True):
            h1, h2, h3 = st.columns([2, 2, 2])
            with h1:
                st.markdown(f"### {logo(g['away'])}{team_chip(g['away'])} @ "
                            f"{logo(g['home'])}{team_chip(g['home'])}",
                            unsafe_allow_html=True)
                st.caption(g["status"])
            with h2:
                if g.get("live_away") is not None and g.get("live_state") != "Scheduled":
                    st.markdown(f"### {logo(g['away'], 28)}{g['live_away']} — "
                                f"{g['live_home']}{logo(g['home'], 28)}",
                                unsafe_allow_html=True)
                    st.caption(f"{live_badge(g.get('live_state', ''))} "
                               f"Q{g.get('period', '')} {g.get('clock', '')}".strip())
                else:
                    st.markdown("### vs")
                    st.caption("Not started")
            with h3:
                st.markdown(f"### {logo(p['winner'])}{team_chip(p['winner'])} "
                            f"{p['win_prob'][p['winner']] * 100:.0f}%",
                            unsafe_allow_html=True)
                st.caption(f"Model: {g['away']} {p['totals'][g['away']]} · "
                           f"{g['home']} {p['totals'][g['home']]}")
            for t in (g["away"], g["home"]):
                if p["out"][t]:
                    st.caption(f"🚑 {t} OUT: {', '.join(p['out'][t])}")
                if p["questionable"][t]:
                    st.caption(f"❓ {t} questionable: {', '.join(p['questionable'][t])}")
            with st.expander("Player predictions"):
                q1, q2 = st.columns(2)
                for col, t in [(q1, g["away"]), (q2, g["home"])]:
                    with col:
                        st.markdown(f"#### {logo(t, 28)}{t} ({p['totals'][t]})",
                                    unsafe_allow_html=True)
                        d = p["players"][t].head(10).copy()
                        d["pred_pts"] = d["pred_pts"].round(1)
                        st.dataframe(d[["player", "pred_pts", "status"]],
                                     width="stretch", hide_index=True)

    if st.button("🔄 Refresh live scores"):
        fresh = {(x["away"], x["home"]): x for x in get_todays_games()}
        for r in st.session_state["slate"]:
            g = r["game"]
            f = fresh.get((g["away"], g["home"]))
            if f:
                g.update({k: f[k] for k in
                          ("live_state", "clock", "period", "live_away", "live_home")})
        st.rerun()

# ---------- results & grading ----------
st.header("Results — predictions vs actuals")
g1, g2 = st.columns([2, 1])
with g1:
    grade_date = st.date_input("Date", value=pd.Timestamp.now() - pd.Timedelta(days=1))
with g2:
    st.write("")
    st.write("")
    do_grade = st.button("Grade games")
if do_grade:
    with st.spinner("Fetching box scores..."):
        try:
            res = grade_slate(grade_date.strftime("%Y-%m-%d"))
        except Exception as e:
            st.error(f"Grading failed: {e}")
            res = None
    if res is None:
        st.warning("No saved predictions for that date — predict the slate first (it auto-saves).")
    else:
        a = res["agg"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Winner calls", a.get("winner_accuracy", "—"))
        m2.metric("Team total MAE", a.get("team_total_mae", "—"))
        m3.metric("Player PTS MAE", f"{a.get('player_pts_mae', '—')} (n={a.get('n_players', 0)})")
        st.dataframe(res["games"], width="stretch", hide_index=True)
        if len(res["players"]):
            worst = res["players"].sort_values("Err", ascending=False).head(15)
            with st.expander("Biggest player misses"):
                st.dataframe(worst[["Game", "Player", "Pred", "Actual", "Err"]],
                             width="stretch", hide_index=True)

# ---------- model freshness ----------
st.header("Model freshness")
try:
    m = json.loads(Path("models/metrics.json").read_text())
    st.caption(f"Model trained {m.get('trained_at', '?')} · test MAE {m.get('player_pts_mae')} · "
               f"{m.get('n_train', 0):,} games".replace(",", ","))
except Exception:
    st.caption("Model metrics unavailable.")
if st.button("🔄 Pull latest games & retrain"):
    import subprocess
    root = Path(".").resolve()
    status_file = root / "data/processed/retrain_status.json"
    status_file.write_text(json.dumps({"state": "running"}))
    cmd = ("cd {r} && python3 -W ignore -m src.update_daily && "
           "python3 -W ignore -m src.build_dataset && "
           "python3 -W ignore -m src.features_rolling && "
           "python3 -W ignore -m src.train && "
           "echo '{{\"state\": \"done\"}}' > {s} || echo '{{\"state\": \"failed\"}}' > {s}").format(
               r=root, s=status_file)
    subprocess.Popen(["nohup", "bash", "-c", cmd], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    st.info("Update started in background (pull → rebuild → retrain, ~10 min). "
            "This page will show fresh metrics when it's done.")
try:
    st.caption(f"Updater: {json.loads(Path('data/processed/retrain_status.json').read_text()).get('state', '?')}")
except Exception:
    pass
st.header("Single matchup")
all_teams = sorted(static_teams.get_teams(), key=lambda t: t["abbreviation"])
abbrevs = [t["abbreviation"] for t in all_teams]
names = {t["abbreviation"]: f"{t['abbreviation']} — {t['full_name']}" for t in all_teams}

c1, c2, c3 = st.columns(3)
with c1:
    team_a = st.selectbox("Team A", abbrevs, index=abbrevs.index("LAL") if "LAL" in abbrevs else 0,
                          format_func=lambda x: names[x])
with c2:
    team_b = st.selectbox("Team B", abbrevs,
                          index=abbrevs.index("BOS") if "BOS" in abbrevs else 1,
                          format_func=lambda x: names[x])
with c3:
    home = st.selectbox("Home team", [team_a, team_b], index=0)

if team_a == team_b:
    st.warning("Pick two different teams.")
    st.stop()

if st.button("Predict matchup", type="primary"):
    with st.spinner("Fetching rosters, injuries + predicting..."):
        try:
            st.session_state["pred"] = predict_matchup(team_a, team_b, home_team=home)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

out = st.session_state.get("pred")
if out is not None and {out["team_a"], out["team_b"]} == {team_a, team_b}:
    st.markdown(f"## {logo(out['team_a'])}{team_chip(out['team_a'])} {out['totals'][out['team_a']]} — "
                f"{out['totals'][out['team_b']]} {logo(out['team_b'])}{team_chip(out['team_b'])}",
                unsafe_allow_html=True)
    st.markdown(f"### ✅ {logo(out['winner'])}{team_chip(out['winner'])} "
                f"({out['win_prob'][out['winner']] * 100:.1f}%)",
                unsafe_allow_html=True)
    st.progress(out["win_prob"][out["team_a"]],
                text=f"P({out['team_a']}) = {out['win_prob'][out['team_a']] * 100:.1f}%")
    st.caption(f"{'Trained model' if out['used_model'] else 'Baseline'} · {out['date']} · "
               f"home: {out['home_team']}")
    for t in [out["team_a"], out["team_b"]]:
        if out["out"][t]:
            st.warning(f"🚑 {t} OUT: {', '.join(out['out'][t])}")
        if out["questionable"][t]:
            st.info(f"❓ {t} questionable (discounted 25%): {', '.join(out['questionable'][t])}")

    with st.expander("Scratch additional players (late news the feed missed)?"):
        opts_a = out["players"][out["team_a"]]["player"].tolist()
        opts_b = out["players"][out["team_b"]]["player"].tolist()
        scratch_a = st.multiselect(f"Scratch from {out['team_a']}", opts_a)
        scratch_b = st.multiselect(f"Scratch from {out['team_b']}", opts_b)
        if st.button("Re-predict with scratches"):
            with st.spinner("Re-predicting..."):
                st.session_state["pred"] = predict_matchup(
                    team_a, team_b, home_team=home,
                    extra_out={out["team_a"]: scratch_a, out["team_b"]: scratch_b})
                st.rerun()

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
            st.subheader(f"{t} ({out['totals'][t]})")
            df = out["players"][t].head(12).copy()
            df["pred_pts"] = df["pred_pts"].round(1)
            df["sigma"] = df["sigma"].round(1)
            df["base_avg10"] = df["base_avg10"].round(1)
            st.dataframe(df[["player", "pred_pts", "sigma", "base_avg10", "status"]],
                         width="stretch", hide_index=True)

# ---------- footer ----------
st.divider()
with st.expander("About this model", expanded=False):
    st.markdown(
        "- **Player scoring:** LightGBM on 194 pre-game features "
        "(rolling form, rest, home court, age, draft pedigree). "
        "Same-game stats are hard-excluded so it can't cheat.\n"
        "- **Injuries:** ESPN report (Out = scratched, "
        "questionable = 25% discount), teammates absorb ~35% of missing scoring.\n"
        "- **Winner:** 10,000 Monte Carlo simulations over per-player uncertainty.\n"
        "- **Freshness:** run `python -m src.update_daily` to fold in new games.")
    with st.expander("Pipeline status", expanded=False):
        for label, frac, detail in get_stages():
            s1, s2 = st.columns([3, 1])
            s1.progress(min(max(frac, 0.0), 1.0), text=label)
            s2.caption(detail)
