"""NBA Predictor — pick any matchup, check today's slate, grade results, manage the model.
Run: streamlit run app.py   (then open http://localhost:8501)
"""
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from nba_api.stats.static import teams as static_teams
from src.predict import predict_matchup
from src.progress import get_stages
from src.today import predict_all_games, summary_table, get_todays_games
from src.grade import grade as grade_slate
from src.config import APP_DIR, now

LOCAL = Path("data/raw/player_game_logs_all.parquet").exists()

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: #F6F6F4;}
.block-container {padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1100px;}
h1 {font-size: 2.6rem !important; font-weight: 750; letter-spacing: -0.03em; line-height: 1.1; margin: 0;}
h2 {font-size: 1.35rem !important; font-weight: 650; letter-spacing: -0.015em; margin: 0.25rem 0 0.1rem;}
h3 {font-size: 1.1rem !important; font-weight: 650;}
.sub {color: #475467; font-size: 1.02rem; margin: 0.45rem 0 0; max-width: 620px;}
.sec-desc {color: #475467; font-size: 0.95rem; margin: 0 0 1rem;}

.hero {display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap;
  padding: 8px 0 22px;}
.statrow {display: flex; gap: 28px; flex-wrap: wrap; margin: 0 0 24px; padding: 16px 20px;
  background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04);}
.stat {display: flex; flex-direction: column; gap: 2px; min-width: 120px;}
.stat .v {font-size: 1.3rem; font-weight: 700; color: #101828; font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;}
.stat .k {font-size: 0.8rem; color: #475467;}

.pill {display: inline-flex; align-items: center; padding: 4px 11px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 600; white-space: nowrap;}
.pill.ok {background: #ECFDF3; color: #067647;}
.pill.run {background: #FFFAEB; color: #B54708;}
.pill.bad {background: #FEF3F2; color: #B42318;}
.pill.live {background: #FEF3F2; color: #B42318;}
.pill.live::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: #D92D20;
  margin-right: 7px; animation: pulse 1.6s ease-in-out infinite;}
.pill.muted {background: #F2F4F7; color: #344054;}
@keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.35;}}

.game {border: 1px solid #E4E7EC; border-radius: 16px; background: #FFFFFF; overflow: hidden; margin-bottom: 10px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 10px 28px -16px rgba(16,24,40,0.18);}
.scoreline {display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch;}
.side {display: flex; align-items: center; gap: 14px; padding: 20px 22px;}
.side.away {justify-content: flex-start;}
.side.home {justify-content: flex-end; text-align: right; flex-direction: row;}
.side img {height: 52px; width: auto;}
.ident .abbr {font-weight: 750; font-size: 1.15rem; color: #101828; letter-spacing: -0.01em;}
.ident .name {font-size: 0.84rem; color: #475467;}
.ident .model {font-size: 0.84rem; color: #344054; margin-top: 4px; font-variant-numeric: tabular-nums;}
.ident .model b {color: #101828; font-size: 1rem;}
.center {display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
  padding: 20px 18px; min-width: 150px; border-left: 1px solid #F2F4F7; border-right: 1px solid #F2F4F7;}
.score {font-size: 1.9rem; font-weight: 750; color: #101828; font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em; line-height: 1;}
.vs {font-size: 0.95rem; font-weight: 600; color: #667085; letter-spacing: 0.02em;}
.meta-row {display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 12px 22px; background: #FAFAF9; border-top: 1px solid #F2F4F7; font-size: 0.9rem; color: #344054;}
.pick-chip {display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px; border-radius: 999px;
  background: #C2410C; color: #FFFFFF; font-weight: 650; font-size: 0.9rem;}
.inj {color: #B42318; font-size: 0.84rem;}

.empty {padding: 26px 26px; border: 1px dashed #D0D5DD; border-radius: 16px; background: #FFFFFF;}
.empty p {margin: 0; color: #475467;}
.empty ul {margin: 12px 0 0; padding-left: 18px; color: #344054; line-height: 1.7;}

.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #E4E7EC;}
.stTabs [data-baseweb="tab"] {padding: 10px 14px; font-weight: 600; color: #475467;}
.stTabs [aria-selected="true"] {color: #101828;}
.stTabs [data-baseweb="tab-highlight"] {background: #C2410C; height: 3px;}
.stButton > button {border-radius: 999px; font-weight: 600; padding: 0.4rem 1.1rem;}
.stButton > button[kind="primary"] {background: #C2410C; border-color: #C2410C;}
.stButton > button[kind="primary"]:hover {background: #9A3412; border-color: #9A3412;}
[data-testid="stDataFrame"] {border: 1px solid #E4E7EC; border-radius: 14px; overflow: hidden; background: #FFFFFF;}
[data-testid="stExpander"] {border: 1px solid #E4E7EC; border-radius: 14px; background: #FFFFFF;}
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px; padding: 14px 18px;}
.foot {color: #667085; font-size: 0.85rem; margin-top: 28px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO_OVERRIDES = {"NOP": "no", "UTA": "utah"}
TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#B4975A", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#8A8D8F", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#E31837",
}
ALL_TEAMS = sorted(static_teams.get_teams(), key=lambda t: t["abbreviation"])
ABBREVS = [t["abbreviation"] for t in ALL_TEAMS]
TEAM_NAMES = {t["abbreviation"]: t["full_name"] for t in ALL_TEAMS}


def logo(abbr, px=44):
    slug = LOGO_OVERRIDES.get(abbr, abbr.lower())
    return (f"<img src='https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png' "
            f"alt='{abbr} logo' height='{px}'>")


def team_block(abbr, side, model_total=None, win=False):
    color = TEAM_COLORS.get(abbr, "#667085")
    edge = "90deg" if side == "away" else "270deg"
    wash = f"background: linear-gradient({edge}, color-mix(in srgb, {color} 14%, #FFFFFF) 0%, #FFFFFF 75%);"
    model = ""
    if model_total is not None:
        model = f"<div class='model'>Model <b>{model_total:.1f}</b></div>"
    ident = (f"<div class='ident'><div class='abbr'>{abbr}</div>"
             f"<div class='name'>{TEAM_NAMES.get(abbr, '')}</div>{model}</div>")
    inner = f"{logo(abbr)}{ident}" if side == "away" else f"{ident}{logo(abbr)}"
    return f"<div class='side {side}' style=\"{wash}\">{inner}</div>"


def game_card(away, home, totals, winner, p_winner, center_html, pill_html, injuries_html=""):
    """One HTML block per game. Keep each part on one line: indented lines inside
    st.markdown are parsed as code blocks."""
    pick = f"<span class='pick-chip'>{winner} · {p_winner * 100:.0f}%</span>"
    inj = f"<span class='inj'>{injuries_html}</span>" if injuries_html else ""
    return (f"<div class='game'>"
            f"<div class='scoreline'>"
            f"{team_block(away, 'away', totals[away], winner == away)}"
            f"<div class='center'>{center_html}{pill_html}</div>"
            f"{team_block(home, 'home', totals[home], winner == home)}"
            f"</div>"
            f"<div class='meta-row'><span>Projected pick</span>{pick}{inj}</div>"
            f"</div>")


def state_pill(g):
    state = g.get("live_state", "Scheduled")
    if state == "Final":
        return "<span class='pill muted'>Final</span>"
    if state == "Scheduled" or g.get("live_away") is None:
        return f"<span class='pill muted'>{g.get('status', '')}</span>"
    q = f"Q{g['period']} " if g.get("period") else ""
    return f"<span class='pill live'>Live · {q}{g.get('clock', '')}</span>"


def center_for(g):
    if g.get("live_away") is None or g.get("live_state") == "Scheduled":
        return "<div class='vs'>vs</div>"
    return f"<div class='score'>{g['live_away']} – {g['live_home']}</div>"


def injury_text(p):
    parts = []
    for t in (p["team_a"], p["team_b"]):
        if p["out"][t]:
            parts.append(f"{t} out: {', '.join(p['out'][t])}")
    return " · ".join(parts)


@st.cache_data(ttl=300)
def data_through():
    try:
        d = pd.read_parquet(APP_DIR / "player_recent.parquet", columns=["GAME_DATE"])["GAME_DATE"].max()
        return pd.to_datetime(d).strftime("%b %d, %Y")
    except Exception:
        return "unknown"


def model_meta():
    try:
        return json.loads(Path("models/metrics.json").read_text())
    except Exception:
        return {}


def retrain_state():
    try:
        return json.loads(Path("data/processed/retrain_status.json").read_text()).get("state")
    except Exception:
        return None


def section(title, desc=""):
    st.markdown(f"<h2>{title}</h2>" + (f"<p class='sec-desc'>{desc}</p>" if desc else ""),
                unsafe_allow_html=True)


# ---------- hero + status (auto-refresh) ----------
@st.fragment(run_every=15)
def hero_status():
    m = model_meta()
    state = retrain_state()
    if state == "running":
        pill = "<span class='pill run'>Updating model</span>"
    elif state == "failed":
        pill = "<span class='pill bad'>Last update failed</span>"
    else:
        pill = "<span class='pill ok'>Model up to date</span>"
    n = m.get("n_train")
    games = f"{n:,}" if isinstance(n, int) else "—"
    st.markdown(
        f"<div class='hero'><div><h1>NBA Predictor</h1>"
        f"<p class='sub'>Player points, team totals and win picks for every game, "
        f"modeled from the box scores back to 1970.</p></div>{pill}</div>"
        f"<div class='statrow'>"
        f"<div class='stat'><span class='v'>{m.get('player_pts_mae', '—')} pts</span>"
        f"<span class='k'>average player error on unseen games</span></div>"
        f"<div class='stat'><span class='v'>{games}</span>"
        f"<span class='k'>player games in training</span></div>"
        f"<div class='stat'><span class='v'>{data_through()}</span>"
        f"<span class='k'>latest game in the data</span></div>"
        f"</div>",
        unsafe_allow_html=True)


hero_status()

tab_today, tab_match, tab_results, tab_model = st.tabs(
    ["Today", "Matchup", "Results", "Model"])

# ---------- Today ----------
with tab_today:
    section(f"Today's games — {now().strftime('%A, %b %d')}", "Every game on today's schedule with the model's totals, pick, and injuries.")
    if st.button("Predict today's slate", type="primary"):
        with st.status("Predicting every game…", expanded=True) as status:
            def cb(i, n, g):
                status.update(label=f"[{i + 1}/{n}] {g['away']} @ {g['home']}")
            try:
                st.session_state["slate"] = predict_all_games(progress_cb=cb)
            except Exception as e:
                st.error(f"Could not load today's schedule: {e}")
            else:
                status.update(label="Slate ready", state="complete")

    slate = st.session_state.get("slate")
    if not slate:
        st.markdown(
            "<div class='empty'><p>No slate loaded yet. Press <b>Predict today's slate</b> to pull "
            "the schedule and run every game.</p><ul>"
            "<li>Model totals and projected winner for each game</li>"
            "<li>Live scores that update while games are on</li>"
            "<li>Players ruled out by the injury report</li></ul></div>",
            unsafe_allow_html=True)
    else:
        df = summary_table(slate)
        keep = [c for c in ["Away", "Home", "Tip", "Live", "Away total", "Home total", "Winner", "P(winner)"]
                if c in df.columns]
        view = df[keep].copy()
        if "P(winner)" in view:
            view["P(winner)"] = view["P(winner)"].map(lambda v: f"{v * 100:.0f}%" if pd.notna(v) else "")
        st.dataframe(view, hide_index=True, width="stretch")
        st.write("")

        for r in slate:
            g = r["game"]
            if r["prediction"] is None:
                st.warning(f"{g['away']} @ {g['home']}: {r['error']}")
                continue
            p = r["prediction"]
            st.markdown(
                game_card(g["away"], g["home"], p["totals"], p["winner"],
                          p["win_prob"][p["winner"]], center_for(g), state_pill(g), injury_text(p)),
                unsafe_allow_html=True)
            with st.expander(f"Player predictions · {g['away']} @ {g['home']}"):
                q1, q2 = st.columns(2)
                for col, t in [(q1, g["away"]), (q2, g["home"])]:
                    with col:
                        st.markdown(f"{logo(t, 28)} **{t}** · model total {p['totals'][t]:.1f}",
                                    unsafe_allow_html=True)
                        d = p["players"][t].head(10)[["player", "pred_pts", "status"]].copy()
                        st.dataframe(
                            d, hide_index=True, width="stretch",
                            column_config={
                                "player": st.column_config.TextColumn("Player"),
                                "pred_pts": st.column_config.NumberColumn("Predicted pts", format="%.1f"),
                                "status": st.column_config.TextColumn("Status"),
                            })

        if st.button("Refresh live scores"):
            fresh = {(x["away"], x["home"]): x for x in get_todays_games()}
            for r in st.session_state["slate"]:
                g = r["game"]
                f = fresh.get((g["away"], g["home"]))
                if f:
                    g.update({k: f[k] for k in
                              ("live_state", "clock", "period", "live_away", "live_home")})
            st.rerun()

# ---------- Matchup ----------
with tab_match:
    section("Any matchup", "Choose two teams and where the game is played. Add late scratches to re-run the numbers.")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        team_a = st.selectbox("Team A", ABBREVS, index=ABBREVS.index("LAL"), format_func=lambda x: f"{x} · {TEAM_NAMES[x]}")
    with c2:
        team_b = st.selectbox("Team B", ABBREVS, index=ABBREVS.index("BOS"), format_func=lambda x: f"{x} · {TEAM_NAMES[x]}")
    with c3:
        home = st.selectbox("Home team", [team_a, team_b], index=0)

    if team_a == team_b:
        st.info("Pick two different teams.")
    else:
        if st.button("Predict matchup", type="primary"):
            with st.spinner("Loading rosters and injury report…"):
                try:
                    st.session_state["pred"] = predict_matchup(team_a, team_b, home_team=home)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        out = st.session_state.get("pred")
        if out is not None and {out["team_a"], out["team_b"]} == {team_a, team_b}:
            st.markdown(
                game_card(out["team_a"], out["team_b"], out["totals"], out["winner"],
                          out["win_prob"][out["winner"]], "<div class='vs'>vs</div>",
                          f"<span class='pill muted'>Home: {out['home_team']}</span>", injury_text(out)),
                unsafe_allow_html=True)
            st.progress(out["win_prob"][out["team_a"]],
                        text=f"{out['team_a']} win probability {out['win_prob'][out['team_a']] * 100:.1f}%")
            st.caption(f"{'Trained model' if out['used_model'] else 'Baseline (no trained model)'} · "
                       f"{out['date']} · Spread {out['spread']:+.1f} for {out['team_a']}")

            for t in (out["team_a"], out["team_b"]):
                if out["questionable"][t]:
                    st.markdown(f"**{t} questionable** (discounted 25%): {', '.join(out['questionable'][t])}")

            with st.expander("Add late scratches"):
                st.caption("For news the injury feed hasn't picked up yet.")
                s1, s2 = st.columns(2)
                with s1:
                    scratch_a = st.multiselect(f"Scratch from {out['team_a']}",
                                               out["players"][out["team_a"]]["player"].tolist())
                with s2:
                    scratch_b = st.multiselect(f"Scratch from {out['team_b']}",
                                               out["players"][out["team_b"]]["player"].tolist())
                if st.button("Re-predict with scratches"):
                    with st.spinner("Re-predicting…"):
                        st.session_state["pred"] = predict_matchup(
                            team_a, team_b, home_team=home,
                            extra_out={out["team_a"]: scratch_a, out["team_b"]: scratch_b})
                        st.rerun()

            with st.expander("Score distribution (10,000 simulations)"):
                sim_df = pd.DataFrame(out["sim_totals"])
                bins = np.histogram_bin_edges(
                    pd.concat([sim_df[out["team_a"]], sim_df[out["team_b"]]]), bins=30)
                hist = pd.DataFrame({
                    out["team_a"]: np.histogram(sim_df[out["team_a"]], bins=bins)[0],
                    out["team_b"]: np.histogram(sim_df[out["team_b"]], bins=bins)[0],
                }, index=[f"{bins[i]:.0f}–{bins[i + 1]:.0f}" for i in range(len(bins) - 1)])
                st.bar_chart(hist, color=["#C2410C", "#475467"])

            p1, p2 = st.columns(2)
            for col, t in [(p1, out["team_a"]), (p2, out["team_b"])]:
                with col:
                    st.markdown(f"{logo(t, 28)} **{t}** · model total {out['totals'][t]:.1f}",
                                unsafe_allow_html=True)
                    d = out["players"][t].head(12)[["player", "pred_pts", "sigma", "base_avg10", "status"]].copy()
                    st.dataframe(
                        d, hide_index=True, width="stretch",
                        column_config={
                            "player": st.column_config.TextColumn("Player"),
                            "pred_pts": st.column_config.NumberColumn("Predicted pts", format="%.1f"),
                            "sigma": st.column_config.NumberColumn("Uncertainty (±)", format="%.1f"),
                            "base_avg10": st.column_config.NumberColumn("Last 10 avg", format="%.1f"),
                            "status": st.column_config.TextColumn("Status"),
                        })

# ---------- Results ----------
with tab_results:
    section("Predictions vs results",
            "Saved slates are graded against final box scores. Pick a date to see how the model did.")
    r1, r2 = st.columns([2, 1])
    with r1:
        grade_date = st.date_input("Game date", value=(now() - pd.Timedelta(days=1)).date())
    with r2:
        st.write("")
        do_grade = st.button("Grade games", type="primary")
    if do_grade:
        with st.spinner("Fetching box scores…"):
            try:
                res = grade_slate(grade_date.strftime("%Y-%m-%d"))
            except Exception as e:
                st.error(f"Grading failed: {e}")
                res = None
        if res is None:
            st.info("No saved predictions for that date. Predict the slate on the same day and it saves automatically.")
        else:
            if res.get("note"):
                st.caption(res["note"])
            a = res["agg"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Winner calls", a.get("winner_accuracy", "—"))
            m2.metric("Team total error", f"{a.get('team_total_mae', '—')} pts")
            m3.metric("Player error", f"{a.get('player_pts_mae', '—')} pts", f"{a.get('n_players', 0)} players")
            st.dataframe(res["games"], hide_index=True, width="stretch")
            if len(res["players"]):
                with st.expander("Largest player misses"):
                    worst = res["players"].sort_values("Err", ascending=False).head(15)
                    st.dataframe(worst[["Game", "Player", "Pred", "Actual", "Err"]],
                                 hide_index=True, width="stretch")

# ---------- Model ----------
with tab_model:
    m = model_meta()
    section("Model", "How accurate the model is, when it was last trained, and how to refresh it.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Player scoring error", f"{m.get('player_pts_mae', '—')} pts",
              help="Average miss on games the model never trained on.")
    m2.metric("Games trained on", f"{m.get('n_train', 0):,}")
    m3.metric("Last trained", str(m.get("trained_at", "—")).replace("T", " "))

    if LOCAL:
        st.write("")
        section("Update the model", "Pulls the latest games, rebuilds features, and retrains. Takes about 10 minutes. "
                                    "A daily job runs this at 6am.")
        if st.button("Pull latest games and retrain"):
            root = Path(".").resolve()
            subprocess.Popen(["nohup", "bash", str(root / "src/daily_update.sh")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            st.success("Update started. Watch the status at the top of the page.")

        st.write("")
        section("Pipeline status")

        @st.fragment(run_every=15)
        def pipeline():
            for label, frac, detail in get_stages():
                c1, c2 = st.columns([3, 1])
                c1.progress(min(max(frac, 0.0), 1.0), text=label)
                c2.caption(detail)

        pipeline()
    else:
        st.write("")
        st.caption("This hosted copy is refreshed when the repository is updated. Retraining runs on the "
                   "maintainer's machine.")

    st.write("")
    with st.expander("How the predictions work"):
        st.markdown(
            "- **Player scoring:** LightGBM on 194 pre-game features: rolling form, rest, home court, age, "
            "and draft pedigree. Same-game box-score stats are excluded so the model can't see the answer.\n"
            "- **Injuries:** ESPN's report. Players listed Out are removed. Questionable players count at 75%. "
            "Teammates pick up about 35% of the missing scoring.\n"
            "- **Winner:** 10,000 simulations over each player's uncertainty.\n"
            "- **Freshness:** the model is retrained daily. Predictions are graded against actual results.")

st.markdown("<p class='foot'>Data from NBA Stats and ESPN. Predictions are estimates, not betting advice.</p>",
            unsafe_allow_html=True)
