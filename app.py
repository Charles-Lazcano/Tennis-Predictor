# app.py — Tennis Match Predictor (Streamlit)

import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Small helpers ──────────────────────────────────────────────────────────────
def safe_index(options: list, value, default: int = 0) -> int:
    """Return a plain int index of value in options, or default if not found."""
    try:
        return int(options.index(value))
    except Exception:
        return int(default)

# ── Robust file loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_players(path: str = "players.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("player") or df.columns[0]
    rank_col = cols.get("rank") or cols.get("p1_rank")  # optional

    out = df[[name_col]].copy()
    out.columns = ["name"]

    if rank_col is None:
        out["rank"] = 1000
    else:
        out["rank"] = pd.to_numeric(df[rank_col], errors="coerce").fillna(1000).astype(int)

    out = (
        out.drop_duplicates(subset=["name"], keep="first")
        .sort_values(["rank", "name"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return out

@st.cache_data(show_spinner=False)
def load_feature_columns(path: str = "feature_columns.csv") -> list[str]:
    """Read exported feature list reliably (handles single-column CSV w/o header)."""
    col = (
        pd.read_csv(path, header=None, dtype=str)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .dropna()
    )
    col = col[col != "0"]
    col = col[col != ""]
    return col.tolist()

@st.cache_resource(show_spinner=False)
def load_model(path: str = "model_logreg.joblib"):
    return joblib.load(path)

# ── Sets models (optional) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sets_model_bo3(path: str = "model_sets_bo3.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_sets_model_bo5(path: str = "model_sets_bo5.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

# ── Total games model (optional) ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_games_model(path: str = "model_games.joblib"):
    """Loads {'model': pipeline, 'mae': float} produced by scripts/train_total_games.py."""
    try:
        return joblib.load(path)
    except Exception:
        return None

# ── Data & models ─────────────────────────────────────────────────────────────
try:
    players_df = load_players("players.csv")
    feature_cols = load_feature_columns("feature_columns.csv")
    clf = load_model("model_logreg.joblib")
except Exception as e:
    st.error(f"Failed to load required files: {e}")
    st.stop()

sets_bo3 = load_sets_model_bo3()
sets_bo5 = load_sets_model_bo5()
games_bundle = load_games_model()  # dict or None

RANK_BY_NAME = dict(zip(players_df["name"], players_df["rank"]))
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
SURF_TO_COL = {s: f"surf_{s}" for s in SURFACES}

# ── Feature engineering & prediction ──────────────────────────────────────────
def feature_row(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    r1 = int(RANK_BY_NAME.get(p1, 1000))
    r2 = int(RANK_BY_NAME.get(p2, 1000))
    row = {"rank_diff": r1 - r2, "best_of": int(best_of)}
    row[SURF_TO_COL.get(surface, f"surf_{surface}")] = 1
    X = pd.DataFrame([row]).fillna(0)

    unknown = set(X.columns) - set(feature_cols)
    if unknown:
        X = X.drop(columns=list(unknown), errors="ignore")
    X = X.reindex(columns=feature_cols, fill_value=0)

    for c in ("rank_diff", "best_of"):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

def predict_proba(p1: str, p2: str, surface: str, best_of: int) -> float:
    X = feature_row(p1, p2, surface, best_of)
    return float(clf.predict_proba(X)[0, 1])

# Raw features for sets / games models (which have their own encoders inside)
def feature_row_raw(p1: str, p2: str, surface: str, best_of: int) -> pd.DataFrame:
    r1 = int(RANK_BY_NAME.get(p1, 1000))
    r2 = int(RANK_BY_NAME.get(p2, 1000))
    return pd.DataFrame([{"rank_diff": r1 - r2, "best_of": int(best_of), "surface": surface}])

def predict_sets_probs(p1: str, p2: str, surface: str, best_of: int):
    model = sets_bo3 if best_of == 3 else sets_bo5
    if model is None:
        return None
    Xraw = feature_row_raw(p1, p2, surface, best_of)
    proba = model.predict_proba(Xraw)[0]
    classes = [int(c) for c in model.classes_]
    return dict(zip(classes, proba))

def show_sets_ui(sets_probs: dict[int, float], best_of: int):
    st.subheader("🧮 Sets prediction")
    if best_of == 3:
        p2 = float(sets_probs.get(2, 0.0))
        p3 = float(sets_probs.get(3, 0.0))
        st.markdown(f"**Best-of-3** · **{p2:.1%}** → 2 sets • **{p3:.1%}** → 3 sets")
        st.progress(int(round(p2 * 100)), text="2 sets")
        st.progress(int(round(p3 * 100)), text="3 sets")
    else:
        p3 = float(sets_probs.get(3, 0.0))
        p4 = float(sets_probs.get(4, 0.0))
        p5 = float(sets_probs.get(5, 0.0))
        st.markdown(f"**Best-of-5** · **{p3:.1%}** → 3 sets • **{p4:.1%}** → 4 sets • **{p5:.1%}** → 5 sets")
        st.progress(int(round(p3 * 100)), text="3 sets")
        st.progress(int(round(p4 * 100)), text="4 sets")
        st.progress(int(round(p5 * 100)), text="5 sets")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎾 Tennis Match Predictor")
st.caption("Pick two players, choose the surface and match format, then click **Predict**.")

# Sticky defaults
if "p1" not in st.session_state:
    st.session_state.p1 = players_df["name"].iloc[0]
if "p2" not in st.session_state:
    st.session_state.p2 = players_df["name"].iloc[min(1, len(players_df) - 1)]
if "surface" not in st.session_state:
    st.session_state.surface = "Hard"
if "bestof" not in st.session_state:
    st.session_state.bestof = 3

names = players_df["name"].tolist()

# Player pickers
col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Player 1", options=names,
                      index=safe_index(names, st.session_state.p1, 0), key="p1_select")
with col2:
    p2 = st.selectbox("Player 2", options=names,
                      index=safe_index(names, st.session_state.p2, 1 if len(names) > 1 else 0), key="p2_select")

st.session_state.p1 = p1
st.session_state.p2 = p2

# Surface + match format
surface = st.selectbox("Surface", SURFACES,
                       index=safe_index(SURFACES, st.session_state.surface, 0), key="surface_select")
st.session_state.surface = surface

best_of = st.radio("Match format", options=[3, 5], horizontal=True,
                   index=0 if st.session_state.bestof == 3 else 1, key="bestof_radio")
st.session_state.bestof = best_of

# Action row
act1, act2 = st.columns([1, 1])
with act1:
    if st.button("Swap players"):
        st.session_state.p1, st.session_state.p2 = st.session_state.p2, st.session_state.p1
        st.rerun()
with act2:
    do_predict = st.button("Predict", type="primary")

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if do_predict:
    if st.session_state.p1 == st.session_state.p2:
        st.warning("Please choose two different players.")
    else:
        try:
            proba = predict_proba(st.session_state.p1, st.session_state.p2,
                                  st.session_state.surface, st.session_state.bestof)
            st.progress(int(round(proba * 100)))
            winner = st.session_state.p1 if proba >= 0.5 else st.session_state.p2
            st.success(
                f"**{winner}** is favored • Win probability for **{st.session_state.p1}** "
                f"vs **{st.session_state.p2}** on **{st.session_state.surface}** "
                f"(best-of-{st.session_state.bestof}): **{proba:.1%}**"
            )

            # Sets prediction
            sets_probs = predict_sets_probs(st.session_state.p1, st.session_state.p2,
                                            st.session_state.surface, st.session_state.bestof)
            if sets_probs is None:
                st.info("Sets model not found. Run: `python scripts/train_sets_games.py`.")
            else:
                show_sets_ui(sets_probs, st.session_state.bestof)

            # Total games prediction
            if games_bundle is None:
                st.info("Total games model not found. Run: `python scripts/train_total_games.py`.")
            else:
                Xraw = feature_row_raw(st.session_state.p1, st.session_state.p2,
                                       st.session_state.surface, st.session_state.bestof)
                g_model = games_bundle["model"]
                mae = float(games_bundle.get("mae", 3.5))
                g_pred = float(g_model.predict(Xraw)[0])
                lo = max(18.0, g_pred - mae)
                hi = g_pred + mae

                st.subheader("🎯 Total games prediction")
                st.write(f"Expected total games: **{g_pred:.1f}**  (± **{mae:.1f}**, ≈ {lo:.1f}–{hi:.1f})")
                st.progress(int(round(min(g_pred, 50) / 50.0 * 100)))
                st.caption("Progress bar scaled to 50 games (rough upper bound for Bo5).")

                with st.expander("Over/Under helper"):
                    default_line = int(round(g_pred))
                    line = st.slider("Set your O/U line", min_value=18, max_value=50, value=default_line, step=1)
                    if g_pred > line:
                        st.success(f"Leans **Over {line}** (estimate {g_pred:.1f}).")
                    elif g_pred < line:
                        st.warning(f"Leans **Under {line}** (estimate {g_pred:.1f}).")
                    else:
                        st.info(f"Right on the line ~{line}.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Info ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Why might I see a scikit-learn version warning?"):
    st.write(
        """
        You trained the model with one version of scikit-learn and are loading it with another.
        That can still work (we suppress the warning here), but for maximum safety you should
        retrain your model in the same environment you deploy.
        """
    )

st.caption(
    "Winner model: Logistic Regression • Inputs: rank difference, surface one-hot, best-of.  "
    "Sets model: logistic regression (separate Bo3/Bo5).  "
    "Total games: gradient boosting regressor with ±MAE band."
)
