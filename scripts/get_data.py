# scripts/get_data.py
import os, re, glob, subprocess, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
JEFF = ROOT / "tennis_atp"
TML  = ROOT / "TML-Database"

def safe_clone(url: str, dest: Path):
    """Clone repo only if it doesn't already exist."""
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)

def csvs_in(folder: Path):
    return sorted(folder.glob("atp_matches_*.csv"))

def is_valid_score(s: str) -> bool:
    if not isinstance(s, str): return False
    s_up = s.upper()
    for bad in ("W/O", "RET", "DEF", "ABN", "UNP", "NULL"):
        if bad in s_up: return False
    return bool(re.search(r"\d-\d", s))

set_token_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)(?:\(\d+\))?\s*$")
def parse_set_games(token: str):
    m = set_token_re.match(token)
    if not m: return None
    return int(m.group(1)) + int(m.group(2))

def split_sets(score: str):
    if not isinstance(score, str): return []
    return [t for t in score.strip().split() if re.search(r"\d-\d", t)]

def compute_sets_games(score: str):
    toks = split_sets(score)
    games = [g for t in toks if (g := parse_set_games(t)) is not None]
    return len(games), (sum(games) if games else None)

def main():
    print("🔹 Cloning repositories if needed...")
    safe_clone("https://github.com/JeffSackmann/tennis_atp.git", JEFF)
    safe_clone("https://github.com/Tennismylife/TML-Database.git", TML)

    print("🔹 Loading yearly match files...")
    paths = [*csvs_in(JEFF), *csvs_in(TML)]
    want_cols = [
        "tourney_id","tourney_name","tourney_date","surface","round",
        "winner_name","loser_name","winner_rank","loser_rank",
        "best_of","score","minutes"
    ]

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
            for c in want_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[want_cols].copy()

            for c in ("winner_rank","loser_rank","minutes","best_of"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            td = pd.to_numeric(df["tourney_date"], errors="coerce")
            df["tourney_date"] = pd.to_datetime(td, format="%Y%m%d", errors="coerce")
            df["year"] = df["tourney_date"].dt.year

            df = df[df["score"].apply(is_valid_score)]
            sg = df["score"].apply(lambda s: pd.Series(compute_sets_games(s),
                                                      index=["num_sets","total_games"]))
            df = pd.concat([df, sg], axis=1)
            df = df[(df["num_sets"]>=2) & df["total_games"].notna()]
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Skipped {p}: {e}")

    print("🔹 Combining all years...")
    data = pd.concat(dfs, ignore_index=True)
    key = ["tourney_id","tourney_date","round","winner_name","loser_name","score"]
    data = data.sort_values("tourney_date").drop_duplicates(subset=key, keep="last")

    out_full = DATA / "combined_matches_1968_2025.csv"
    out_slim = DATA / "combined_slim_for_model.csv"
    slim_cols = [
        "tourney_date","surface","round","best_of",
        "winner_name","loser_name","winner_rank","loser_rank",
        "score","num_sets","total_games","minutes","year"
    ]
    data.to_csv(out_full, index=False)
    data[slim_cols].to_csv(out_slim, index=False)
    print(f"✅ Saved:\n - {out_full}\n - {out_slim}")

if __name__ == "__main__":
    main()
