# pipeline_panel_to_backtest.py
# Combinaison: génération d'alertes P/FCF < 20 à partir du panel + backtest équi-pondéré (rééquilibrage uniquement lors des achats)

import os
import re
import io
import json
import time
import math
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x

# ====================== CONFIG GLOBALE ======================
API_KEY   = os.getenv("FMP_API_KEY", "U5alSX6tkG83UKjwphIAEJD9NAdL5dk0")
BASE_URL  = "https://financialmodelingprep.com/api/v3"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

# Entrée panel large (tickers en ligne, colonnes YYYYQn -> 'Oui'/'Non')
PANEL_CSV = "resultats_panel_T12020_T12025_by_ticker.csv"

# Seuil alerte P/FCF
PFCF_TH   = 20.0

# Fenêtre de prix pour détecter les alertes et backtester
PRICES_FROM = "2019-01-01"
PRICES_TO   = "2025-04-30"

# Fenêtre de backtest (dates quotidiennes)
START = pd.Timestamp("2020-04-01")
END   = pd.Timestamp("2025-03-31")

# Benchmark
BENCH_TICK     = "^GSPC"   # S&P 500
BENCH_FALLBACK = "SPY"

# Sorties
OUT_ALERTS  = "alerts_pf_cf_below20.csv"
OUT_SUMMARY = "alerts_pf_cf_summary.csv"
OUT_EQUITY  = "bt_equity_curve.csv"
OUT_TRADES  = "bt_trades.csv"
OUT_POS     = "bt_positions_lastday.csv"

# Parallélisation & cache
MAX_WORKERS = 8
CACHE_DIR = ".fmp_cache"
CACHE_TTL_HOURS = 24
GLOBAL_PAUSE_EVERY = 200
GLOBAL_PAUSE_SECS = 1.0

# ====================== Infra réseau + cache ======================
_req_lock = threading.Lock()
_req_count = 0

def _mk_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, _mk_key(url) + ".json")

def _read_cache(url: str) -> Optional[Any]:
    if CACHE_TTL_HOURS <= 0:
        return None
    p = _cache_path(url)
    if not os.path.exists(p):
        return None
    age_h = (time.time() - os.path.getmtime(p)) / 3600.0
    if age_h > CACHE_TTL_HOURS:
        try:
            os.remove(p)
        except Exception:
            pass
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_cache(url: str, data: Any) -> None:
    if CACHE_TTL_HOURS <= 0:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = _cache_path(url)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def get_json(url: str, retries: int = 3, backoff: float = 0.6) -> Optional[Any]:
    cached = _read_cache(url)
    if cached is not None:
        return cached
    global _req_count
    with _req_lock:
        _req_count += 1
        if GLOBAL_PAUSE_EVERY and _req_count % GLOBAL_PAUSE_EVERY == 0:
            time.sleep(GLOBAL_PAUSE_SECS)
    last = None
    for attempt in range(retries + 1):
        try:
            import requests
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("Error Message"):
                last = RuntimeError(data["Error Message"])
            else:
                _write_cache(url, data)
                return data
        except Exception as e:
            last = e
        time.sleep(backoff * (attempt + 1))
    return None

# ====================== Parsing panel & dates ======================
def parse_quarter_colnames(df: pd.DataFrame) -> List[str]:
    """ détecte les colonnes YYYYQ[1-4] """
    qcols = []
    for c in df.columns:
        if re.fullmatch(r"\d{4}Q[1-4]", str(c)):
            qcols.append(c)
    return qcols

def quarter_label_from_date(d: pd.Timestamp) -> str:
    p = d.to_period("Q")
    return f"{p.year}Q{p.quarter}"

def qlabel_to_period_end(qlabel: str) -> pd.Timestamp:
    """ '2020Q1' -> 2020-03-31 """
    year = int(qlabel[:4])
    qnum = int(qlabel[-1])
    month = 3 * qnum
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return end

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} doit contenir la colonne 'ticker'.")
    qcols = parse_quarter_colnames(df)
    if not qcols:
        raise ValueError("Aucune colonne trimestre (YYYYQn) détectée dans le panel.")
    df = df[["ticker"] + qcols].copy()
    for c in qcols:
        df[c] = df[c].astype(str).str.lower().str.strip().map({"oui": True, "non": False})
    # ticker en upper/trim
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df

def is_eligible(panel_row: pd.Series, d: pd.Timestamp) -> bool:
    lab = quarter_label_from_date(d)

    # 🔒 Si on dépasse le dernier trimestre du panel (2025Q1),
    # on considère que l’éligibilité reste "gelée" = True
    if lab > "2025Q1":
        return True

    if lab in panel_row.index:
        val = panel_row[lab]
        return bool(val) if not pd.isna(val) else False
    return False

# ====================== Données FMP utiles aux alertes ======================
def fetch_quarterly_financials(ticker: str, limit: int = 60) -> Dict[str, pd.DataFrame]:
    urls = {
        "income": f"{BASE_URL}/income-statement/{ticker}?period=quarter&limit={limit}&apikey={API_KEY}",
        "cf":     f"{BASE_URL}/cash-flow-statement/{ticker}?period=quarter&limit={limit}&apikey={API_KEY}",
    }
    out = {}
    for k, u in urls.items():
        data = get_json(u)
        if not isinstance(data, list) or len(data) == 0:
            out[k] = pd.DataFrame()
        else:
            df = pd.DataFrame(data)
            if "date" not in df.columns:
                out[k] = pd.DataFrame()
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                out[k] = df
    return out

def build_quarter_panel_fcfps_ttm(ticker: str) -> pd.DataFrame:
    """
    Construit un DF trimestriel avec: date (fin de T), FCF_4Q, shares, FCF per share TTM (fcfps_ttm).
    """
    fin = fetch_quarterly_financials(ticker, limit=60)
    inc, cf = fin["income"], fin["cf"]
    if inc.empty or cf.empty:
        return pd.DataFrame(columns=["date","fcfps_ttm"])

    shares = inc["weightedAverageShsOutDil"].fillna(inc.get("weightedAverageShsOut"))
    inc2 = inc[["date"]].copy()
    inc2["shares"] = pd.to_numeric(shares, errors="coerce")

    cf2 = cf[["date","freeCashFlow"]].copy()
    cf2["freeCashFlow"] = pd.to_numeric(cf2["freeCashFlow"], errors="coerce")

    df = pd.merge(inc2, cf2, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["date","fcfps_ttm"])

    df["fcf_4q"] = df["freeCashFlow"].rolling(window=4, min_periods=4).sum()
    df["fcfps_ttm"] = df.apply(
        lambda r: (r["fcf_4q"] / r["shares"]) if (pd.notna(r["fcf_4q"]) and pd.notna(r["shares"]) and r["shares"] not in (0, None)) else None,
        axis=1
    )
    return df[["date","fcfps_ttm"]]

def fetch_prices_daily_fmp(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"{BASE_URL}/historical-price-full/{ticker}?from={start}&to={end}&apikey={API_KEY}"
    data = get_json(url)
    if not data or "historical" not in data:
        return pd.DataFrame(columns=["date","close"])
    df = pd.DataFrame(data["historical"])
    if df.empty:
        return pd.DataFrame(columns=["date","close"])
    df = df[["date","close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def map_daily_to_quarter_fcfps(daily: pd.DataFrame, q_fcfps: pd.DataFrame) -> pd.DataFrame:
    """ Pour chaque jour, assigne le fcfps_ttm du dernier trimestre connu <= date du jour. """
    if daily.empty or q_fcfps.empty:
        d = daily.copy()
        d["fcfps_ttm"] = None
        return d
    d = daily.sort_values("date").copy()
    q = q_fcfps.dropna(subset=["fcfps_ttm"]).sort_values("date").copy()
    if q.empty:
        d["fcfps_ttm"] = None
        return d
    merged = pd.merge_asof(d, q, on="date", direction="backward")
    return merged

def safe_div(a, b):
    try:
        if b in (None, 0) or (isinstance(b, float) and (math.isinf(b) or math.isnan(b))):
            return None
        return a / b
    except Exception:
        return None

# ====================== Génération d'alertes par ticker ======================
def process_ticker_alerts(ticker: str, eligible_quarters: List[str]) -> pd.DataFrame:
    """
    Retourne toutes les alertes P/FCF < PFCF_TH pour ce ticker, restreintes aux trimestres éligibles (Oui).
    """
    if not eligible_quarters:
        return pd.DataFrame(columns=["date","ticker","price","fcfps_ttm","p_to_fcf","quarter_end"])

    q_fcfps = build_quarter_panel_fcfps_ttm(ticker)
    prices  = fetch_prices_daily_fmp(ticker, PRICES_FROM, PRICES_TO)

    if prices.empty or q_fcfps.empty:
        return pd.DataFrame(columns=["date","ticker","price","fcfps_ttm","p_to_fcf","quarter_end"])

    daily = map_daily_to_quarter_fcfps(prices, q_fcfps)
    daily.rename(columns={"close":"price"}, inplace=True)
    daily["p_to_fcf"] = [safe_div(p, f) for p, f in zip(daily["price"], daily["fcfps_ttm"])]

    alerts_all = []
    q_dates = sorted(q_fcfps["date"].unique().tolist())
    for qlabel in eligible_quarters:
        q_end = qlabel_to_period_end(qlabel)
        prevs = [d for d in q_dates if d < q_end]
        prev_end = prevs[-1] if prevs else None

        if prev_end is None:
            mask = (daily["date"] <= q_end)
        else:
            mask = (daily["date"] > prev_end) & (daily["date"] <= q_end)

        win = daily.loc[mask].copy()
        if win.empty:
            continue
        sub = win[(pd.notna(win["p_to_fcf"])) & (win["p_to_fcf"] < PFCF_TH)].copy()
        if sub.empty:
            continue
        sub["ticker"] = ticker
        sub["quarter_end"] = q_end
        alerts_all.append(sub[["date","ticker","price","fcfps_ttm","p_to_fcf","quarter_end"]])

    if alerts_all:
        out = pd.concat(alerts_all, ignore_index=True).sort_values(["ticker","date"])
        return out
    return pd.DataFrame(columns=["date","ticker","price","fcfps_ttm","p_to_fcf","quarter_end"])

def build_alerts_from_panel(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule les alertes P/FCF<TH pour tous les tickers du panel sur les trimestres éligibles.
    Retourne (df_alerts, df_summary).
    """
    qcols = [c for c in panel.columns if c != "ticker"]
    eligible_map: Dict[str, List[str]] = {}
    for _, row in panel.iterrows():
        t = str(row["ticker"]).strip().upper()
        elig = [c for c in qcols if bool(row[c])]
        if elig:
            eligible_map[t] = elig

    tickers = sorted(set(panel["ticker"].astype(str).str.upper().str.strip().tolist()))
    print(f"🔎 {len(tickers)} tickers dans le panel ; {len(eligible_map)} avec ≥1 trimestre éligible.")

    alerts_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_ticker_alerts, t, eligible_map.get(t, [])): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Alerting"):
            try:
                df = fut.result()
                if not df.empty:
                    alerts_rows.append(df)
            except Exception:
                pass

    if not alerts_rows:
        return pd.DataFrame(columns=["date","ticker","price","fcfps_ttm","p_to_fcf","quarter_end"]), pd.DataFrame(columns=["ticker","quarter_end","nb_alert_days_below_20"])

    df_alerts = pd.concat(alerts_rows, ignore_index=True).sort_values(["ticker","date"])
    summary = (
        df_alerts.groupby(["ticker","quarter_end"])
                 .size().reset_index(name="nb_alert_days_below_20")
                 .sort_values(["ticker","quarter_end"])
    )
    return df_alerts, summary

# ====================== Prix pour le backtest (yfinance) ======================
def fetch_prices_yf(universe: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Retourne:
      - px: DataFrame Adj Close (colonnes = tickers, index = dates)
      - bench: Series 'bench' (Adj Close)
      - universe filtré (tickers conservés)
    """
    if not universe:
        raise ValueError("Univers vide.")

    data = yf.download(universe, start=start, end=end, auto_adjust=False, progress=False, threads=True)

    # Extraire l'Adj Close de manière robuste
    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex):
            if ("Adj Close" in data.columns.get_level_values(0)):
                px = data["Adj Close"].copy()
            else:
                raise RuntimeError("Colonnes 'Adj Close' introuvables (multiindex).")
        else:
            # mono-titre
            if "Adj Close" in data.columns:
                if len(universe) == 1:
                    px = data[["Adj Close"]].copy()
                    px.columns = [universe[0]]
                else:
                    s = data["Adj Close"].squeeze()
                    if isinstance(s, pd.Series):
                        px = s.to_frame(name=universe[0])
                    else:
                        px = data[["Adj Close"]].copy()
                        px.columns = [universe[0]]
            else:
                s = data.squeeze()
                if isinstance(s, pd.Series):
                    px = s.to_frame(name=universe[0])
                else:
                    raise RuntimeError("Structure inattendue des prix (pas 'Adj Close').")
    elif isinstance(data, pd.Series):
        px = data.to_frame(name=universe[0])
    else:
        raise RuntimeError("Format de prix inattendu.")

    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # Drop colonnes vides
    cols_before = list(px.columns)
    px = px.dropna(how="all", axis=1)
    dropped = sorted(set(cols_before) - set(px.columns))
    if dropped:
        print(f"\n⚠️  Tickers sans prix (ignorés): {dropped}\n")
        universe = [t for t in universe if t in px.columns]

    # Benchmark
    b = yf.download(BENCH_TICK, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    bench = None
    if isinstance(b, pd.DataFrame) and "Adj Close" in b.columns:
        bench = b["Adj Close"].astype(float).dropna()
    elif isinstance(b, pd.Series):
        bench = b.astype(float).dropna()

    if bench is None or bench.empty:
        b2 = yf.download(BENCH_FALLBACK, start=start, end=end, auto_adjust=False, progress=False, threads=True)
        if isinstance(b2, pd.DataFrame) and "Adj Close" in b2.columns:
            bench = b2["Adj Close"].astype(float).dropna()
        elif isinstance(b2, pd.Series):
            bench = b2.astype(float).dropna()
        else:
            raise RuntimeError("Impossible d’obtenir le benchmark (ni ^GSPC ni SPY).")
    bench.name = "bench"

    px = px.ffill()
    bench = bench.sort_index().ffill()
    return px, bench, universe

def fetch_prices(universe, start, end):
    import yfinance as yf
    import pandas as pd

    if not universe:
        raise ValueError("Univers vide.")

    data = yf.download(universe, start=start, end=end, auto_adjust=False, progress=False, threads=True)

    # Extraction robuste Adj Close
    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex) and ("Adj Close" in data.columns.get_level_values(0)):
            px = data["Adj Close"].copy()
        elif "Adj Close" in data.columns:
            if len(universe) == 1:
                px = data[["Adj Close"]].copy()
                px.columns = [universe[0]]
            else:
                px = data["Adj Close"].copy()
        else:
            raise RuntimeError("Colonnes 'Adj Close' introuvables.")
    elif isinstance(data, pd.Series):
        px = data.to_frame(name=universe[0])
    else:
        raise RuntimeError("Format inattendu pour les prix.")

    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # 🔑 Tronquer sur la fenêtre demandée
    px = px.loc[(px.index >= start) & (px.index <= end)]

    # Benchmark
    b = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False, threads=True)
    if isinstance(b, pd.DataFrame) and "Adj Close" in b.columns:
        bench = b["Adj Close"].astype(float).dropna()
    elif isinstance(b, pd.Series):
        bench = b.astype(float).dropna()
    else:
        bench = None

    if bench is None or bench.empty:
        b2 = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False, threads=True)
        if isinstance(b2, pd.DataFrame) and "Adj Close" in b2.columns:
            bench = b2["Adj Close"].astype(float).dropna()
        elif isinstance(b2, pd.Series):
            bench = b2.astype(float).dropna()
        else:
            raise RuntimeError("Impossible de récupérer le benchmark.")

    bench = bench.loc[(bench.index >= start) & (bench.index <= end)]
    bench.name = "bench"

    return px.ffill(), bench.ffill(), [t for t in universe if t in px.columns]

# ====================== Backtest ======================
def backtest(alerts: pd.DataFrame, panel: pd.DataFrame):
    # Univers = tickers qui ont au moins un "Oui" OU une alerte
    pan_true_tickers = panel.loc[(panel.drop(columns=["ticker"]).any(axis=1)),"ticker"].tolist()
    alert_tickers = alerts["ticker"].unique().tolist()
    universe_all = sorted(set(pan_true_tickers) | set(alert_tickers))

    px, bench, universe = fetch_prices(universe_all, START, END)
    px = px.loc[(px.index >= START) & (px.index <= END)]
    bench = bench.loc[(bench.index >= START) & (bench.index <= END)]
    dates = px.index

    # NAV benchmark alignée
    bench_aligned = bench.reindex(dates, method="ffill")
    if bench_aligned.dropna().empty:
        raise RuntimeError("Benchmark vide après alignement.")
    bench0 = bench_aligned.dropna().iloc[0]
    bench_nav_series = 100.0 * (bench_aligned / bench0)

    # Panel indexé
    pidx = panel.set_index("ticker")
    # injecter les tickers présents dans px mais absents du panel → jamais éligibles
    for t in universe:
        if t not in pidx.index:
            pidx.loc[t,:] = False
    pidx = pidx.sort_index()

    # Alertes par date
    alerts = alerts[(alerts["date"] >= START) & (alerts["date"] <= END)].copy()
    alerts_by_date: Dict[pd.Timestamp, List[str]] = {}
    for _, r in alerts.iterrows():
        d = pd.to_datetime(r["date"])
        t = r["ticker"]
        if t not in universe:
            continue
        alerts_by_date.setdefault(d, []).append(t)

    # Boucle NAV
    nav = 100.0
    cash = nav
    holdings: Dict[str, float] = {}   # ticker -> shares
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []

    for d in dates:
        # Ventes: si critère non respecté → sortir
        to_sell = []
        for t, sh in list(holdings.items()):
            row = pidx.loc[t]
            if not is_eligible(row, d):
                price = px.at[d, t] if t in px.columns else np.nan
                if pd.notna(price):
                    proceeds = sh * float(price)
                    cash += proceeds
                    trades.append({"date": d, "ticker": t, "side": "SELL", "price": float(price), "shares": sh, "value": proceeds})
                to_sell.append(t)
        for t in to_sell:
            holdings.pop(t, None)

        # Achats: si alerte P/FCF < TH aujourd'hui **et** éligible
        buys_today: List[str] = []
        if d in alerts_by_date:
            for t in alerts_by_date[d]:
                if t in holdings:
                    continue
                if t not in px.columns or pd.isna(px.at[d, t]):
                    continue
                if is_eligible(pidx.loc[t], d):
                    buys_today.append(t)

        # Rééquilibrage EQUIPONDÉRÉ uniquement si achat(s)
        if buys_today:
            after_buy_names = list(holdings.keys()) + buys_today
            port_val = cash + sum(holdings[t] * float(px.at[d, t]) for t in holdings)
            if port_val > 0:
                target_w = 1.0 / len(after_buy_names)
                target_val_each = port_val * target_w

                # 1) Vendre l’excès des anciennes positions
                for t in list(holdings.keys()):
                    price = float(px.at[d, t])
                    cur_val = holdings[t] * price
                    diff = target_val_each - cur_val
                    if diff < 0:
                        sell_val = -diff
                        sell_sh = min(holdings[t], sell_val / price) if price > 0 else 0.0
                        if sell_sh > 0:
                            proceeds = sell_sh * price
                            holdings[t] -= sell_sh
                            cash += proceeds
                            trades.append({"date": d, "ticker": t, "side": "SELL", "price": price, "shares": sell_sh, "value": proceeds})

                # 2) Acheter les nouvelles positions
                port_val = cash + sum(holdings[t] * float(px.at[d, t]) for t in holdings)
                target_val_each = port_val * target_w
                for t in buys_today:
                    price = float(px.at[d, t])
                    buy_val = max(0.0, target_val_each)
                    buy_sh = buy_val / price if price > 0 else 0.0
                    cost = buy_sh * price
                    if buy_sh > 0 and cash >= cost:
                        holdings[t] = holdings.get(t, 0.0) + buy_sh
                        cash -= cost
                        trades.append({"date": d, "ticker": t, "side": "BUY", "price": price, "shares": buy_sh, "value": cost})

                # 3) Optionnel: compléter si reste un peu de cash
                for t in list(holdings.keys()):
                    price = float(px.at[d, t])
                    cur_val = holdings[t] * price
                    diff = target_val_each - cur_val
                    if diff > 0 and cash > 0:
                        add_val = min(diff, cash)
                        add_sh = add_val / price if price > 0 else 0.0
                        if add_sh > 0:
                            cost = add_sh * price
                            holdings[t] += add_sh
                            cash -= cost
                            trades.append({"date": d, "ticker": t, "side": "BUY", "price": price, "shares": add_sh, "value": cost})

        # Valorisation & NAV
        port_val = cash + sum(holdings.get(t,0.0) * float(px.at[d, t]) for t in holdings if t in px.columns)
        equity_curve.append({"date": d, "nav": port_val, "bench": float(bench_nav_series.loc[d])})

    # Exports backtest
    ec = pd.DataFrame(equity_curve)
    ec.to_csv(OUT_EQUITY, index=False)

    tr = pd.DataFrame(trades).sort_values("date") if trades else pd.DataFrame(columns=["date","ticker","side","price","shares","value"])
    tr.to_csv(OUT_TRADES, index=False)

    # Positions finales
    pos_rows = []
    if len(px.index) > 0:
        last_date = px.index[-1]
        for t, sh in holdings.items():
            price = float(px.at[last_date, t]) if t in px.columns and pd.notna(px.at[last_date, t]) else np.nan
            pos_rows.append({"ticker": t, "shares": sh, "last_price": price, "value": (sh * price) if price == price else np.nan})
    pos_df = pd.DataFrame(pos_rows, columns=["ticker","shares","last_price","value"])
    if not pos_df.empty:
        pos_df = pos_df.sort_values("value", ascending=False)
    pos_df.to_csv(OUT_POS, index=False)

    print(f"✅ Exports écrits : {OUT_EQUITY}, {OUT_TRADES}, {OUT_POS}")

# ====================== MAIN ======================
def main():
    print("▶️ Chargement du panel…")
    panel = load_panel(PANEL_CSV)

    # 1) Générer les alertes à partir du panel
    print(f"🛎️  Calcul des alertes P/FCF<{PFCF_TH} (fenêtre prix {PRICES_FROM} → {PRICES_TO})…")
    alerts, summary = build_alerts_from_panel(panel)

    if alerts.empty:
        print("ℹ️  Aucune alerte détectée. (Le backtest sera vide.)")
    else:
        alerts.to_csv(OUT_ALERTS, index=False)
        summary.to_csv(OUT_SUMMARY, index=False)
        print(f"✅ {OUT_ALERTS} ({len(alerts)} lignes)")
        print(f"✅ {OUT_SUMMARY}")

    # 2) Backtest à partir des alertes + panel
    print("🚦 Lancement du backtest…")
    # Par sécurité, ne garder que les alertes dont le trimestre est éligible
    if not alerts.empty:
        pidx = panel.set_index("ticker")
        keep = []
        for _, r in alerts.iterrows():
            t, d = r["ticker"], r["date"]
            if t in pidx.index and is_eligible(pidx.loc[t], pd.to_datetime(d)):
                keep.append(True)
            else:
                keep.append(False)
        alerts = alerts[keep]
    backtest(alerts, panel)

    if CACHE_TTL_HOURS > 0:
        print(f"💾 Cache: {CACHE_DIR} (TTL {CACHE_TTL_HOURS} h)")

def run_pipeline(
    panel_csv_path: str,
    api_key: str,
    pfcf_th: float = 20.0,
    prices_from: str = "2019-01-01",
    prices_to: str   = "2025-04-30",
    start_bt: str    = "2020-04-01",
    end_bt: str      = "2025-04-30",
    bench_sym: str   = "^GSPC",
    bench_fallback: str = "SPY",
    max_workers: int = 8,
):
    """
    Exécute: (1) build alerts depuis panel, (2) backtest.
    Retourne (df_alerts, df_summary, df_equity, df_trades, df_pos).
    Écrit aussi les CSV standards dans le répertoire courant.
    """
    # Patch des paramètres globaux (si tu as gardé les const au début du fichier)
    global API_KEY, PFCF_TH, PRICES_FROM, PRICES_TO, START, END, BENCH_TICK, BENCH_FALLBACK, MAX_WORKERS
    API_KEY        = api_key or API_KEY
    PFCF_TH        = pfcf_th
    PRICES_FROM    = prices_from
    PRICES_TO      = prices_to
    START          = pd.Timestamp(start_bt)
    END            = pd.Timestamp(end_bt)
    BENCH_TICK     = bench_sym
    BENCH_FALLBACK = bench_fallback
    MAX_WORKERS    = max_workers

    panel = load_panel(panel_csv_path)

    alerts, summary = build_alerts_from_panel(panel)
    if not alerts.empty:
        alerts.to_csv(OUT_ALERTS, index=False)
        summary.to_csv(OUT_SUMMARY, index=False)

    # filtrage sécurité
    if not alerts.empty:
        pidx = panel.set_index("ticker")
        keep = []
        for _, r in alerts.iterrows():
            t, d = r["ticker"], r["date"]
            if t in pidx.index and is_eligible(pidx.loc[t], pd.to_datetime(d)):
                keep.append(True)
            else:
                keep.append(False)
        alerts = alerts[keep]

    # backtest
    backtest(alerts, panel)

    # relire les sorties pour renvoyer à l’app
    df_equity  = pd.read_csv(OUT_EQUITY, parse_dates=["date"])
    df_trades  = pd.read_csv(OUT_TRADES, parse_dates=["date"], dayfirst=True)
    df_pos     = pd.read_csv(OUT_POS)
    return alerts, summary, df_equity, df_trades, df_pos

if __name__ == "__main__":
    main()
