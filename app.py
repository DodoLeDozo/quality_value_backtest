import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px

# import ton pipeline
import pipeline_panel_to_backtest as pipe

st.set_page_config(page_title="Backtest Quality-Value", layout="wide")

# ============================
# PAGE D'ACCUEIL
# ============================
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("tbs_logo.png"):
            st.image("tbs_logo.png", width=200)
        else:
            st.markdown("<h4 style='text-align:center;'>[Logo TBS]</h4>", unsafe_allow_html=True)

        st.markdown("<h1 style='text-align:center;'>Projet de Thèse – Stratégie Quality-Value</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#E63946;'>MS Expert en Banque et Ingénierie Financière — "
                    "TBS Education</h3>", unsafe_allow_html=True)

        st.markdown(
            """
            <p style='text-align:center; max-width:700px; margin:auto; margin-top:20px; font-size:18px; line-height:1.6;'>
                Ce projet de recherche vise à construire et backtester une stratégie d’investissement
                basée sur des critères fondamentaux stricts (croissance, rentabilité, discipline financière),
                combinant l’approche <b>Quality</b> et <b>Value</b>.
            </p>
            """,
            unsafe_allow_html=True
        )

        # Ajout d’espace avant le bouton
        st.markdown("<br><br>", unsafe_allow_html=True)

        if st.button("🚀 Entrer dans le Dashboard", use_container_width=True):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# ============================
# DASHBOARD
# ============================
st.title("📊 Backtest Moat + P/FCF < 20")
st.caption("Panel → Alertes → Backtest → Analyse Risque → Top & Flop → Positions → Rapport PDF")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres")

    api_key = st.text_input("FMP API Key", value=os.getenv("FMP_API_KEY", ""), type="password")

    pfcf_th = st.number_input("Seuil P/FCF", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

    st.subheader("Fenêtre de prix (alertes)")
    prices_from = st.date_input("De", value=pd.to_datetime("2019-01-01")).strftime("%Y-%m-%d")
    prices_to = st.date_input("À", value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

    st.subheader("Fenêtre backtest")
    start_bt = st.date_input("Début", value=pd.to_datetime("2020-04-01")).strftime("%Y-%m-%d")
    end_bt = st.date_input("Fin", value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

    # Sélecteur benchmark dynamique
    benchmarks = {
        "S&P 500": "^GSPC",
        "CAC 40": "^FCHI",
        "MSCI World (ETF URTH)": "URTH"
    }
    bench_choice = st.selectbox("Benchmark", options=list(benchmarks.keys()), index=0)
    bench = benchmarks[bench_choice]

    workers = st.slider("Parallélisme (FMP)", 1, 16, 8)

st.markdown("### 1) Charger le panel")
uploaded = st.file_uploader("CSV panel", type=["csv"])
panel_path_text = st.text_input("Ou chemin local", value="resultats_panel_T12020_T12025_by_ticker.csv")
run_btn = st.button("▶️ Lancer le pipeline")

# --- Onglets ---
backtest_tab, topflop_tab, pos_tab, weight_tab, pdf_tab, alerts_tab = st.tabs(
    ["📈 Backtest", "🏆 Top & Flop", "📂 Positions finales",
     "📑 Pondérations trimestrielles", "📄 Rapport PDF", "🔔 Alertes"]
)

def save_upload_to_tmp(uploaded_file) -> str:
    bytes_io = io.BytesIO(uploaded_file.read())
    dest = "uploaded_panel.csv"
    with open(dest, "wb") as f:
        f.write(bytes_io.getbuffer())
    return dest

if run_btn:
    if uploaded is not None:
        panel_path = save_upload_to_tmp(uploaded)
    else:
        panel_path = panel_path_text.strip()
    if not panel_path or not os.path.exists(panel_path):
        st.error("Panel introuvable.")
        st.stop()

    with st.spinner("Calcul en cours…"):
        alerts, summary, df_equity, df_trades, df_pos = pipe.run_pipeline(
            panel_csv_path=panel_path,
            api_key=api_key,
            pfcf_th=pfcf_th,
            prices_from=prices_from,
            prices_to=prices_to,
            start_bt=start_bt,
            end_bt=end_bt,
            bench_sym=bench,
            bench_fallback="SPY",
            max_workers=workers,
        )

    # --- ALERTES ---
    with alerts_tab:
        if alerts.empty:
            st.info("Aucune alerte détectée.")
        else:
            st.dataframe(alerts.head(1000))

    # --- BACKTEST ---
    with backtest_tab:
        st.subheader(f"📈 Performance du portefeuille vs {bench_choice}")

        if not df_equity.empty:
            eq = df_equity.copy()
            eq["equity_norm"] = 100.0 * (eq["nav"] / eq["nav"].iloc[0])
            eq["bench_norm"] = 100.0 * (eq["bench"] / eq["bench"].iloc[0])

            chart_df = eq.set_index("date")[["equity_norm", "bench_norm"]]
            chart_df = chart_df.rename(columns={
                "equity_norm": "Portefeuille",
                "bench_norm": bench_choice
            })

            fig = px.line(chart_df, labels={"value": "Performance (base 100)", "date": "Date"})
            fig.update_traces(line=dict(width=2))
            fig.data[0].line.color = "blue"
            fig.data[1].line.color = "orange"
            st.plotly_chart(fig, use_container_width=True)

            perf_port = (chart_df["Portefeuille"].iloc[-1] - 100)
            perf_bench = (chart_df[bench_choice].iloc[-1] - 100)

            st.markdown(f"**Performance totale Portefeuille : {perf_port:.2f}% vs {bench_choice} : {perf_bench:.2f}%**")

            # ✅ Ajout du CAGR (corrigé)
            eq["date"] = pd.to_datetime(eq["date"])  # conversion explicite
            eq = eq.set_index("date")

            years = (eq.index[-1] - eq.index[0]).days / 365
            if years > 0:
                cagr = (eq['equity_norm'].iloc[-1] / 100) ** (1 / years) - 1
                st.markdown(f"**CAGR (annualisé) : {cagr:.2%}**")
            else:
                st.markdown("**CAGR (annualisé) : N/A**")

            # === Metrics robustes ===
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if years > 0:
                    cagr_metric = (eq['equity_norm'].iloc[-1] / 100) ** (1 / years) - 1
                    st.metric("CAGR", f"{cagr_metric:.2%}")
                else:
                    st.metric("CAGR", "N/A")
            with col2:
                vol = eq['equity_norm'].pct_change().std() * (252 ** 0.5)
                st.metric("Volatilité", f"{vol:.2%}")
            with col3:
                if eq['equity_norm'].pct_change().std() > 0:
                    sharpe = (eq['equity_norm'].pct_change().mean() /
                              eq['equity_norm'].pct_change().std()) * (252 ** 0.5)
                    st.metric("Sharpe", f"{sharpe:.2f}")
                else:
                    st.metric("Sharpe", "N/A")
            with col4:
                roll_max = eq['equity_norm'].cummax()
                dd = (eq['equity_norm'] - roll_max) / roll_max
                st.metric("Max Drawdown", f"{dd.min():.2%}")

            st.download_button("⬇️ Télécharger bt_equity_curve.csv",
                               data=df_equity.to_csv(index=False).encode("utf-8"),
                               file_name="bt_equity_curve.csv",
                               mime="text/csv")
        else:
            st.info("Pas de courbe Equity (peut-être aucune alerte).")


    # --- TOP & FLOP ---
    with topflop_tab:
        if df_trades is not None and not df_trades.empty:
            perf_per_ticker = df_trades.groupby("ticker").apply(
                lambda x: (x["price"].iloc[-1] / x["price"].iloc[0] - 1) * 100
            ).sort_values(ascending=False)

            top_5 = perf_per_ticker.head(5)
            flop_5 = perf_per_ticker.tail(5)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top 5")
                st.table(top_5.apply(lambda x: f"{x:.2f}%"))
            with col2:
                st.subheader("💀 Flop 5")
                st.table(flop_5.apply(lambda x: f"{x:.2f}%"))
        else:
            st.info("Pas de données de trades pour calculer les Top & Flop.")

    # --- POSITIONS FINALES ---
    with pos_tab:
        st.subheader("📂 Positions finales")
        if df_pos is not None and not df_pos.empty:
            st.dataframe(df_pos)
            st.download_button(
                "⬇️ Télécharger bt_positions_lastday.csv",
                data=df_pos.to_csv(index=False).encode("utf-8"),
                file_name="bt_positions_lastday.csv",
                mime="text/csv"
            )
        else:
            st.info("Pas de position finale (peut-être aucune alerte ou aucun trade exécuté).")

    # --- PONDÉRATIONS TRIMESTRIELLES ---
    with weight_tab:
        if df_trades is not None and not df_trades.empty:
            df_trades["Quarter"] = pd.to_datetime(df_trades["date"]).dt.to_period("Q")
            quarterly_positions = df_trades.groupby(["Quarter", "ticker"]).agg({
                "shares": "sum",
                "price": "mean",
                "value": "sum"
            }).reset_index()

            for q, group in quarterly_positions.groupby("Quarter"):
                st.subheader(f"Pondérations {q}")
                total_value = group["value"].sum()
                group["pond_%"] = group["value"] / total_value * 100
                st.dataframe(group[["ticker", "pond_%"]].set_index("ticker").round(2))
        else:
            st.info("Pas de données disponibles pour afficher les pondérations trimestrielles.")

    # --- RAPPORT PDF ---
    with pdf_tab:
        if os.path.exists("rapport_fonds.pdf"):
            with open("rapport_fonds.pdf", "rb") as f:
                st.download_button("⬇️ Télécharger le Rapport PDF", f, file_name="rapport_fonds.pdf")
        else:
            st.info("Rapport PDF non encore généré.")
