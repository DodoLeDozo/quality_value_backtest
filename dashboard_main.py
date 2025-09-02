import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pipeline_panel_to_backtest as pipe

# Import ton générateur de rapport PDF
import fund_report_strategy as report


# === Caching des résultats pour accélérer ===
@st.cache_data
def run_cached_pipeline(panel_path, api_key, pfcf_th, prices_from, prices_to, start_bt, end_bt, bench, workers):
    return pipe.run_pipeline(
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


def run():
    # --- HEADER ---
    st.title("📊 Backtest Moat + P/FCF < 20")
    st.caption("Panel → Alertes → Backtest (rééquilibrage équipondéré seulement lors des achats)")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        api_key = st.text_input("FMP API Key", value=os.getenv("FMP_API_KEY", ""), type="password")
        pfcf_th = st.number_input("Seuil P/FCF", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

        st.subheader("Fenêtre de prix (alertes)")
        prices_from = st.date_input("De", value=pd.to_datetime("2019-01-01")).strftime("%Y-%m-%d")
        prices_to   = st.date_input("À",  value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

        st.subheader("Fenêtre backtest")
        start_bt = st.date_input("Début", value=pd.to_datetime("2020-04-01")).strftime("%Y-%m-%d")
        end_bt   = st.date_input("Fin",   value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

        bench = st.selectbox("Benchmark", options=["^GSPC", "SPY"], index=0)
        workers = st.slider("Parallélisme (FMP)", 1, 16, 8)

    # --- UPLOAD PANEL ---
    st.markdown("### 1) Charger le panel")
    uploaded = st.file_uploader("CSV panel (colonnes: ticker, 2020Q1, 2020Q2, ... avec 'Oui'/'Non')", type=["csv"])
    panel_path_text = st.text_input("Ou chemin local du panel (optionnel)", value="resultats_panel_T12020_T12025_by_ticker.csv")
    run_btn = st.button("▶️ Lancer le pipeline")

    # --- TABS ---
    tabs = st.tabs([
        "🔔 Alertes",
        "📈 Backtest",
        "📊 Analyse Risque",
        "🏆 Top & Flop",
        "📂 Positions finales",
        "📑 Pondérations trimestrielles",
        "📄 Rapport PDF"
    ])

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
            st.error("❌ Panel introuvable. Charge un CSV ou renseigne un chemin valide.")
            st.stop()

        with st.spinner("⚡ Calcul des alertes et lancement du backtest…"):
            alerts, summary, df_equity, df_trades, df_pos = run_cached_pipeline(
                panel_path, api_key, pfcf_th, prices_from, prices_to, start_bt, end_bt, bench, workers
            )

        # --- ALERTES ---
        with tabs[0]:
            st.subheader("🔔 Alertes P/FCF < seuil")
            if alerts.empty:
                st.info("Aucune alerte détectée.")
            else:
                st.dataframe(alerts.head(1000))
                st.download_button("⬇️ Télécharger alerts_pf_cf_below20.csv",
                                data=alerts.to_csv(index=False).encode("utf-8"),
                                file_name="alerts_pf_cf_below20.csv")

            st.subheader("Résumé des alertes par trimestre")
            if summary is not None and not summary.empty:
                st.dataframe(summary)
            else:
                st.info("Pas de résumé disponible.")

        # --- BACKTEST ---
        with tabs[1]:
            st.subheader("📈 Performance du portefeuille vs S&P 500")
            if not df_equity.empty:
                eq = df_equity.copy()
                eq["equity_norm"] = 100.0 * (eq["nav"] / eq["nav"].iloc[0])
                eq["bench_norm"] = 100.0 * (eq["bench"] / eq["bench"].iloc[0])

                fig = px.line(eq, x="date", y=["equity_norm", "bench_norm"],
                            labels={"value":"Performance (base 100)", "date":"Date"},
                            title="NAV vs S&P 500")
                st.plotly_chart(fig, use_container_width=True)

                # Perf totale affichée
                perf_total = eq["equity_norm"].iloc[-1] - 100
                bench_total = eq["bench_norm"].iloc[-1] - 100
                st.markdown(f"**Performance totale portefeuille : {perf_total:.2f}%** vs **S&P500 : {bench_total:.2f}%**")

                # Metrics
                returns = eq["equity_norm"].pct_change().dropna()
                cagr = (eq["equity_norm"].iloc[-1] / eq["equity_norm"].iloc[0])**(252/len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe = returns.mean()/returns.std() * np.sqrt(252)
                max_dd = ((eq["equity_norm"]/eq["equity_norm"].cummax())-1).min()

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("CAGR", f"{cagr:.2%}")
                with col2: st.metric("Volatilité", f"{volatility:.2%}")
                with col3: st.metric("Sharpe", f"{sharpe:.2f}")
                with col4: st.metric("Max Drawdown", f"{max_dd:.2%}")

                st.download_button("⬇️ Télécharger bt_equity_curve.csv",
                                data=df_equity.to_csv(index=False).encode("utf-8"),
                                file_name="bt_equity_curve.csv")
            else:
                st.info("Pas de courbe Equity (aucune alerte).")

        # --- ANALYSE RISQUE ---
        with tabs[2]:
            st.subheader("📊 Analyse du risque")
            if not df_equity.empty:
                eq = df_equity.copy()
                returns = eq["nav"].pct_change().dropna()

                fig_hist = px.histogram(returns, nbins=30, title="Distribution des rendements",
                                        labels={"value":"Rendement"}, opacity=0.7)
                st.plotly_chart(fig_hist, use_container_width=True)

                eq["cummax"] = eq["nav"].cummax()
                eq["drawdown"] = eq["nav"]/eq["cummax"] - 1
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=eq["date"], y=eq["drawdown"],
                                            fill="tozeroy", mode="lines", line_color="red"))
                fig_dd.update_layout(title="Drawdown", yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.info("Pas de données de risque.")

        # --- TOP & FLOP ---
        with tabs[3]:
            st.subheader("🏆 Top & Flop des titres")
            if df_trades is not None and not df_trades.empty and df_pos is not None and not df_pos.empty:
                perf_per_ticker = df_trades.groupby("ticker").apply(
                    lambda x: (x["price"].iloc[-1] / x["price"].iloc[0] - 1) * 100
                ).sort_values(ascending=False)

                pos_final = df_pos.set_index("ticker")["value"]
                total_val = pos_final.sum()
                weights = pos_final / total_val * 100

                top5 = perf_per_ticker.head(5).to_frame("Perf %")
                top5["Pondération %"] = weights.reindex(top5.index)

                flop5 = perf_per_ticker.tail(5).to_frame("Perf %")
                flop5["Pondération %"] = weights.reindex(flop5.index)

                st.write("**Top 5 performers :**")
                st.dataframe(top5)

                st.write("**Flop 5 performers :**")
                st.dataframe(flop5)
            else:
                st.info("Pas de trades exécutés ou pas de positions finales.")

        # --- POSITIONS FINALES ---
        with tabs[4]:
            st.subheader("📂 Positions finales")
            if df_pos is not None and not df_pos.empty:
                st.dataframe(df_pos)
                st.download_button("⬇️ Télécharger bt_positions_lastday.csv",
                                data=df_pos.to_csv(index=False).encode("utf-8"),
                                file_name="bt_positions_lastday.csv")
            else:
                st.info("Pas de position finale.")

        # --- PONDÉRATIONS TRIMESTRIELLES ---
        with tabs[5]:
            st.subheader("📑 Pondérations trimestrielles")
            if df_trades is not None and not df_trades.empty:
                trades = df_trades.copy()
                trades["date"] = pd.to_datetime(trades["date"])
                trades["Quarter"] = trades["date"].dt.to_period("Q")
                positions_q = trades.groupby(["Quarter", "ticker"]).agg({
                    "shares":"sum", "value":"sum"
                }).reset_index()

                for q, group in positions_q.groupby("Quarter"):
                    st.markdown(f"### {q}")
                    total = group["value"].sum()
                    group["pondération %"] = group["value"] / total * 100
                    st.dataframe(group[["ticker", "shares", "pondération %"]])
            else:
                st.info("Pas de données de positions trimestrielles.")

        # --- RAPPORT PDF ---
        with tabs[6]:
            st.subheader("📄 Rapport PDF")
            if not df_equity.empty and not df_trades.empty and not df_pos.empty:
                if st.button("📝 Générer le rapport PDF"):
                    report.generate_report("bt_equity_curve.csv", "bt_trades.csv", "bt_positions_lastday.csv")
                    with open("rapport_fonds.pdf", "rb") as f:
                        st.download_button("⬇️ Télécharger le rapport PDF",
                                           data=f,
                                           file_name="rapport_fonds.pdf",
                                           mime="application/pdf")
            else:
                st.info("Données insuffisantes pour générer un rapport PDF.")
