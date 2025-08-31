import os
import io
import pandas as pd
import streamlit as st

# import ton pipeline (le fichier précédent doit être dans le même dossier)
import pipeline_panel_to_backtest as pipe

st.set_page_config(page_title="Backtest Moat P/FCF", layout="wide")

st.title("🧪 Backtest Moat + P/FCF < 20")
st.caption("Panel → Alertes → Backtest (rééquilibrage équipondéré seulement lors des achats)")

# --- SIDEBAR: paramètres ---
with st.sidebar:
    st.header("Paramètres")

    api_key = st.text_input("FMP API Key", value=os.getenv("FMP_API_KEY", ""), type="password",
                            help="Clé FinancialModelingPrep (gratuite → fmpcloud.io).")

    pfcf_th = st.number_input("Seuil P/FCF", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

    st.subheader("Fenêtre de prix (alertes)")
    prices_from = st.date_input("De", value=pd.to_datetime("2019-01-01")).strftime("%Y-%m-%d")
    prices_to   = st.date_input("À",  value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

    st.subheader("Fenêtre backtest")
    start_bt = st.date_input("Début", value=pd.to_datetime("2020-04-01")).strftime("%Y-%m-%d")
    end_bt   = st.date_input("Fin",   value=pd.to_datetime("2025-04-30")).strftime("%Y-%m-%d")

    bench = st.selectbox("Benchmark", options=["^GSPC", "SPY"], index=0)
    workers = st.slider("Parallélisme (FMP)", 1, 16, 8)

st.markdown("### 1) Charger le panel")
uploaded = st.file_uploader("CSV panel (colonnes: ticker, 2020Q1, 2020Q2, ... avec 'Oui'/'Non')", type=["csv"])

st.info("Astuce : tu peux aussi pointer vers un fichier déjà présent sur le disque dans la zone ci-dessous.")
panel_path_text = st.text_input("Ou chemin local du panel (optionnel)", value="resultats_panel_T12020_T12025_by_ticker.csv")

run_btn = st.button("▶️ Lancer le pipeline")

# conteneurs de sortie
alerts_tab, backtest_tab = st.tabs(["🔔 Alertes", "📈 Backtest"])

def save_upload_to_tmp(uploaded_file) -> str:
    # sauvegarde l’upload dans un fichier temporaire
    bytes_io = io.BytesIO(uploaded_file.read())
    dest = "uploaded_panel.csv"
    with open(dest, "wb") as f:
        f.write(bytes_io.getbuffer())
    return dest

if run_btn:
    # résolve chemin panel
    if uploaded is not None:
        panel_path = save_upload_to_tmp(uploaded)
    else:
        panel_path = panel_path_text.strip()
    if not panel_path or not os.path.exists(panel_path):
        st.error("Panel introuvable. Charge un CSV ou renseigne un chemin valide.")
        st.stop()

    if not api_key:
        st.warning("Tu n’as pas fourni de clé FMP : si elle n’est pas dans les variables d’environnement, les appels peuvent échouer/ralentir.")

    with st.spinner("Calcul des alertes et lancement du backtest…"):
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

    # --- ALARMES ---
    with alerts_tab:
        st.subheader("Alertes P/FCF < seuil")
        if alerts.empty:
            st.info("Aucune alerte détectée.")
        else:
            st.dataframe(alerts.head(1000))
            st.download_button("⬇️ Télécharger alerts_pf_cf_below20.csv",
                               data=alerts.to_csv(index=False).encode("utf-8"),
                               file_name="alerts_pf_cf_below20.csv",
                               mime="text/csv")

        st.subheader("Résumé des alertes par trimestre")
        if summary is not None and not summary.empty:
            st.dataframe(summary)
            st.download_button("⬇️ Télécharger alerts_pf_cf_summary.csv",
                               data=summary.to_csv(index=False).encode("utf-8"),
                               file_name="alerts_pf_cf_summary.csv",
                               mime="text/csv")

    # --- BACKTEST ---
    with backtest_tab:
        st.subheader("Courbe de performances (NAV vs Benchmark)")
        if not df_equity.empty:
            eq = df_equity.copy()
            eq["equity_norm"] = 100.0 * (eq["nav"] / eq["nav"].iloc[0])
            # 'bench' est déjà une NAV benchmark 100 au démarrage dans ton pipeline
            eq = eq.rename(columns={"bench":"bench_norm"})
            chart_df = eq.set_index("date")[["equity_norm", "bench_norm"]]
            st.line_chart(chart_df)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Performance portefeuille", f"{(eq['equity_norm'].iloc[-1]-100):.2f}%")
            with col2:
                st.metric("Performance benchmark", f"{(eq['bench_norm'].iloc[-1]-100):.2f}%")
            with col3:
                st.metric("Surperformance", f"{(eq['equity_norm'].iloc[-1]-eq['bench_norm'].iloc[-1]):.2f} pts")

            st.download_button("⬇️ Télécharger bt_equity_curve.csv",
                               data=df_equity.to_csv(index=False).encode("utf-8"),
                               file_name="bt_equity_curve.csv",
                               mime="text/csv")
        else:
            st.info("Pas de courbe Equity (peut-être aucune alerte).")

        st.subheader("Trades")
        if df_trades is not None and not df_trades.empty:
            st.dataframe(df_trades)
            st.download_button("⬇️ Télécharger bt_trades.csv",
                               data=df_trades.to_csv(index=False).encode("utf-8"),
                               file_name="bt_trades.csv",
                               mime="text/csv")
        else:
            st.info("Aucun trade exécuté.")

        st.subheader("Positions finales")
        if df_pos is not None and not df_pos.empty:
            st.dataframe(df_pos)
            st.download_button("⬇️ Télécharger bt_positions_lastday.csv",
                               data=df_pos.to_csv(index=False).encode("utf-8"),
                               file_name="bt_positions_lastday.csv",
                               mime="text/csv")
        else:
            st.info("Pas de position finale.")
