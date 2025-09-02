# 📊 Quality-Value Backtest

Projet développé dans le cadre de mon **MS Expert en Banque et Ingénierie Financière** à TBS Education.  
Ce projet vise à construire, backtester et analyser une stratégie d’investissement **combinant les approches Quality et Value**, à travers un pipeline Python et une application interactive **Streamlit**.

---

## 🚀 Fonctionnalités principales

- Chargement d’un panel d’actions (CSV ou chemin local)  
- Détection des alertes basées sur le multiple **P/FCF**  
- Backtest d’un portefeuille équipondéré vs benchmark (S&P 500, CAC 40, MSCI World)  
- Analyse de performance :
  - CAGR, volatilité, Sharpe ratio, max drawdown  
  - Courbe NAV vs benchmark  
  - Drawdown et distribution des rendements  
- Top & Flop performers  
- Pondérations trimestrielles avec affichage des positions  
- Génération automatique d’un **rapport PDF** (performance, graphiques, top/flop, pondérations)  

---

## 🛠️ Installation

Cloner le dépôt et se placer dans le projet :  
```bash
git clone https://github.com/DodoLeDozo/quality_value_backtest.git
cd quality_value_backtest

