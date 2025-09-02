import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import calendar
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# === Charger les fichiers ===
equity = pd.read_csv("bt_equity_curve.csv")
trades = pd.read_csv("bt_trades.csv")
positions = pd.read_csv("bt_positions_lastday.csv")

# Conversion des dates
equity['date'] = pd.to_datetime(equity['date'])
trades['date'] = pd.to_datetime(trades['date'])
equity.set_index('date', inplace=True)

# === Calcul des rendements ===
equity['returns'] = equity['nav'].pct_change()
equity['bench_returns'] = equity['bench'].pct_change()
excess_returns = equity['returns'] - equity['bench_returns']

# === Performance Globale ===
start_val = equity['nav'].iloc[0]
end_val = equity['nav'].iloc[-1]
years = (equity.index[-1] - equity.index[0]).days / 365
cagr = (end_val / start_val) ** (1/years) - 1

# Risque
volatility = equity['returns'].std() * np.sqrt(252)
sharpe = (equity['returns'].mean() / equity['returns'].std()) * np.sqrt(252)
sortino = (equity['returns'].mean() /
           equity.loc[equity['returns'] < 0, 'returns'].std()) * np.sqrt(252)

# Alpha / Beta
cov_matrix = np.cov(equity['returns'].dropna(), equity['bench_returns'].dropna())
beta = cov_matrix[0, 1] / cov_matrix[1, 1]
alpha = (equity['returns'].mean() - beta * equity['bench_returns'].mean()) * 252

# Tracking error & IR
tracking_error = excess_returns.std() * np.sqrt(252)
information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Max Drawdown
cum_returns = (1 + equity['returns']).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Value at Risk
VaR_95 = equity['returns'].mean() - 1.65 * equity['returns'].std()

# === Graphiques classiques ===
plt.figure(figsize=(8,5))
plt.plot(equity.index, equity['nav'], label="Fund NAV", color="blue")
plt.plot(equity.index, equity['bench'], label="S&P 500", color="orange")
plt.legend()
plt.title("Performance vs S&P 500")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("perf_vs_bench.png")
plt.close()

plt.figure(figsize=(8,4))
plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.5)
plt.title("Drawdown")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("drawdown.png")
plt.close()

plt.figure(figsize=(8,4))
plt.hist(equity['returns'].dropna(), bins=30, color="green", alpha=0.7)
plt.title("Distribution des Rendements")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hist_returns.png")
plt.close()

# === Performance par titre ===
perf_per_ticker = trades.groupby("ticker", group_keys=False).apply(
    lambda x: (x['price'].iloc[-1] / x['price'].iloc[0] - 1) * 100
).sort_values(ascending=False)

top_5 = perf_per_ticker.head(5)
worst_5 = perf_per_ticker.tail(5)

# === Positions trimestrielles ===
trades['Quarter'] = trades['date'].dt.to_period("Q")
quarterly_positions = trades.groupby(['Quarter', 'ticker']).agg({
    'shares': 'sum',
    'price': 'mean',
    'value': 'sum'
}).reset_index()

# === Fonction logo FMP ===
def get_company_logo(ticker, api_key="U5alSX6tkG83UKjwphIAEJD9NAdL5dk0"):
    try:
        url = f"https://financialmodelingprep.com/image-stock/{ticker}.png?apikey={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            filename = f"logo_{ticker}.png"
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
    except:
        pass
    return None

# === Gestion du cache ===
cache_file = "company_cache.csv"
if os.path.exists(cache_file):
    cache = pd.read_csv(cache_file)
else:
    cache = pd.DataFrame(columns=["ticker", "name", "logo_file"])

company_data = {}
for t in quarterly_positions['ticker'].unique():
    cached = cache.loc[cache['ticker'] == t]
    if not cached.empty and pd.notna(cached['logo_file'].values[0]) and os.path.exists(str(cached['logo_file'].values[0])):
        name = cached['name'].values[0]
        logo = cached['logo_file'].values[0]
    else:
        try:
            info = yf.Ticker(t).info
            name = info.get("longName", t)
        except:
            name = t
        logo = get_company_logo(t)
        cache = cache[cache['ticker'] != t]
        cache = pd.concat([cache, pd.DataFrame([{"ticker": t, "name": name, "logo_file": logo}])])
        cache.to_csv(cache_file, index=False)
    company_data[t] = {"name": name, "logo": logo}

# === Heatmaps rendements mensuels ===
monthly_returns = equity['returns'].resample('ME').apply(lambda x: (1+x).prod()-1)
monthly_bench = equity['bench_returns'].resample('ME').apply(lambda x: (1+x).prod()-1)

df_strat = monthly_returns.to_frame(name="Stratégie")
df_strat["Année"] = df_strat.index.year
df_strat["Mois"] = df_strat.index.month

df_bench = monthly_bench.to_frame(name="S&P 500")
df_bench["Année"] = df_bench.index.year
df_bench["Mois"] = df_bench.index.month

heatmap_strat = df_strat.pivot(index="Année", columns="Mois", values="Stratégie")
heatmap_bench = df_bench.pivot(index="Année", columns="Mois", values="S&P 500")
heatmap_diff = heatmap_strat - heatmap_bench

def plot_heatmap(data, title, filename):
    plt.figure(figsize=(8,3.5))
    sns.heatmap(data, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
                cbar_kws={'label': 'Rendement mensuel'})
    plt.title(title)
    plt.xticks(ticks=np.arange(12)+0.5, labels=[calendar.month_abbr[i] for i in range(1,13)])
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_heatmap(heatmap_strat, "Heatmap rendements mensuels — Stratégie", "heatmap_strat.png")
plot_heatmap(heatmap_bench, "Heatmap rendements mensuels — S&P 500", "heatmap_bench.png")
plot_heatmap(heatmap_diff, "Sur/Sous-performance vs S&P 500", "heatmap_diff.png")

# === Génération PDF ===
pdf = FPDF()
pdf.add_page()

pdf.set_fill_color(30, 144, 255)
pdf.set_text_color(255, 255, 255)
pdf.set_font("Helvetica", 'B', 16)
pdf.cell(0, 15, "Rapport de Performance du Fonds",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT,
         align="C", fill=True)

pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", size=12)
pdf.ln(10)

pdf.multi_cell(0, 10, text=f"""
Performance Globale (5 ans) :
- Valeur initiale : {start_val:,.2f}
- Valeur finale : {end_val:,.2f}
- CAGR : {cagr:.2%}
- Volatilité annualisée : {volatility:.2%}
- Sharpe Ratio : {sharpe:.2f}
- Sortino Ratio : {sortino:.2f}
- Max Drawdown : {max_drawdown:.2%}
- VaR 95% : {VaR_95:.2%}

Comparaison Benchmark (S&P 500) :
- Alpha de Jensen : {alpha:.2%}
- Bêta : {beta:.2f}
- Tracking Error : {tracking_error:.2%}
- Information Ratio : {information_ratio:.2f}
""")

# === Page unique avec les 3 graphiques de performance ===
pdf.add_page()
pdf.set_font("Helvetica", 'B', 14)
pdf.cell(0, 10, "Graphiques de Performance",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

pdf.ln(5)  # espace en haut

img_w = 180
img_h = 60

pdf.image("perf_vs_bench.png", x=15, y=pdf.get_y(), w=img_w, h=img_h)
pdf.ln(img_h + 5)

pdf.image("drawdown.png", x=15, y=pdf.get_y(), w=img_w, h=img_h)
pdf.ln(img_h + 5)

pdf.image("hist_returns.png", x=15, y=pdf.get_y(), w=img_w, h=img_h)

# === Page Heatmaps (verticales) ===
pdf.add_page()
pdf.set_font("Helvetica", 'B', 14)
pdf.cell(0, 10, "Analyse des Rendements Mensuels",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

pdf.image("heatmap_strat.png", x=15, y=30, w=180)
pdf.image("heatmap_bench.png", x=15, y=110, w=180)
pdf.image("heatmap_diff.png", x=15, y=190, w=180)

# === TOP & FLOP (même page) ===
pdf.add_page()
pdf.set_font("Helvetica", 'B', 14)
pdf.cell(0, 10, "TOP 5 & FLOP 5 Performers", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

pdf.set_font("Helvetica", size=10)
pdf.ln(5)
pdf.set_fill_color(200, 200, 200)
pdf.cell(95, 8, "TOP 5", border=1, fill=True)
pdf.cell(95, 8, "FLOP 5", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

for i in range(5):
    left = top_5.iloc[i] if i < len(top_5) else None
    right = worst_5.iloc[i] if i < len(worst_5) else None
    left_text = f"{top_5.index[i]} - {company_data[top_5.index[i]]['name']} : {top_5.iloc[i]:.2f}%" if left is not None else ""
    right_text = f"{worst_5.index[i]} - {company_data[worst_5.index[i]]['name']} : {worst_5.iloc[i]:.2f}%" if right is not None else ""
    pdf.cell(95, 8, left_text[:60], border=1)
    pdf.cell(95, 8, right_text[:60], border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# === Positions trimestrielles compactes ===
for q, group in quarterly_positions.groupby("Quarter"):
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, f"Positions - {q}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", size=8)
    total_value = group['value'].sum()

    pdf.set_fill_color(200, 200, 200)
    pdf.cell(25, 8, "Ticker", border=1, fill=True, align="C")
    pdf.cell(75, 8, "Société", border=1, fill=True, align="C")
    pdf.cell(25, 8, "Pond.", border=1, fill=True, align="C")
    pdf.cell(40, 8, "Logo", border=1, fill=True, align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    row_height = 8
    for idx, row in group.iterrows():
        t = row['ticker']
        pct = row['value'] / total_value * 100
        name = company_data[t]["name"]
        logo = company_data[t]["logo"]

        fill = 245 if idx % 2 == 0 else 255
        pdf.set_fill_color(fill, fill, fill)

        pdf.cell(25, row_height, t, border=1, align="C", fill=True)
        pdf.cell(75, row_height, name[:40], border=1, align="C", fill=True)
        pdf.cell(25, row_height, f"{pct:.2f}%", border=1, align="C", fill=True)
        if logo:
            pdf.cell(40, row_height, "", border=1, align="C", fill=True,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.image(logo, x=pdf.get_x()-40+15, y=pdf.get_y()+1, w=8, h=6)
            pdf.ln(row_height)
        else:
            pdf.cell(40, row_height, "N/A", border=1, align="C", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Sauvegarde
pdf.output("rapport_fonds.pdf")
print("✅ Rapport PDF généré : rapport_fonds.pdf")
