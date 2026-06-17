import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Options Day Trade Predictor", page_icon="📈", layout="wide")

# ── constants ──────────────────────────────────────────────────────────────────
BUDGET = 35.0
MAX_PDT_PER_WEEK = 3

# Cheap, liquid, optionable stocks that often trade options under $0.35/contract
WATCHLIST = [
    "SIRI", "VALE", "ITUB", "PBR", "BBD", "ABEV", "NOK", "PLUG",
    "SOFI", "NIO", "RIVN", "LCID", "OPEN", "UWMC", "CLOV",
    "MARA", "RIOT", "CLSK", "HUT", "CIFR",
    "HOOD", "WISH", "BARK", "WKHS", "GOEV",
    "F", "BAC", "C", "T", "INTC", "AMD", "SNAP",
    "PLTR", "SPCE", "NKLA", "IDEX", "SNDL",
    "AMCX", "BB", "GME", "AMC",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def get_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def get_macd(series: pd.Series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd.iloc[-1], signal.iloc[-1]


def get_bb(series: pd.Series, period: int = 20):
    ma = series.rolling(period).mean().iloc[-1]
    std = series.rolling(period).std().iloc[-1]
    return ma - 2 * std, ma, ma + 2 * std


def score_trade(rsi, macd_val, macd_sig, close, bb_low, bb_high, volume, avg_volume):
    """Return (direction, score 0-100, reasons[])"""
    call_score = 0
    put_score = 0
    reasons = []

    # RSI signals
    if rsi < 30:
        call_score += 25
        reasons.append(f"RSI {rsi} → oversold (CALL +25)")
    elif rsi < 45:
        call_score += 10
        reasons.append(f"RSI {rsi} → leaning bullish (CALL +10)")
    elif rsi > 70:
        put_score += 25
        reasons.append(f"RSI {rsi} → overbought (PUT +25)")
    elif rsi > 55:
        put_score += 10
        reasons.append(f"RSI {rsi} → leaning bearish (PUT +10)")

    # MACD crossover
    if macd_val > macd_sig:
        call_score += 20
        reasons.append(f"MACD bullish crossover (CALL +20)")
    else:
        put_score += 20
        reasons.append(f"MACD bearish crossover (PUT +20)")

    # Bollinger Band touch
    if close <= bb_low:
        call_score += 20
        reasons.append(f"Price at lower BB → bounce likely (CALL +20)")
    elif close >= bb_high:
        put_score += 20
        reasons.append(f"Price at upper BB → reversal likely (PUT +20)")

    # Volume surge (momentum confirmation)
    if avg_volume > 0 and volume > 1.5 * avg_volume:
        reasons.append(f"Volume surge {volume/avg_volume:.1f}x avg → momentum confirmed (+15 both)")
        if call_score >= put_score:
            call_score += 15
        else:
            put_score += 15

    direction = "CALL" if call_score >= put_score else "PUT"
    score = max(call_score, put_score)
    score = min(score, 100)
    return direction, score, reasons


def analyze_stock(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="60d", interval="1d")
        if hist is None or len(hist) < 30:
            return None

        info = tk.info or {}
        close = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2]
        volume = int(hist["Volume"].iloc[-1])
        avg_volume = int(hist["Volume"].rolling(20).mean().iloc[-1])

        rsi = get_rsi(hist["Close"])
        macd_val, macd_sig = get_macd(hist["Close"])
        bb_low, bb_mid, bb_high = get_bb(hist["Close"])

        pct_change = round((close - prev_close) / prev_close * 100, 2)
        direction, score, reasons = score_trade(
            rsi, macd_val, macd_sig, close, bb_low, bb_high, volume, avg_volume
        )

        # Estimate cheapest near-money option premium
        iv = info.get("impliedVolatility") or 0.6
        days_to_exp = 1  # same-day / 0DTE mindset; use nearest Friday
        t = days_to_exp / 252
        estimated_premium = round(close * iv * (t ** 0.5) * 0.4, 2)  # rough ATM premium
        estimated_premium = max(estimated_premium, 0.01)
        contracts_affordable = int(BUDGET / (estimated_premium * 100))

        # Estimate realistic intraday target (1 σ move)
        daily_move_pct = iv / (252 ** 0.5)
        target_move = round(close * daily_move_pct, 2)
        target_profit_per_contract = round(target_move * 100 * 0.5, 2)  # delta ~0.5 ATM
        total_potential_profit = round(target_profit_per_contract * contracts_affordable, 2)

        return {
            "ticker": ticker,
            "price": round(close, 2),
            "pct_change": pct_change,
            "volume": volume,
            "avg_volume": avg_volume,
            "rsi": rsi,
            "macd": round(macd_val, 4),
            "macd_signal": round(macd_sig, 4),
            "bb_low": round(bb_low, 2),
            "bb_mid": round(bb_mid, 2),
            "bb_high": round(bb_high, 2),
            "iv": round(iv * 100, 1),
            "direction": direction,
            "score": score,
            "reasons": reasons,
            "estimated_premium": estimated_premium,
            "contracts_affordable": contracts_affordable,
            "target_move": target_move,
            "target_profit": total_potential_profit,
            "sector": info.get("sector", "N/A"),
            "name": info.get("shortName", ticker),
        }
    except Exception:
        return None


def find_options_chain(ticker: str, direction: str):
    """Return the best near-money option within budget."""
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None

        today = datetime.today().date()
        # Pick nearest expiry (ideally this week's Friday)
        future_exps = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        if not future_exps:
            return None
        nearest_exp = future_exps[0]

        chain = tk.option_chain(nearest_exp)
        df = chain.calls if direction == "CALL" else chain.puts

        price = tk.history(period="2d")["Close"].iloc[-1]

        # Filter affordable contracts (ask * 100 <= BUDGET)
        df = df[df["ask"] * 100 <= BUDGET].copy()
        if df.empty:
            return None

        # Prefer near-the-money: minimize |strike - price|
        df["moneyness"] = abs(df["strike"] - price)
        df = df.sort_values("moneyness")
        best = df.iloc[0]

        return {
            "expiry": nearest_exp,
            "strike": best["strike"],
            "ask": best["ask"],
            "bid": best["bid"],
            "volume": best.get("volume", 0),
            "openInterest": best.get("openInterest", 0),
            "impliedVolatility": round(best.get("impliedVolatility", 0) * 100, 1),
            "inTheMoney": best.get("inTheMoney", False),
            "cost_per_contract": round(best["ask"] * 100, 2),
            "contracts": int(BUDGET // (best["ask"] * 100)),
        }
    except Exception:
        return None


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("📈 Daily Options Trade Predictor")
st.caption(f"Budget: **${BUDGET}** · PDT limit: **{MAX_PDT_PER_WEEK} trades/week** · Powered by Yahoo Finance")

with st.expander("⚠️ Risk Disclaimer", expanded=False):
    st.warning(
        "Options trading involves substantial risk and is not suitable for all investors. "
        "This tool is for educational and informational purposes only. Past signals do not "
        "guarantee future results. Never risk money you cannot afford to lose."
    )

# Sidebar controls
st.sidebar.header("⚙️ Settings")
budget_input = st.sidebar.number_input("Your Budget ($)", value=35.0, min_value=5.0, max_value=500.0, step=5.0)
pdt_remaining = st.sidebar.number_input("PDT trades remaining this week", value=3, min_value=0, max_value=3)
top_n = st.sidebar.slider("Stocks to scan", min_value=5, max_value=len(WATCHLIST), value=20)
min_score = st.sidebar.slider("Min confidence score", 0, 100, 40)
st.sidebar.markdown("---")
st.sidebar.markdown("**PDT Rule reminder:** You can make at most 3 round-trip day trades in 5 business days with an account under $25,000.")

if pdt_remaining == 0:
    st.error("🚫 You have 0 PDT trades remaining this week. Wait until Monday to trade again.")
    st.stop()

# Scan button
if st.button("🔍 Scan for Today's Best Trade", type="primary", use_container_width=True):
    scan_list = WATCHLIST[:top_n]

    progress = st.progress(0, text="Scanning stocks…")
    results = []

    for i, ticker in enumerate(scan_list):
        progress.progress((i + 1) / len(scan_list), text=f"Analyzing {ticker}…")
        r = analyze_stock(ticker)
        if r and r["score"] >= min_score and r["contracts_affordable"] >= 1:
            results.append(r)

    progress.empty()

    if not results:
        st.warning("No qualifying trades found. Try lowering the minimum score or expanding the scan.")
        st.stop()

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]

    # ── TOP PICK ──────────────────────────────────────────────────────────────
    st.markdown("---")
    direction_emoji = "🟢" if best["direction"] == "CALL" else "🔴"
    st.markdown(f"## {direction_emoji} TODAY'S TOP PICK: **{best['ticker']}** — {best['direction']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stock Price", f"${best['price']}", f"{best['pct_change']}%")
    col2.metric("Confidence Score", f"{best['score']}/100")
    col3.metric("Direction", best["direction"])
    col4.metric("Est. Implied Vol", f"{best['iv']}%")

    st.markdown(f"**Company:** {best['name']} · **Sector:** {best['sector']}")

    # Options chain lookup
    with st.spinner("Fetching live options chain…"):
        opt = find_options_chain(best["ticker"], best["direction"])

    st.markdown("### 💰 Recommended Contract")
    if opt:
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        oc1.metric("Expiry", opt["expiry"])
        oc2.metric("Strike", f"${opt['strike']}")
        oc3.metric("Ask (cost)", f"${opt['ask']}/share")
        oc4.metric("Cost / Contract", f"${opt['cost_per_contract']}")
        oc5.metric("Contracts w/ ${:.0f}".format(budget_input), str(opt["contracts"]))

        st.markdown(
            f"**Bid/Ask:** ${opt['bid']} / ${opt['ask']} · "
            f"**OI:** {opt['openInterest']:,} · "
            f"**Volume:** {opt['volume']} · "
            f"**IV:** {opt['impliedVolatility']}% · "
            f"**ITM:** {'Yes' if opt['inTheMoney'] else 'No'}"
        )

        target_gain = round(best["target_move"] * 100 * 0.5 * opt["contracts"], 2)
        stop_loss = round(opt["cost_per_contract"] * opt["contracts"] * 0.5, 2)
        st.markdown(
            f"""
            | Scenario | Value |
            |---|---|
            | Total cost | **${opt['cost_per_contract'] * opt['contracts']:.2f}** |
            | 🎯 Target profit (1σ move) | **+${target_gain}** |
            | 🛑 Stop-loss (50% of premium) | **-${stop_loss}** |
            | Risk/Reward | **{round(target_gain/stop_loss, 2) if stop_loss else 'N/A'}:1** |
            """
        )
    else:
        st.info(
            f"No options contract under ${budget_input:.0f} found for {best['ticker']}. "
            "Try checking the broker directly for 0DTE contracts."
        )
        # Show estimated values from stock analysis
        c1, c2 = st.columns(2)
        c1.metric("Est. Premium", f"${best['estimated_premium']}/share")
        c2.metric("Est. Profit Target", f"+${best['target_profit']}")

    # Signal breakdown
    st.markdown("### 📊 Signal Breakdown")
    for r in best["reasons"]:
        st.write(f"• {r}")

    # Technical indicators
    st.markdown("### 🔬 Technical Indicators")
    ti1, ti2, ti3, ti4, ti5 = st.columns(5)
    ti1.metric("RSI (14)", best["rsi"], help="<30 oversold, >70 overbought")
    ti2.metric("MACD", best["macd"])
    ti3.metric("BB Low", f"${best['bb_low']}")
    ti4.metric("BB Mid", f"${best['bb_mid']}")
    ti5.metric("BB High", f"${best['bb_high']}")

    vol_ratio = round(best["volume"] / best["avg_volume"], 2) if best["avg_volume"] else 0
    v1, v2 = st.columns(2)
    v1.metric("Today's Volume", f"{best['volume']:,}")
    v2.metric("Volume vs 20-day Avg", f"{vol_ratio}x")

    # ── TRADE PLAN ────────────────────────────────────────────────────────────
    st.markdown("### 📋 Intraday Trade Plan")
    open_time = "9:35 AM ET"  # wait 5 min after open
    close_time = "3:45 PM ET"  # exit before last 15 min
    st.markdown(
        f"""
        | Step | Action |
        |---|---|
        | **Entry** | Buy **{opt['contracts'] if opt else best['contracts_affordable']}x {best['ticker']} {best['direction']}** at market open (~{open_time}) |
        | **Target exit** | Sell when premium increases **50–100%** or stock moves **${best['target_move']:.2f}** in your direction |
        | **Stop loss** | Exit if premium drops **50%** from entry |
        | **Hard exit** | Close all before **{close_time}** — never hold options overnight on a day trade |
        | **PDT note** | This counts as **1 of {pdt_remaining}** remaining trades this week |
        """
    )

    # ── OTHER CANDIDATES ─────────────────────────────────────────────────────
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### 🔄 Other Candidates (ranked by score)")
        rows = []
        for r in results[1:6]:
            rows.append({
                "Ticker": r["ticker"],
                "Price": f"${r['price']}",
                "Direction": r["direction"],
                "Score": r["score"],
                "RSI": r["rsi"],
                "MACD": "↑" if r["macd"] > r["macd_signal"] else "↓",
                "IV%": r["iv"],
                "Vol Ratio": round(r["volume"] / r["avg_volume"], 2) if r["avg_volume"] else "—",
                "Est. Premium": f"${r['estimated_premium']}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── FULL SCAN TABLE ───────────────────────────────────────────────────────
    with st.expander("📄 Full Scan Results"):
        all_rows = []
        for r in results:
            all_rows.append({
                "Ticker": r["ticker"],
                "Name": r["name"],
                "Price": r["price"],
                "Chg%": r["pct_change"],
                "Direction": r["direction"],
                "Score": r["score"],
                "RSI": r["rsi"],
                "IV%": r["iv"],
                "Est Premium": r["estimated_premium"],
                "Contracts": r["contracts_affordable"],
                "Est Profit": r["target_profit"],
            })
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True)

# ── EDUCATION SIDEBAR ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
### 📚 Quick Guide

**CALL** → buy if you expect price ↑
**PUT** → buy if you expect price ↓

**Score guide:**
- 70–100 → Strong signal
- 50–69 → Moderate signal
- 40–49 → Weak (use caution)

**$35 budget tips:**
- Look for options priced **$0.05–$0.35/share**
- 1 contract = 100 shares of premium
- Prefer **high OI** (>500) for liquidity
- Sell before 3:45 PM ET to avoid theta decay trap
"""
)
