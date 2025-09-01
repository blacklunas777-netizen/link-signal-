import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random

# Configure page

st.set_page_config(page_title=“LINK Trading Signals Dashboard”,
page_icon=“🔗”,
layout=“wide”,
initial_sidebar_state=“expanded”)

# Custom CSS for better styling

st.markdown(”””

<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .signal-active {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .signal-inactive {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>

“””, unsafe_allow_html=True)

# Cache data fetching to avoid repeated API calls

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_link_data(days=120):
“”“Fetch LINK price and volume data from CoinGecko API with retry logic”””
max_retries = 3
base_delay = 2

```
for attempt in range(max_retries):
    try:
        url = "https://api.coingecko.com/api/v3/coins/chainlink/market_chart"
        params = {"vs_currency": "usd", "days": days}

        # Add random delay to avoid hitting rate limits
        if attempt > 0:
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            time.sleep(delay)

        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 429:  # Too Many Requests
            if attempt < max_retries - 1:
                st.warning(
                    f"Rate limited. Retrying in {base_delay * (2 ** attempt)} seconds..."
                )
                continue
            else:
                st.error(
                    "API rate limit exceeded. Please wait a few minutes and try again."
                )
                return None

        resp.raise_for_status()
        data = resp.json()

        prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
        vols = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])

        # Convert ms to datetime, merge, resample to daily
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms")
        vols["date"] = pd.to_datetime(vols["ts"], unit="ms")

        df = prices.merge(vols[["date", "volume"]], on="date")
        df = df.set_index("date")[["price", "volume"]]
        df = df.resample("D").last().ffill()
        return df

    except requests.exceptions.RequestException as e:
        if attempt < max_retries - 1:
            st.warning(
                f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying..."
            )
            continue
        else:
            st.error(
                f"Failed to fetch data after {max_retries} attempts: {str(e)}"
            )
            return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

return None
```

@st.cache_data(ttl=300)
def calculate_technical_indicators(df):
“”“Calculate all technical indicators separately for better caching”””
if df is None or len(df) < 30:
return None

```
df_copy = df.copy()

# Moving averages
df_copy["EMA9"] = df_copy["price"].ewm(span=9, adjust=False).mean()
df_copy["EMA21"] = df_copy["price"].ewm(span=21, adjust=False).mean()
df_copy["SMA20"] = df_copy["price"].rolling(20).mean()

# MACD
ema12 = df_copy["price"].ewm(span=12, adjust=False).mean()
ema26 = df_copy["price"].ewm(span=26, adjust=False).mean()
df_copy["MACD"] = ema12 - ema26
df_copy["MACD_signal"] = df_copy["MACD"].ewm(span=9, adjust=False).mean()

# RSI
delta = df_copy["price"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
rs = avg_gain / (avg_loss + 1e-8)  # Add small epsilon to avoid division by zero
df_copy["RSI"] = 100 - (100 / (1 + rs))

# Bollinger Bands
df_copy["STD20"] = df_copy["price"].rolling(20).std()
df_copy["BB_upper"] = df_copy["SMA20"] + 2 * df_copy["STD20"]
df_copy["BB_lower"] = df_copy["SMA20"] - 2 * df_copy["STD20"]

# Stochastic Oscillator
low14 = df_copy["price"].rolling(14).min()
high14 = df_copy["price"].rolling(14).max()
df_copy["%K"] = ((df_copy["price"] - low14) / (high14 - low14 + 1e-8)) * 100
df_copy["%D"] = df_copy["%K"].rolling(3).mean()

# Volume moving average
df_copy["volume_sma20"] = df_copy["volume"].rolling(20).mean()

return df_copy
```

def detect_buy_signals(df_with_indicators):
“”“Compute all six buy-signal flags from pre-calculated indicators”””
if df_with_indicators is None or len(df_with_indicators) < 30:
return None

```
signals = {}
df = df_with_indicators

try:
    # Check if we have enough data
    if len(df) < 2:
        return {key: False for key in ["MA_crossover", "MACD", "RSI", "Bollinger", "Stochastic", "Volume"]}

    # 1) Moving-Average Crossover (EMA9 > EMA21)
    if not df["EMA9"].isna().iloc[-2:].any() and not df["EMA21"].isna().iloc[-2:].any():
        prev_e9, prev_e21 = df["EMA9"].iloc[-2], df["EMA21"].iloc[-2]
        cur_e9, cur_e21 = df["EMA9"].iloc[-1], df["EMA21"].iloc[-1]
        signals["MA_crossover"] = (prev_e9 < prev_e21) and (cur_e9 > cur_e21)
    else:
        signals["MA_crossover"] = False

    # 2) MACD Bullish Crossover Above Zero
    if not df["MACD"].isna().iloc[-2:].any() and not df["MACD_signal"].isna().iloc[-2:].any():
        prev_m, prev_s = df["MACD"].iloc[-2], df["MACD_signal"].iloc[-2]
        cur_m, cur_s = df["MACD"].iloc[-1], df["MACD_signal"].iloc[-1]
        signals["MACD"] = (prev_m < prev_s) and (cur_m > cur_s) and (cur_m > 0)
    else:
        signals["MACD"] = False

    # 3) RSI Rebound from Oversold
    if not df["RSI"].isna().iloc[-2:].any():
        prev_r, cur_r = df["RSI"].iloc[-2], df["RSI"].iloc[-1]
        signals["RSI"] = (prev_r < 30) and (cur_r > 30)
    else:
        signals["RSI"] = False

    # 4) Bollinger Band Breakout
    if not df["BB_upper"].isna().iloc[-2:].any():
        prev_p, cur_p = df["price"].iloc[-2], df["price"].iloc[-1]
        prev_u, cur_u = df["BB_upper"].iloc[-2], df["BB_upper"].iloc[-1]
        signals["Bollinger"] = (prev_p <= prev_u) and (cur_p > cur_u)
    else:
        signals["Bollinger"] = False

    # 5) Stochastic Oscillator
    if not df["%K"].isna().iloc[-2:].any() and not df["%D"].isna().iloc[-2:].any():
        prev_k, prev_d = df["%K"].iloc[-2], df["%D"].iloc[-2]
        cur_k, cur_d = df["%K"].iloc[-1], df["%D"].iloc[-1]
        signals["Stochastic"] = (prev_k < prev_d) and (cur_k > cur_d) and (prev_k < 20)
    else:
        signals["Stochastic"] = False

    # 6) Volume Surge
    if not df["volume_sma20"].isna().iloc[-2:].any():
        avg_vol_val = df["volume_sma20"].iloc[-2]
        cur_vol = df["volume"].iloc[-1]
        signals["Volume"] = cur_vol > avg_vol_val * 1.7
    else:
        signals["Volume"] = False

    return signals

except Exception as e:
    st.error(f"Error calculating signals: {str(e)}")
    return None
```

def create_price_chart(df_with_indicators):
“”“Create interactive price chart with technical indicators”””
fig = make_subplots(
rows=4, cols=1,
subplot_titles=(‘Price & Moving Averages’, ‘MACD’, ‘RSI’, ‘Stochastic Oscillator’),
vertical_spacing=0.08,
row_heights=[0.4, 0.2, 0.2, 0.2],
specs=[[{“secondary_y”: False}], [{“secondary_y”: False}],
[{“secondary_y”: False}], [{“secondary_y”: False}]]
)

```
# Price and Bollinger Bands
fig.add_trace(go.Scatter(
    x=df_with_indicators.index,
    y=df_with_indicators['price'],
    name='LINK Price',
    line=dict(color='#1f77b4', width=2)
), row=1, col=1)

# Moving averages
if 'EMA9' in df_with_indicators.columns:
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['EMA9'],
        name='EMA9',
        line=dict(color='orange', width=1)
    ), row=1, col=1)

if 'EMA21' in df_with_indicators.columns:
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['EMA21'],
        name='EMA21',
        line=dict(color='red', width=1)
    ), row=1, col=1)

# Bollinger Bands
if all(col in df_with_indicators.columns for col in ['BB_upper', 'BB_lower', 'SMA20']):
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['BB_upper'],
        name='BB Upper',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['BB_lower'],
        name='BB Lower',
        line=dict(color='gray', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)

# MACD
if all(col in df_with_indicators.columns for col in ['MACD', 'MACD_signal']):
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['MACD_signal'],
        name='Signal',
        line=dict(color='red')
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

# RSI
if 'RSI' in df_with_indicators.columns:
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# Stochastic
if all(col in df_with_indicators.columns for col in ['%K', '%D']):
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['%K'],
        name='%K',
        line=dict(color='blue')
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_with_indicators.index,
        y=df_with_indicators['%D'],
        name='%D',
        line=dict(color='orange')
    ), row=4, col=1)
    
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)

fig.update_layout(
    height=800,
    title_text="LINK Technical Analysis",
    showlegend=True,
    hovermode='x unified'
)
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

return fig
```

def create_volume_chart(df):
“”“Create volume chart”””
fig = go.Figure()

```
# Color bars based on price movement
colors = ['green']  # First bar
for i in range(1, len(df)):
    if df['price'].iloc[i] >= df['price'].iloc[i-1]:
        colors.append('green')
    else:
        colors.append('red')

fig.add_trace(go.Bar(
    x=df.index,
    y=df['volume'],
    name='Volume',
    marker_color=colors,
    opacity=0.7
))

# Add 20-day volume average
if len(df) >= 20:
    vol_avg = df['volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=vol_avg,
        name='20-day Avg Volume',
        line=dict(color='blue', dash='dash')
    ))

fig.update_layout(
    title="LINK Trading Volume",
    xaxis_title="Date",
    yaxis_title="Volume (USD)",
    height=300,
    hovermode='x unified'
)

return fig
```

def display_signal_grid(signals):
“”“Display signals in a nice grid format”””
signal_names = {
“MA_crossover”: “Moving Average Crossover”,
“MACD”: “MACD Bullish Crossover”,
“RSI”: “RSI Oversold Recovery”,
“Bollinger”: “Bollinger Band Breakout”,
“Stochastic”: “Stochastic Oscillator”,
“Volume”: “Volume Surge”
}

```
# Create 2x3 grid
cols = st.columns(3)
for i, (key, name) in enumerate(signal_names.items()):
    col = cols[i % 3]
    with col:
        if signals[key]:
            st.success(f"🟢 {name}")
        else:
            st.error(f"🔴 {name}")
```

def main():
st.title(“🔗 Chainlink (LINK) Trading Signals Dashboard”)
st.markdown(“Real-time technical analysis using 6 key indicators”)

```
# Sidebar controls
st.sidebar.header("⚙️ Configuration")
days = st.sidebar.selectbox("📅 Time Period", [30, 60, 90, 120], index=2)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Fetch data
with st.spinner("📊 Fetching LINK data..."):
    df = fetch_link_data(days)

if df is None:
    st.error("❌ Failed to fetch data. Please check your internet connection and try again.")
    st.stop()

# Calculate technical indicators
with st.spinner("🔍 Calculating technical indicators..."):
    df_with_indicators = calculate_technical_indicators(df)

if df_with_indicators is None:
    st.error("❌ Failed to calculate technical indicators.")
    st.stop()

# Calculate signals
with st.spinner("📈 Analyzing signals..."):
    signals = detect_buy_signals(df_with_indicators)

if signals is None:
    st.error("❌ Failed to calculate trading signals.")
    st.stop()

# Current price and basic info
current_price = df['price'].iloc[-1]
price_change = df['price'].iloc[-1] - df['price'].iloc[-2] if len(df) >= 2 else 0
price_change_pct = (price_change / df['price'].iloc[-2] * 100) if len(df) >= 2 and df['price'].iloc[-2] != 0 else 0

# Header metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Current Price", f"${current_price:.4f}",
              f"{price_change:+.4f} ({price_change_pct:+.2f}%)")

with col2:
    current_volume = df['volume'].iloc[-1]
    st.metric("📊 24h Volume", f"${current_volume:,.0f}")

with col3:
    active_signals = sum(signals.values())
    st.metric("🎯 Active Signals", f"{active_signals}/6")

with col4:
    signal_strength = active_signals / 6 * 100
    st.metric("💪 Signal Strength", f"{signal_strength:.0f}%")

# Overall signal assessment
st.subheader("🚦 Trading Signal Assessment")

if active_signals >= 4:
    st.success("🚀 **STRONG BUY SIGNAL** - Multiple indicators aligned! Consider entering position.")
elif active_signals >= 2:
    st.warning("⚠️ **MODERATE BUY SIGNAL** - Some indicators active. Wait for more confirmation.")
else:
    st.info("ℹ️ **WEAK SIGNAL** - Wait for more confirmation before entering.")

# Signal status grid
st.subheader("📊 Individual Signal Status")
display_signal_grid(signals)

# Charts
st.subheader("📈 Technical Analysis Charts")

# Price chart with indicators
price_fig = create_price_chart(df_with_indicators)
st.plotly_chart(price_fig, use_container_width=True)

# Volume chart
volume_fig = create_volume_chart(df)
st.plotly_chart(volume_fig, use_container_width=True)

# Signal details
with st.expander("📋 Signal Details & Explanations"):
    st.markdown("""
    **Trading Signal Explanations:**
    
    - **Moving Average Crossover**: EMA9 crosses above EMA21 (bullish momentum)
    - **MACD**: MACD line crosses above signal line while positive (trend confirmation)
    - **RSI**: RSI rebounds from oversold territory (below 30) - bounce from support
    - **Bollinger Bands**: Price breaks above upper band (momentum breakout)
    - **Stochastic**: %K crosses above %D from oversold (below 20) - reversal signal
    - **Volume Surge**: Current volume >170% of 20-day average (institutional interest)
    
    **Risk Management:**
    - Never risk more than 2% of your portfolio on a single trade
    - Use stop-losses 5-10% below entry price
    - Take profits at key resistance levels
    - Consider market conditions and overall crypto sentiment
    """)

# Current indicator values
with st.expander("🔢 Current Indicator Values"):
    if len(df_with_indicators) > 0:
        latest = df_with_indicators.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**EMA9**: ${latest.get('EMA9', 'N/A'):.4f}")
            st.write(f"**EMA21**: ${latest.get('EMA21', 'N/A'):.4f}")
            st.write(f"**RSI**: {latest.get('RSI', 'N/A'):.2f}")
        
        with col2:
            st.write(f"**MACD**: {latest.get('MACD', 'N/A'):.4f}")
            st.write(f"**MACD Signal**: {latest.get('MACD_signal', 'N/A'):.4f}")
            st.write(f"**%K**: {latest.get('%K', 'N/A'):.2f}")
        
        with col3:
            st.write(f"**BB Upper**: ${latest.get('BB_upper', 'N/A'):.4f}")
            st.write(f"**BB Lower**: ${latest.get('BB_lower', 'N/A'):.4f}")
            st.write(f"**%D**: {latest.get('%D', 'N/A'):.2f}")

# Data table
with st.expander("📊 Recent Price Data"):
    display_cols = ['price', 'volume', 'EMA9', 'EMA21', 'RSI', 'MACD']
    available_cols = [col for col in display_cols if col in df_with_indicators.columns]
    st.dataframe(
        df_with_indicators[available_cols].tail(10).round(4),
        use_container_width=True
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Disclaimer**: This dashboard is for educational purposes only. Not financial advice.")
st.sidebar.write(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

if **name** == “**main**”:
main()
