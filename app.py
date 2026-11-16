import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from io import StringIO
import sqlite3
from pathlib import Path

st.set_page_config(page_title="Stock Intelligence", page_icon="📈", layout="wide")
DEFAULT = "AAPL,MSFT,NVDA,AMD,AVGO,AMZN,META,GOOGL,TSLA,NFLX,PLTR,CRM,ORCL,MU,QCOM,AMAT,LRCX,KLAC,PANW,CRWD,NOW,UBER,SHOP,COIN,HOOD,JPM,V,COST,WMT,UNH"
HORIZONS = {"1 week": 5, "1 month": 21, "3 months": 63}
DB_PATH = Path(__file__).resolve().parent / "stock_picks_history.sqlite3"

def load_prices(ticker):
    raw = yf.download(ticker, period='2y', interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    return raw[['Open','High','Low','Close','Volume']].dropna().sort_index()

def compute(df, spy):
    x = df.copy()
    close = x['Close']
    x['SMA3'] = close.rolling(3).mean()
    x['EMA20'] = close.ewm(span=20, adjust=False).mean()
    x['SMA29'] = close.rolling(29).mean()
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    x['MACD'] = fast - slow
    x['Signal'] = x['MACD'].ewm(span=9, adjust=False).mean()
    x['Histogram'] = x['MACD'] - x['Signal']
    x['PrevHistogram'] = x['Histogram'].shift(1)
    x['MACD_BullCross'] = (x['MACD'] > x['Signal']) & (x['MACD'].shift(1) <= x['Signal'].shift(1))
    lo = x['Low'].rolling(14).min()
    hi = x['High'].rolling(14).max()
    raw_k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    x['SlowK'] = raw_k.rolling(2).mean()
    x['SlowD'] = x['SlowK'].rolling(2).mean()  # configurable approximation to StockCharts slow stochastic 14,2
    x['Stoch_BullCross'] = (x['SlowK'] > x['SlowD']) & (x['SlowK'].shift(1) <= x['SlowD'].shift(1))
    x['Uptrend'] = (close > x['EMA20']) & (x['EMA20'] > x['EMA20'].shift(5))
    # Confirmation: both crosses within 5 trading sessions, plus rising EMA20 and price above EMA20.
    x['MACD_Recent'] = x['MACD_BullCross'].rolling(5, min_periods=1).max().astype(bool)
    x['Stoch_Recent'] = x['Stoch_BullCross'].rolling(5, min_periods=1).max().astype(bool)
    x['Dual_BullConfirm'] = x['MACD_Recent'] & x['Stoch_Recent'] & x['Uptrend']
    x['Dual_BullNew'] = x['Dual_BullConfirm'] & ~x['Dual_BullConfirm'].shift(1).fillna(False)
    x['VolumeRatio'] = x['Volume'] / x['Volume'].rolling(20).mean().replace(0, np.nan)
    x['Ret5'] = close.pct_change(5)
    x['Ret21'] = close.pct_change(21)
    x['Ret63'] = close.pct_change(63)
    x['Volatility'] = close.pct_change().rolling(63).std() * np.sqrt(252)
    prev_close = close.shift(1)
    true_range = pd.concat([(x['High']-x['Low']), (x['High']-prev_close).abs(), (x['Low']-prev_close).abs()], axis=1).max(axis=1)
    x['ATR14'] = true_range.rolling(14).mean()
    benchmark = spy['Close'].reindex(x.index).ffill()
    x['RelSPY63'] = (close / close.shift(63) - 1) - (benchmark / benchmark.shift(63) - 1)
    x['Spread'] = ((close / close.iloc[0]) / (benchmark / benchmark.iloc[0]) - 1) * 100
    return x.dropna()

def clip_score(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo) * 100, 0, 100))

def score(row, horizon, strategy):
    """Heuristic signal ranking, not a predicted probability or return."""
    price = float(row['Close'])
    trend = .5 * clip_score(100*(price/row['EMA20']-1),-10,10) + .5*clip_score(100*(row['SMA3']/row['SMA29']-1),-8,8)
    macd_strength = clip_score(100*row['Histogram']/price,-2,2)
    relative = clip_score(100*row['RelSPY63'],-25,25)
    volume = clip_score(row['VolumeRatio'],.5,2)
    recent_macd = 100 if row['MACD_Recent'] else (55 if row['MACD'] > row['Signal'] else 0)
    recent_stoch = 100 if row['Stoch_Recent'] else (45 if row['SlowK'] > row['SlowD'] else 0)
    momentum = clip_score(100*row['Ret5'],-12,12)
    if strategy == 'Momentum':
        components = {'MACD crossover':recent_macd,'Stochastic crossover':recent_stoch,
                      'Trend':trend,'Relative strength':relative,'Volume':volume,
                      'Price momentum':momentum}
        weights = {'1 week':[.25,.20,.15,.10,.10,.20],
                   '1 month':[.20,.15,.25,.20,.10,.10],
                   '3 months':[.15,.10,.30,.30,.05,.10]}[horizon]
    else:
        oversold = 100-clip_score(row['SlowK'],10,65)
        recovery = clip_score(100*(row['Histogram']-row['PrevHistogram'])/price,-.4,.4)
        components = {'MACD crossover':recent_macd,'Stochastic crossover':recent_stoch,
                      'Oversold':oversold,'MACD recovery':recovery,
                      'Relative strength':relative,'Volume':volume}
        weights = {'1 week':[.25,.25,.25,.15,.05,.05],
                   '1 month':[.25,.20,.20,.15,.15,.05],
                   '3 months':[.20,.15,.15,.20,.25,.05]}[horizon]
    total = sum(v*w for v,w in zip(components.values(),weights))
    return round(total,1), components

def get_sp500():
    # Wikipedia table is refreshed daily; ticker punctuation is normalized for Yahoo.
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url,headers={'User-Agent':'StockIntelligenceResearch/1.0 (educational stock screener)'},timeout=25)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = next(t for t in tables if 'Symbol' in t.columns and 'Security' in t.columns)
    return list(dict.fromkeys(str(x).replace('.','-').strip() for x in table['Symbol'] if pd.notna(x)))

def load_batch(tickers):
    # A batch request is much faster than 500 individual requests.
    try:
        raw = yf.download(tickers,period='2y',interval='1d',auto_adjust=True,
                          progress=False,threads=True,group_by='ticker',timeout=40)
    except Exception:
        return {}
    found={}
    for ticker in tickers:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                df=raw[ticker].copy()
            elif len(tickers)==1:
                df=raw.copy()
            else:
                continue
            df=df[['Open','High','Low','Close','Volume']].dropna().sort_index()
            if len(df)>=150:
                found[ticker]=df
        except (KeyError,ValueError,TypeError):
            continue
    return found

def potential_sell_target(row, horizon):
    """Illustrative upside scenario, NOT a predicted price or guaranteed exit."""
    days = HORIZONS[horizon]
    price = float(row['Close'])
    atr = float(row['ATR14'])
    if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
        return None
    # Same crossover rules; target is an ATR-based upside scenario scaled to holding period.
    return round(price + 1.5 * atr * np.sqrt(days / 5), 2)

def tracker_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS picks (
        scan_date TEXT NOT NULL, scan_week TEXT NOT NULL, universe TEXT NOT NULL,
        strategy TEXT NOT NULL, horizon TEXT NOT NULL, rank INTEGER NOT NULL,
        ticker TEXT NOT NULL, score REAL NOT NULL, entry_date TEXT NOT NULL,
        entry_price REAL NOT NULL, spy_entry REAL NOT NULL,
        target_days INTEGER NOT NULL, exit_date TEXT, exit_price REAL,
        spy_exit REAL, stock_return_pct REAL, spy_return_pct REAL,
        PRIMARY KEY (scan_week, universe, strategy, horizon, rank)
    )""")
    # Upgrade an existing tracker database without deleting saved selections.
    cols = {r[1] for r in conn.execute('PRAGMA table_info(picks)').fetchall()}
    if 'sell_target' not in cols:
        conn.execute('ALTER TABLE picks ADD COLUMN sell_target REAL')
    conn.commit()
    return conn

def original_crossover_top(frame, horizon, strategy):
    c = frame[(frame['Horizon'] == horizon) & (frame['Strategy'] == strategy)].copy()
    c = c[c['MACD cross ≤5d'] | c['Stoch cross ≤5d']].copy()
    if c.empty:
        return c
    c['Crossover count'] = c['MACD cross ≤5d'].astype(int) + c['Stoch cross ≤5d'].astype(int)
    return c.sort_values(['Dual bullish confirmation','Crossover count','Score','Ticker'],
                         ascending=[False,False,False,True]).head(5)

def save_weekly_picks(frame, histories, spy, universe_name):
    # One immutable snapshot per ISO calendar week/universe/strategy/horizon/rank.
    # Historical returns are evaluated from subsequent trading closes, never today's price.
    common_dates = spy.index
    dates = [x.index[-1] for x in histories.values() if not x.empty]
    if not dates:
        return 0
    scan_dt = min(max(dates), common_dates[-1])
    scan_day = pd.Timestamp(scan_dt).date()
    scan_week = f'{scan_day.isocalendar().year}-W{scan_day.isocalendar().week:02d}'
    spy_close = spy['Close'].loc[spy.index <= scan_dt]
    if spy_close.empty:
        return 0
    spy_entry = float(spy_close.iloc[-1])
    added = 0
    with tracker_connection() as conn:
        for horizon, days in HORIZONS.items():
            for strategy in ('Momentum','Rebound'):
                picks = original_crossover_top(frame, horizon, strategy)
                for rank, (_, pick) in enumerate(picks.iterrows(), 1):
                    ticker = pick['Ticker']
                    hist = histories[ticker]
                    stock_row = hist.loc[hist.index <= scan_dt]
                    if stock_row.empty:
                        continue
                    entry = stock_row.iloc[-1]
                    cur = conn.execute("""INSERT OR IGNORE INTO picks
                        (scan_date,scan_week,universe,strategy,horizon,rank,ticker,score,
                         entry_date,entry_price,spy_entry,target_days,sell_target)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(scan_day),scan_week,universe_name,strategy,horizon,rank,
                         ticker,float(pick['Score']),str(stock_row.index[-1].date()),
                         float(entry['Close']),spy_entry,days,
                         float(pick['Potential sell price']) if pd.notna(pick['Potential sell price']) else None))
                    added += max(cur.rowcount, 0)
    return added

def update_matured_picks(histories, spy):
    # Only fill outcomes once enough later sessions have completed.
    updated = 0
    with tracker_connection() as conn:
        pending = conn.execute('SELECT scan_week,universe,strategy,horizon,rank,ticker,entry_date,target_days,entry_price,spy_entry FROM picks WHERE exit_date IS NULL').fetchall()
        for week, universe, strategy, horizon, rank, ticker, entry_day, days, entry, spy_entry in pending:
            if ticker not in histories:
                continue
            stock = histories[ticker]['Close']
            later = stock.loc[stock.index > pd.Timestamp(entry_day)]
            if len(later) < days:
                continue
            exit_day = later.index[days - 1]
            benchmark = spy['Close'].loc[spy.index <= exit_day]
            if benchmark.empty or benchmark.index[-1].date() != exit_day.date():
                continue
            exit_price, spy_exit = float(later.iloc[days-1]), float(benchmark.iloc[-1])
            conn.execute("""UPDATE picks SET exit_date=?,exit_price=?,spy_exit=?,
                stock_return_pct=?,spy_return_pct=? WHERE scan_week=? AND universe=?
                AND strategy=? AND horizon=? AND rank=?""",
                (str(exit_day.date()),exit_price,spy_exit,100*(exit_price/entry-1),
                 100*(spy_exit/spy_entry-1),week,universe,strategy,horizon,rank))
            updated += 1
    return updated

st.title("📈 Stock Intelligence")
st.caption("Your top five stock opportunities and saved picks. Original crossover settings unchanged.")
with st.sidebar:
    st.header("Scanner")
    universe_choice = st.radio("Stock universe", ["S&P 500 (all constituents)", "Custom watchlist"])
    symbols = st.text_area("Tickers", DEFAULT) if universe_choice == "Custom watchlist" else ""
    run = st.button("🔎 Scan stocks", type="primary", use_container_width=True)
    st.caption("MACD 12/26/9 · Slow Stochastic 14/2 · SMA 3 · EMA 20 · SMA 29 · SPY relative strength")

# Historical selections are available immediately, without running another scan.
try:
    with tracker_connection() as conn:
        history = pd.read_sql_query("SELECT * FROM picks ORDER BY scan_date DESC, rank", conn)
except Exception as exc:
    history = pd.DataFrame()
    st.warning(f"Could not open saved picks: {exc}")

if run:
    try:
        universe = get_sp500() if universe_choice.startswith("S&P") else list(dict.fromkeys(s.strip().upper().replace('.', '-') for s in symbols.replace('\n', ',').split(',') if s.strip() and s.strip().upper() != 'SPY'))
        if not universe:
            st.error("Enter at least one ticker.")
            st.stop()
        spy = load_prices('SPY')
        if spy.empty:
            st.error("SPY market data unavailable. Try again later.")
            st.stop()
        histories, records, failed = {}, [], []
        progress = st.progress(0, text=f"Scanning {len(universe)} stocks...")
        for offset in range(0, len(universe), 35):
            chunk = universe[offset:offset+35]
            batch = load_batch(chunk)
            for symbol in chunk:
                try:
                    raw = batch.get(symbol)
                    if raw is None or len(raw) < 150:
                        failed.append(symbol)
                        continue
                    x = compute(raw, spy)
                    if x.empty:
                        failed.append(symbol)
                        continue
                    histories[symbol] = x
                    last = x.iloc[-1]
                    for h in HORIZONS:
                        for strategy in ('Momentum', 'Rebound'):
                            value, components = score(last, h, strategy)
                            records.append({'Ticker': symbol, 'Strategy': strategy, 'Horizon': h,
                                'Score': value, 'Last price': round(float(last['Close']),2),
                                'Potential sell price': potential_sell_target(last, h),
                                'MACD cross ≤5d': bool(last['MACD_Recent']),
                                'Stoch cross ≤5d': bool(last['Stoch_Recent']),
                                'Dual bullish confirmation': bool(last['Dual_BullConfirm'])})
                except Exception:
                    failed.append(symbol)
            progress.progress(min((offset + len(chunk))/len(universe), 1.0))
        progress.empty()
        if not records:
            st.error("No usable stock data returned. Please try again later.")
        else:
            results = pd.DataFrame(records)
            universe_name = 'S&P 500' if universe_choice.startswith('S&P') else 'Custom: ' + ','.join(sorted(histories))
            added = save_weekly_picks(results, histories, spy, universe_name)
            updated = update_matured_picks(histories, spy)
            st.session_state['latest_scan'] = results
            st.session_state['latest_scan_date'] = str(max(x.index[-1] for x in histories.values()).date())
            st.success(f"Scanned {len(histories)} of {len(universe)} stocks. {added} picks saved; {updated} outcomes updated.")
            if failed:
                st.warning(f"{len(failed)} tickers unavailable or missing data: " + ', '.join(failed[:15]))
            with tracker_connection() as conn:
                history = pd.read_sql_query("SELECT * FROM picks ORDER BY scan_date DESC, rank", conn)
    except Exception as exc:
        st.error(f"Scan failed: {exc}")

strategy = st.radio("Strategy", ['Momentum', 'Rebound'], horizontal=True)
horizon = st.radio("Timeframe", list(HORIZONS), horizontal=True)
st.subheader("🏆 Current top five")
if 'latest_scan' in st.session_state:
    frame = original_crossover_top(st.session_state['latest_scan'], horizon, strategy)
    if frame.empty:
        st.info("No stocks currently meet the original bullish crossover rules.")
    else:
        display = frame[['Ticker','Score','Last price','Potential sell price','MACD cross ≤5d','Stoch cross ≤5d','Dual bullish confirmation']].copy()
        display['Potential gain %'] = ((display['Potential sell price'] / display['Last price'] - 1) * 100).round(1)
        display.insert(0,'Rank',range(1,len(display)+1))
        st.dataframe(display.rename(columns={'Last price':'Entry reference price','MACD cross ≤5d':'MACD crossover','Stoch cross ≤5d':'Stochastic crossover','Dual bullish confirmation':'Dual confirmation'}), hide_index=True, use_container_width=True)
        st.caption('Latest scan date: ' + st.session_state.get('latest_scan_date','unknown'))
        st.caption('Potential sell price = entry reference price + 1.5 × 14-day ATR × √(holding days / 5). This is an illustrative upside scenario, not a statistically calibrated forecast, resistance level, or recommendation to sell.')
elif not history.empty:
    past = history[(history.strategy == strategy)&(history.horizon == horizon)]
    if past.empty:
        st.info("No saved picks for this strategy and timeframe. Click Scan stocks.")
    else:
        last_date = past.scan_date.max()
        picks = past[past.scan_date == last_date].sort_values('rank')
        st.dataframe(picks[['rank','ticker','score','entry_price','sell_target']].rename(columns={'rank':'Rank','ticker':'Stock','score':'Score','entry_price':'Saved entry price','sell_target':'Potential sell price'}), hide_index=True, use_container_width=True)
        st.caption(f"Most recent saved picks: {last_date}. Run Scan stocks for fresh selections.")
else:
    st.info("Click Scan stocks to find your first five picks. Previous picks will appear here automatically.")

st.divider()
st.subheader("📅 Previous picks")
if history.empty:
    st.info("No saved history yet. After your first scan, the app will remember your picks.")
else:
    filtered = history[(history.strategy == strategy)&(history.horizon == horizon)].copy()
    completed = filtered.dropna(subset=['stock_return_pct'])
    a,b,c = st.columns(3)
    a.metric('Picks saved',len(filtered))
    b.metric('Completed',len(completed))
    c.metric('Win rate', f"{(completed.stock_return_pct > 0).mean()*100:.1f}%" if len(completed) else 'Pending')
    if len(completed):
        d,e = st.columns(2)
        d.metric('Average stock return',f"{completed.stock_return_pct.mean():+.2f}%")
        e.metric('Average return vs SPY',f"{(completed.stock_return_pct - completed.spy_return_pct).mean():+.2f}%")
    dates = sorted(filtered.scan_date.unique(), reverse=True)
    chosen = st.selectbox('Saved week', dates) if dates else None
    if chosen:
        picks = filtered[filtered.scan_date == chosen].sort_values('rank')
        cols = ['rank','ticker','score','entry_price','sell_target','exit_date','stock_return_pct','spy_return_pct']
        view = picks[cols].rename(columns={'rank':'Rank','ticker':'Stock','score':'Score','entry_price':'Entry price','sell_target':'Potential sell price','exit_date':'Exit date','stock_return_pct':'Return %','spy_return_pct':'SPY return %'})
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption('Blank returns mean the holding period has not finished or data is unavailable. Older saved picks may have no sell target because targets were not recorded when those picks were made.')
    st.download_button('Download all saved picks (CSV)', history.to_csv(index=False).encode('utf-8'), file_name='stock_intelligence_pick_history.csv', mime='text/csv')
st.caption('Research tool only. Sell targets are illustrative ATR scenarios, not predicted prices. Scores are not probabilities or return forecasts. Weekly scans still require clicking Scan stocks; closing the app does not schedule a scan.')
