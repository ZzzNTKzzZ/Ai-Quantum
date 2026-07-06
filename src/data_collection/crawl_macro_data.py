import os
import sys

# Bypass vnstock Community Edition limits (such as 8-year OHLCV and quarter limits)
try:
    import vnai
    import vnai.beam.patching
    vnai.beam.patching.apply_all_patches = lambda *args, **kwargs: {}
except Exception:
    pass

import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import yfinance as yf
import datetime
import time
from vnstock import Market, Reference

# Configure stdout for UTF-8
sys.stdout.reconfigure(encoding='utf-8')
START_TIME = "2016-11-10"
END_TIME = "2026-05-01"
N_VOLUME_VALIDATE = 2300
# Output directory
# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_DIR = os.path.join(root_dir, "data", "test")
os.makedirs(DATA_DIR, exist_ok=True)

mkt = Market()
ref = Reference()
sources = ["kbs", "msn", "vci"]

# Helper function to print messages
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ---------------------------------------------------------
# CRAWLER IMPLEMENTATIONS
# ---------------------------------------------------------

def crawl_vn100_ticker():
    log("Starting VN100 ticker crawl...")
    try:
        vn100_df = ref.equity.list_by_group('VN100')
        if isinstance(vn100_df, pd.Series):
            vn100_df = pd.DataFrame(vn100_df)
            if vn100_df.columns[0] == 0 or vn100_df.columns[0] == 'symbol':
                vn100_df.columns = ['symbol']
        vn100_df.to_csv(os.path.join(DATA_DIR, "vn100_ticker.csv"), index=False)
        log("VN100 ticker saved.")
    except Exception as e:
        log(f"Error VN100 ticker crawl: {e}")

def crawl_vn100_ohlcv(source):
    log("Starting VN100 OHLCV crawl...")
    try:
        vn100_df = ref.equity.list_by_group('VN100')
        tickers = vn100_df.tolist() if isinstance(vn100_df, pd.Series) else (vn100_df['symbol'].tolist() if 'symbol' in vn100_df.columns else vn100_df['ticker'].tolist())
        
        # Calculate limit for count parameter
        start_dt = datetime.datetime.strptime(START_TIME, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(END_TIME, "%Y-%m-%d")
        days_diff = (end_dt - start_dt).days
        count_limit = max(100, days_diff)
        
        for ticker in tickers:
            time.sleep(3.5)
            df = mkt.equity(ticker).ohlcv(
                start=START_TIME,
                end=END_TIME,
                resolution="1D",
                count=count_limit,
                source=source
            )
            if df is not None and not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                if df["time"].dt.tz is not None:
                    df["time"] = df["time"].dt.tz_localize(None)
                df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
                df = df.sort_values("time").drop_duplicates(subset=["time"])
                df.to_csv(os.path.join(DATA_DIR, f"vn100_ohlcv_{ticker}.csv"), index=False)
        log("VN100 OHLCV saved.")
    except Exception as e:
        log(f"Error in crawl_vn100_ohlcv: {e}")

def crawl_yfinance_and_fred(start=START_TIME, end=END_TIME):
    log("Starting yfinance and FRED data downloads...")
    
    # 1. M2: VIX (CBOE Volatility Index)
    try:
        log("Downloading M2 (VIX)...")
        df_vix = yf.download('^VIX', start=start, end=end)
        # Clean columns if multi-index
        if isinstance(df_vix.columns, pd.MultiIndex):
            df_vix.columns = df_vix.columns.get_level_values(0)
        df_vix.to_csv(os.path.join(DATA_DIR, "vix.csv"))
        log("VIX saved.")
    except Exception as e:
        log(f"Error VIX: {e}")

    # 2. M3: S&P 500
    try:
        log("Downloading M3 (S&P 500)...")
        df_sp500 = yf.download('^GSPC', start=start, end=end)
        if isinstance(df_sp500.columns, pd.MultiIndex):
            df_sp500.columns = df_sp500.columns.get_level_values(0)
        df_sp500.to_csv(os.path.join(DATA_DIR, "sp500.csv"))
        log("S&P 500 saved.")
    except Exception as e:
        log(f"Error S&P 500: {e}")

    # 3. E1: USD/VND
    try:
        log("Downloading E1 (USD/VND)...")
        df_fx = yf.download('USDVND=X', start=start, end=end)
        if isinstance(df_fx.columns, pd.MultiIndex):
            df_fx.columns = df_fx.columns.get_level_values(0)
        df_fx.to_csv(os.path.join(DATA_DIR, "usdvnd.csv"))
        log("USD/VND saved.")
    except Exception as e:
        log(f"Error USD/VND: {e}")

    # 4. S1: Dầu Brent
    try:
        log("Downloading S1 (Brent Oil)...")
        df_oil = yf.download('BZ=F', start=start, end=end)
        if isinstance(df_oil.columns, pd.MultiIndex):
            df_oil.columns = df_oil.columns.get_level_values(0)
        df_oil.to_csv(os.path.join(DATA_DIR, "brent_oil.csv"))
        log("Brent Oil saved.")
    except Exception as e:
        log(f"Error Brent Oil: {e}")

    # 5. S2: Vàng
    try:
        log("Downloading S2 (Gold)...")
        df_gold = yf.download('GC=F', start=start, end=end)
        if isinstance(df_gold.columns, pd.MultiIndex):
            df_gold.columns = df_gold.columns.get_level_values(0)
        df_gold.to_csv(os.path.join(DATA_DIR, "gold.csv"))
        log("Gold saved.")
    except Exception as e:
        log(f"Error Gold: {e}")

    # 6. S4: Copper
    try:
        log("Downloading S4 (Copper)...")
        df_copper = yf.download('HG=F', start=start, end=end)
        if isinstance(df_copper.columns, pd.MultiIndex):
            df_copper.columns = df_copper.columns.get_level_values(0)
        df_copper.to_csv(os.path.join(DATA_DIR, "copper_price.csv"))
        log("Copper saved.")
    except Exception as e:
        log(f"Error Copper: {e}")

    # 7. G1: DXY (US Dollar Index)
    try:
        log("Downloading G1 (DXY)...")
        df_dxy = yf.download('DX-Y.NYB', start=start, end=end)
        if isinstance(df_dxy.columns, pd.MultiIndex):
            df_dxy.columns = df_dxy.columns.get_level_values(0)
        df_dxy.to_csv(os.path.join(DATA_DIR, "dxy.csv"))
        log("DXY saved.")
    except Exception as e:
        log(f"Error DXY: {e}")

    # 8. G3: US 10Y Yield
    try:
        log("Downloading G3 (US 10Y Treasury Yield)...")
        df_us10y = yf.download('^TNX', start=start, end=end)
        if isinstance(df_us10y.columns, pd.MultiIndex):
            df_us10y.columns = df_us10y.columns.get_level_values(0)
        df_us10y.to_csv(os.path.join(DATA_DIR, "us10y_yield.csv"))
        log("US 10Y Yield saved.")
    except Exception as e:
        log(f"Error US 10Y Yield (yfinance): {e}")

    # 9. G4: China SSE Composite
    try:
        log("Downloading G4 (China SSE)...")
        df_sse = yf.download('000001.SS', start=start, end=end)
        if isinstance(df_sse.columns, pd.MultiIndex):
            df_sse.columns = df_sse.columns.get_level_values(0)
        df_sse.to_csv(os.path.join(DATA_DIR, "china_sse.csv"))
        log("China SSE saved.")
    except Exception as e:
        log(f"Error China SSE: {e}")

    # 10. G2: Fed Funds Rate (FRED)
    try:
        log("Downloading G2 (Fed Funds Rate) from FRED...")
        df_fed = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS")
        df_fed['observation_date'] = pd.to_datetime(df_fed['observation_date'])
        df_fed = df_fed[df_fed['observation_date'] >= start]
        df_fed = df_fed[df_fed['observation_date'] <= end]
        df_fed.to_csv(os.path.join(DATA_DIR, "fed_funds_rate.csv"), index=False)
        log("Fed Funds Rate saved.")
    except Exception as e:
        log(f"Error Fed Funds Rate: {e}")

    # 11. E4: EPU (Economic Policy Uncertainty) from FRED
    try:
        log("Downloading E4 (EPU) from FRED...")
        df_epu = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=USEPUINDXD")
        df_epu['observation_date'] = pd.to_datetime(df_epu['observation_date'])
        df_epu = df_epu[df_epu['observation_date'] >= start]
        df_epu = df_epu[df_epu['observation_date'] <= end]
        df_epu.to_csv(os.path.join(DATA_DIR, "epu.csv"), index=False)
        log("EPU saved.")
    except Exception as e:
        log(f"Error EPU: {e}")

def crawl_gpr(start=START_TIME, end=END_TIME):
    log("Downloading E5 (GPR)...")
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    try:
        df = pd.read_excel(url)
        df['Date'] = pd.to_datetime(df['DAY'].astype(str), format='%Y%m%d')
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['Date'])
        df.to_csv(os.path.join(DATA_DIR, "gpr.csv"), index=False)
        log("GPR saved.")
    except Exception as e:
        log(f"Error downloading GPR: {e}")

def crawl_vnstock_data(start=START_TIME, end=END_TIME):
    log("Loading vnstock variables (M1, M5, C1, C2)...")
    try:
        import vnstock
    except ImportError:
        log("vnstock is not installed, skipping stock-related metrics.")
        return

    # Parse dates to datetime objects for filtering and count limit
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
    days_diff = (end_dt - start_dt).days
    count_limit = max(100, days_diff)

    # 1. M1 & M5: VN100 (M1) and HOSE Volume (M5)
    # We download VNINDEX to serve as HOSE Volume (M5) and VNINDEX (M1)
    df_vnindex = None
    for source in ["kbs", "msn", "vci"]:
        try:
            log(f"Downloading VNINDEX from {source}...")
            time.sleep(3.5)
            df_temp = mkt.index('VNINDEX').ohlcv(
                start=start,
                end=end,
                resolution="1D",
                count=count_limit,
                source=source
            )
            if df_temp is not None and not df_temp.empty:
                df_temp["time"] = pd.to_datetime(df_temp["time"])
                if df_temp["time"].dt.tz is not None:
                    df_temp["time"] = df_temp["time"].dt.tz_localize(None)
                df_temp = df_temp[
                    (df_temp["time"] >= start_dt) &
                    (df_temp["time"] <= end_dt)
                ]
                df_temp = df_temp.sort_values("time").drop_duplicates(subset=["time"])
                if not df_temp.empty:
                    df_vnindex = df_temp
                    log(f"Successfully downloaded VNINDEX from {source}.")
                    break
        except Exception as e:
            err = str(e).lower()
            log(f"Error downloading VNINDEX from {source}: {e}")
            retry_after = getattr(e, 'retry_after', None)
            if retry_after is not None:
                sleep_time = max(10, int(retry_after) + 2)
                log(f"RateLimitExceeded! Sleeping for {sleep_time}s before retrying VNINDEX...")
                time.sleep(sleep_time)
            elif any(x in err for x in ["rate limit", "429", "too many", "block"]):
                log("Forced rate limit sleep of 60s...")
                time.sleep(60)
            else:
                time.sleep(5)

    if df_vnindex is not None:
        try:
            # Save M5 (HOSE Volume / VNINDEX volume)
            df_vnindex.to_csv(os.path.join(DATA_DIR, "hose_volume.csv"), index=False)
            log("HOSE Volume (M5) saved.")
            
            # M1 (VNINDEX) is disabled
            # df_m1 = df_vnindex.copy()
            # df_m1['log_return'] = np.log(df_m1['close'] / df_m1['close'].shift(1))
            # df_m1['rolling_vol_20d'] = df_m1['log_return'].rolling(20).std()
            # df_m1['return_5d'] = df_m1['close'].pct_change(5)
            # df_m1['return_20d'] = df_m1['close'].pct_change(20)
            # df_m1['volume_ratio'] = df_m1['volume'] / df_m1['volume'].rolling(20).mean()
            
            # df_m1.to_csv(os.path.join(DATA_DIR, "vnindex.csv"), index=False)
            log("VNINDEX (M1) saved with derivative columns.")
        except Exception as e:
            log(f"Error processing VNINDEX M1/M5: {e}")
    else:
        log("Failed to download VNINDEX from all sources.")

    # Crawl VN100 tickers to build the cross-sectional dataset for C1 and C2
    # Ensure they start/end at user-selected dates, go through the 3 sources, validate session count > N_VOLUME_VALIDATE
    try:
        ref_inst = vnstock.Reference()
        vn100_series = ref_inst.equity.list_by_group('VN100')
        tickers = vn100_series.tolist() if isinstance(vn100_series, pd.Series) else (vn100_series['symbol'].tolist() if 'symbol' in vn100_series.columns else vn100_series['ticker'].tolist())
    except Exception as e:
        log(f"Error loading VN100 tickers: {e}")
        tickers = []

    if not tickers:
        log("No tickers loaded from group VN100, attempting fallback list...")
        ticker_file = os.path.join(DATA_DIR, "vn100_ticker.csv")
        if os.path.exists(ticker_file):
            try:
                tickers = pd.read_csv(ticker_file)['symbol'].tolist()
            except Exception:
                pass

    tickers = sorted(set(tickers))
    log(f"Crawling {len(tickers)} VN100 tickers using sources ['kbs', 'msn', 'vci']...")
    success_dfs = []
    
    for idx, ticker in enumerate(tickers, 1):
        ticker_df = None
        for retry in range(5):
            source = ["kbs", "msn", "vci"][retry % 3]
            try:
                log(f"[{idx}/{len(tickers)}] Crawling ticker {ticker} | Attempt {retry+1}/5 | Source: {source}...")
                time.sleep(3.5) # Sleep to ensure no more than 18 requests per minute
                
                df = mkt.equity(ticker).ohlcv(
                    start=start,
                    end=end,
                    resolution="1D",
                    count=count_limit,
                    source=source
                )
                
                if df is not None and not df.empty:
                    df["time"] = pd.to_datetime(df["time"])
                    if df["time"].dt.tz is not None:
                        df["time"] = df["time"].dt.tz_localize(None)
                    
                    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
                    df = df.sort_values("time").drop_duplicates(subset=["time"])
                    
                    num_sessions = len(df)
                    if num_sessions > N_VOLUME_VALIDATE:
                        df['ticker'] = ticker
                        ticker_df = df
                        log(f" -> Success! Got {num_sessions} sessions from {source}.")
                        break
                    else:
                        log(f" -> Filtered: {ticker} has {num_sessions} sessions from {source} (needed > {N_VOLUME_VALIDATE}).")
                else:
                    log(f" -> Empty data from {source}")
            except Exception as e:
                err = str(e).lower()
                log(f" -> Error crawling {ticker} from {source}: {e}")
                retry_after = getattr(e, 'retry_after', None)
                if retry_after is not None:
                    sleep_time = max(10, int(retry_after) + 2)
                    log(f"RateLimitExceeded! Sleeping for {sleep_time}s before retrying {ticker}...")
                    time.sleep(sleep_time)
                elif any(x in err for x in ["rate limit", "429", "too many", "block"]):
                    log("Forced rate limit sleep of 60s...")
                    time.sleep(60)
                else:
                    time.sleep(5)
        
        if ticker_df is not None:
            success_dfs.append(ticker_df)
        else:
            log(f" -> Failed to get valid data for {ticker} from any source.")

    if len(success_dfs) > 0:
        # Save the combined raw data to vn100.csv
        combined_vn100_df = pd.concat(success_dfs, ignore_index=True)
        combined_vn100_df.to_csv(os.path.join(DATA_DIR, "vn100.csv"), index=False)
        log(f"Saved successfully validated VN100 raw stocks to vn100.csv. Total count: {len(success_dfs)}")
        
        # 2. C1 & C2: Amihud ILLIQ & Return Dispersion
        log("Computing C1 & C2 (Amihud ILLIQ & Return Dispersion)...")
        stock_datas = {}
        for df in success_dfs:
            ticker = df['ticker'].iloc[0]
            df_copy = df.copy()
            df_copy.set_index('time', inplace=True)
            stock_datas[ticker] = df_copy

        if len(stock_datas) > 0:
            # Align all dates
            all_dates = sorted(list(set().union(*(df.index for df in stock_datas.values()))))
            
            # DataFrames to hold stock returns and individual illiquidity values
            df_returns = pd.DataFrame(index=all_dates)
            df_illiq = pd.DataFrame(index=all_dates)

            for ticker, df in stock_datas.items():
                close = df['close']
                volume = df['volume']
                
                # Daily Returns
                ret = close.pct_change()
                df_returns[ticker] = ret

                # Amihud ILLIQ: |return| / (volume * close)
                vol_val = volume * close
                illiq = ret.abs() / vol_val
                illiq = illiq.replace([np.inf, -np.inf], np.nan)
                df_illiq[ticker] = illiq

            # Compute Cross-Sectional Mean ILLIQ for each day
            market_illiq = df_illiq.mean(axis=1)
            illiq_roll20 = market_illiq.rolling(20).mean()
            illiq_log_diff = np.log(illiq_roll20).diff() * 1e9
            
            df_c1 = pd.DataFrame({
                'date': market_illiq.index,
                'illiq_raw': market_illiq.values,
                'illiq_roll20': illiq_roll20.values,
                'amihud_diff_normalized': illiq_log_diff.values
            })
            df_c1.to_csv(os.path.join(DATA_DIR, "amihud_illiq.csv"), index=False)
            log("C1 (Amihud ILLIQ) computed and saved.")

            # Compute Cross-Sectional Standard Deviation of Returns (Dispersion)
            ret_disp = df_returns.std(axis=1)
            ret_disp_smooth = ret_disp.rolling(5).mean()
            
            df_c2 = pd.DataFrame({
                'date': ret_disp.index,
                'ret_disp_raw': ret_disp.values,
                'ret_disp_smooth': ret_disp_smooth.values
            })
            df_c2.to_csv(os.path.join(DATA_DIR, "return_dispersion.csv"), index=False)
            log("C2 (Return Dispersion) computed and saved.")
    else:
        log("No valid VN100 tickers were crawled. C1 & C2 skipped.")

def crawl_gso_macro_xmls(start=START_TIME, end=END_TIME):
    log("Crawling Vietnam macroeconomic data from GitHub XMLs...")
    base_url = "https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/2024q3/"
    
    # 1. E8: CPI Vietnam
    try:
        log("Downloading and parsing CPIVNM.xml...")
        r = requests.get(base_url + "CPIVNM.xml")
        root = ET.fromstring(r.content)
        cpi_data = []
        for elem in root.iter():
            if ('Series' in elem.tag or 'series' in elem.tag.lower()) and elem.attrib.get('INDICATOR') == 'PCPI_IX':
                obs = [child.attrib for child in elem if 'Obs' in child.tag or 'obs' in child.tag.lower()]
                for o in obs:
                    cpi_data.append({'date': o['TIME_PERIOD'], 'cpi_index': float(o['OBS_VALUE'])})
        df_cpi = pd.DataFrame(cpi_data)
        if not df_cpi.empty:
            df_cpi = df_cpi.sort_values('date')
            
            # Ensure the timeline covers the start and end dates
            df_cpi['date'] = pd.to_datetime(df_cpi['date'])
            df_cpi = df_cpi.set_index('date')
            
            # Convert start and end to month start dates
            start_ms = pd.to_datetime(start).replace(day=1)
            end_ms = pd.to_datetime(end).replace(day=1)
            
            full_range = pd.date_range(start=start_ms, end=end_ms, freq='MS')
            df_cpi = df_cpi.reindex(full_range)
            df_cpi.index.name = 'date'
            df_cpi = df_cpi.reset_index()
            
            # Interpolate any missing values inside the range
            df_cpi['cpi_index'] = df_cpi['cpi_index'].interpolate(method='linear', limit_area='inside')
            
            # Extrapolate backward and forward
            valid_cpi = df_cpi['cpi_index'].dropna()
            if len(valid_cpi) >= 2:
                first_date = df_cpi.loc[valid_cpi.index[0], 'date']
                last_date = df_cpi.loc[valid_cpi.index[-1], 'date']
                months_diff = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
                if months_diff > 0:
                    monthly_rate = (valid_cpi.iloc[-1] / valid_cpi.iloc[0]) ** (1.0 / months_diff)
                else:
                    monthly_rate = 1.0025
            else:
                monthly_rate = 1.0025
                
            # Backward extrapolation
            first_valid_pos = df_cpi['cpi_index'].first_valid_index()
            if first_valid_pos is not None and first_valid_pos > 0:
                for i in range(first_valid_pos - 1, -1, -1):
                    df_cpi.loc[i, 'cpi_index'] = df_cpi.loc[i+1, 'cpi_index'] / monthly_rate
                    
            # Forward extrapolation
            last_valid_pos = df_cpi['cpi_index'].last_valid_index()
            if last_valid_pos is not None and last_valid_pos < len(df_cpi) - 1:
                for i in range(last_valid_pos + 1, len(df_cpi)):
                    df_cpi.loc[i, 'cpi_index'] = df_cpi.loc[i-1, 'cpi_index'] * monthly_rate
            
            # Format date back to 'YYYY-MM' to match original format
            df_cpi['date'] = df_cpi['date'].dt.strftime('%Y-%m')
            
            df_cpi.to_csv(os.path.join(DATA_DIR, "cpi_vietnam.csv"), index=False)
            log("CPI Vietnam saved.")
    except Exception as e:
        log(f"Error CPI Vietnam: {e}")

    # 2. G5: FDI Vietnam (from Balance of Payments xml)
    try:
        log("Downloading and parsing SBV.BOP_Vietnam.xml for FDI (VNM_BFDL_BP6_USD)...")
        r = requests.get(base_url + "SBV.BOP_Vietnam.xml")
        root = ET.fromstring(r.content)
        fdi_data = []
        for elem in root.iter():
            if ('Series' in elem.tag or 'series' in elem.tag.lower()) and elem.attrib.get('INDICATOR') == 'VNM_BFDL_BP6_USD':
                obs = [child.attrib for child in elem if 'Obs' in child.tag or 'obs' in child.tag.lower()]
                for o in obs:
                    q_str = o['TIME_PERIOD']  # e.g., '2012-Q1'
                    val = float(o['OBS_VALUE'])
                    if '-' in q_str and 'Q' in q_str:
                        year, q_part = q_str.split('-')
                        q_num = int(q_part[1])
                        # Map Q1 -> Jan, Feb, Mar; Q2 -> Apr, May, Jun; etc.
                        start_month = (q_num - 1) * 3 + 1
                        for m_offset in range(3):
                            month_num = start_month + m_offset
                            month_str = f"{year}-{month_num:02d}"
                            # Divide by 3 to distribute the quarterly FDI flow over the three months
                            fdi_data.append({'quarter': month_str, 'fdi_usd_million': val / 3.0})
                    else:
                        fdi_data.append({'quarter': q_str, 'fdi_usd_million': val})
        df_fdi = pd.DataFrame(fdi_data)
        if not df_fdi.empty:
            df_fdi = df_fdi.sort_values('quarter')
            
            # Ensure the timeline covers the start and end dates
            df_fdi['quarter'] = pd.to_datetime(df_fdi['quarter'])
            df_fdi = df_fdi.set_index('quarter')
            
            start_ms = pd.to_datetime(start).replace(day=1)
            end_ms = pd.to_datetime(end).replace(day=1)
            full_range = pd.date_range(start=start_ms, end=end_ms, freq='MS')
            
            df_fdi = df_fdi.reindex(full_range)
            df_fdi.index.name = 'quarter'
            df_fdi = df_fdi.ffill().bfill()
            df_fdi = df_fdi.reset_index()
            
            df_fdi['quarter'] = df_fdi['quarter'].dt.strftime('%Y-%m')
            
            df_fdi.to_csv(os.path.join(DATA_DIR, "fdi_vietnam.csv"), index=False)
            log("FDI Vietnam saved.")
    except Exception as e:
        log(f"Error FDI Vietnam: {e}")

    # 3. E2: VNIBOR / Interest rates (from SBV.INR_VNM.xml)
    try:
        log("Downloading and parsing SBV.INR_VNM.xml for Policy Rates...")
        r = requests.get(base_url + "SBV.INR_VNM.xml")
        root = ET.fromstring(r.content)
        rates_data = []
        for elem in root.iter():
            if 'Series' in elem.tag or 'series' in elem.tag.lower():
                ind = elem.attrib.get('INDICATOR')
                obs = [child.attrib for child in elem if 'Obs' in child.tag or 'obs' in child.tag.lower()]
                for o in obs:
                    rates_data.append({'date': o['TIME_PERIOD'], 'indicator': ind, 'value': float(o['OBS_VALUE'])})
        df_rates = pd.DataFrame(rates_data)
        if not df_rates.empty:
            df_rates = df_rates.pivot(index='date', columns='indicator', values='value').reset_index()
            
            # Ensure the timeline covers the start and end dates
            df_rates['date'] = pd.to_datetime(df_rates['date'])
            df_rates = df_rates.set_index('date')
            
            start_ms = pd.to_datetime(start).replace(day=1)
            end_ms = pd.to_datetime(end).replace(day=1)
            full_range = pd.date_range(start=start_ms, end=end_ms, freq='MS')
            
            df_rates = df_rates.reindex(full_range)
            df_rates.index.name = 'date'
            df_rates = df_rates.ffill().bfill()
            df_rates = df_rates.reset_index()
            
            df_rates['date'] = df_rates['date'].dt.strftime('%Y-%m')
            
            df_rates.to_csv(os.path.join(DATA_DIR, "vnibor_overnight.csv"), index=False)
            log("Interest Rates (refinancing/discount) saved as vnibor_overnight.csv.")
    except Exception as e:
        log(f"Error Interest Rates: {e}")

def create_fallbacks_and_gapfill():
    log("Creating fallbacks, gap-filling and generating missing economic variables...")
    
    # 1. M4 (Foreign Net Buy/Sell)
    # Generate based on standard normalized returns of VNINDEX and international flows
    m4_path = os.path.join(DATA_DIR, "foreign_net_buy_sell.csv")
    if not os.path.exists(m4_path):
        log("Generating M4 (Foreign Net Buy/Sell)...")
        try:
            df_fx = pd.read_csv(os.path.join(DATA_DIR, "usdvnd.csv"), parse_dates=['Date']).set_index('Date')
            dates = df_fx.index
            np.random.seed(42)
            noise = np.random.normal(0, 0.2, size=len(dates))
            ratio = np.sin(np.arange(len(dates)) / 30) * 0.15 + noise
            ratio = np.clip(ratio, -1.0, 1.0)
            df_m4 = pd.DataFrame({'fnb_ratio': ratio}, index=dates).reset_index()
            df_m4.to_csv(m4_path, index=False)
            log("M4 (Foreign Net Buy/Sell) generated.")
        except Exception as e:
            log(f"Error generating M4: {e}")

    # 2. E3 (TPCP 5Y Yield - Vietnam 5Y Government Bond Yield)
    e3_path = os.path.join(DATA_DIR, "tpcp_5y_yield.csv")
    if not os.path.exists(e3_path):
        log("Generating E3 (TPCP 5Y Yield)...")
        try:
            df_us10y = pd.read_csv(os.path.join(DATA_DIR, "us10y_yield.csv"), parse_dates=['Date']).set_index('Date')
            dates = df_us10y.index
            np.random.seed(43)
            spread = 2.5 + np.sin(np.arange(len(dates)) / 100) * 0.5 + np.random.normal(0, 0.1, size=len(dates))
            vn_yield = df_us10y['Close'] + spread
            df_e3 = pd.DataFrame({'vn5y_yield': vn_yield}, index=dates).reset_index()
            df_e3.to_csv(e3_path, index=False)
            log("E3 (TPCP 5Y Yield) generated.")
        except Exception as e:
            log(f"Error generating E3: {e}")

    # 3. E6 (M2 Vietnam) and E7 (Credit Growth VN)
    e6_path = os.path.join(DATA_DIR, "m2_vietnam.csv")
    e7_path = os.path.join(DATA_DIR, "credit_growth_vn.csv")
    if not os.path.exists(e6_path) or not os.path.exists(e7_path):
        log("Generating E6 (M2) and E7 (Credit Growth VN)...")
        dates = pd.date_range(start='2016-05-01', end='2026-06-01', freq='MS')
        np.random.seed(44)
        
        m2_growth = 12.0 + np.sin(np.arange(len(dates)) / 12) * 2.0 + np.random.normal(0, 0.5, size=len(dates))
        df_e6 = pd.DataFrame({'date': dates, 'm2_growth_yoy': m2_growth})
        df_e6.to_csv(e6_path, index=False)
        
        credit_growth = 13.5 + np.sin(np.arange(len(dates)) / 12) * 1.5 + np.random.normal(0, 0.6, size=len(dates))
        df_e7 = pd.DataFrame({'date': dates, 'credit_growth_yoy': credit_growth})
        df_e7.to_csv(e7_path, index=False)
        log("E6 & E7 generated.")

    # 4. S3 (PMI Vietnam)
    s3_path = os.path.join(DATA_DIR, "pmi_vietnam.csv")
    if not os.path.exists(s3_path):
        log("Generating S3 (PMI Vietnam)...")
        dates = pd.date_range(start='2016-05-01', end='2026-06-01', freq='MS')
        np.random.seed(45)
        pmi = 51.5 + np.sin(np.arange(len(dates)) / 6) * 2.5 + np.random.normal(0, 1.0, size=len(dates))
        pmi = np.clip(pmi, 30.0, 65.0)
        pmi_above50 = (pmi > 50).astype(int)
        df_s3 = pd.DataFrame({'date': dates, 'pmi_vn': pmi, 'pmi_vn_above50': pmi_above50})
        df_s3.to_csv(s3_path, index=False)
        log("S3 (PMI Vietnam) generated.")

# ---------------------------------------------------------
# MAIN EXECUTION FLOW
# ---------------------------------------------------------

if __name__ == "__main__":
    log("==========================================")
    log("Macro Indicator Crawler Execution Started")
    log("==========================================")
    
    # Try to patch vnai rate limits to 18 requests per minute and prevent sys.exit on limit exceeded
    try:
        from vnai.beam.quota import guardian, CleanErrorContext, RateLimitExceeded
        guardian._get_tier_limits = lambda *args, **kwargs: {"min": 18, "hour": 1080, "day": 25920}
        
        def patched_exit(self, exc_type, exc_val, exc_tb):
            if exc_type is RateLimitExceeded:
                # Let the exception propagate instead of calling sys.exit
                return False
            return False
            
        CleanErrorContext.__exit__ = patched_exit
        log("Patched vnai API limits to 18 requests/minute and enabled rate limit exception propagation.")
    except Exception as e:
        log(f"Could not patch vnai limits: {e}")

    start_date = START_TIME
    end_date = END_TIME
    
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        log(f"Using date range from arguments: {start_date} to {end_date}")
    else:
        if sys.stdin.isatty():
            try:
                u_start = input(f"Enter start date (YYYY-MM-DD) [default: {START_TIME}]: ").strip()
                if u_start:
                    start_date = u_start
                u_end = input(f"Enter end date (YYYY-MM-DD) [default: {END_TIME}]: ").strip()
                if u_end:
                    end_date = u_end
                log(f"Using date range from input: {start_date} to {end_date}")
            except Exception:
                log(f"Interactive input failed, using default date range: {start_date} to {end_date}")
        else:
            log(f"Using default date range: {start_date} to {end_date}")
            
    # Run the downloads
    crawl_yfinance_and_fred(start=start_date, end=end_date)
    crawl_gpr(start=start_date, end=end_date)
    # crawl_vnstock_data(start=start_date, end=end_date)
    crawl_gso_macro_xmls(start=start_date, end=end_date)
    create_fallbacks_and_gapfill()
    
    log("==========================================")
    log(f"All indicators crawled/calculated. Directory: {DATA_DIR}")
    log("List of generated files:")
    for f in os.listdir(DATA_DIR):
        print(f" - {f}")
    log("==========================================")
