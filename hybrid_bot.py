# hybrid_bot_live_v5.py
# FINAL SCRIPT: Updated with robust Timedelta calculation fix for Pandas datetime error.
# Requirements: kiteconnect, pandas, numpy, pytz

import logging
import time
import math
import sys
from datetime import datetime, timedelta, date
import pytz
import numpy as np
import pandas as pd
from kiteconnect import KiteConnect

# --------------------------------------------------------------------------
# 🛑 CRITICAL: CONFIGURATION 🛑
# --------------------------------------------------------------------------

# Keys provided by the user:
API_KEY = "5vr1pnen7qi6nw1q"
API_SECRET = "ev69pkdtddzor7n9y9sbe32ww0ovk02o" 

# Token will be fetched at runtime.
ACCESS_TOKEN = None


INDIAN_TIMEZONE = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = datetime.strptime("09:15", "%H:%M").time()
MARKET_CLOSE = datetime.strptime("15:30", "%H:%M").time()

INDEX_SYMBOL_KITE = "NSE:NIFTY 50"
SYMBOL_PREFIX = "NIFTY"

NUM_LOTS = 1
LOT_SIZE = 75

# Delta target range
DELTA_MIN = 0.25
DELTA_MAX = 0.30

# Wave / Survivor thresholds
ADX_THRESHOLD = 40
WAVE_GAP = 15
SURVIVOR_BREAKOUT = 40

# Order settings
PLACE_MARKET_FOR_SURVIVOR = True
FALLBACK_IV = 0.20

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HybridOptionsNextTueBot")

# --------------------------------------------------------------------------
# ✅ AUTHENTICATION FLOW
# --------------------------------------------------------------------------

def get_live_access_token():
    logger.warning("-------------------------------------------------------")
    logger.warning("🔑 DAILY LOGIN REQUIRED (Kite Connect Access Token)")
    logger.warning("-------------------------------------------------------")

    # 1. Provide Login URL
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={API_KEY}"
    print("\nSTEP 1: GET REQUEST TOKEN")
    print(f"1. Open this URL in your browser and log in: {login_url}")
    print("2. After successful login, you will be redirected to your Redirect URL.")
    print("3. Copy the 'request_token' from the address bar of the redirected URL.")
    print("   (e.g., ...?status=success&request_token=xyz123abc456...)\n")

    # 2. Get Request Token from User
    request_token = input("STEP 2: Paste the 'request_token' here (or press Ctrl+C to exit): ")

    # 3. Exchange Tokens
    try:
        kite_temp = KiteConnect(api_key=API_KEY)
        session_data = kite_temp.generate_session(request_token, api_secret=API_SECRET)
        new_access_token = session_data.get("access_token")
        username = session_data.get("user_name")

        if not new_access_token:
            logger.error(f"Token generation failed. Response: {session_data}")
            sys.exit(1)

        logger.info(f"✅ SUCCESS! Logged in as: {username}. New ACCESS_TOKEN obtained.")
        logger.warning(f"Note: This token is valid until {session_data.get('login_time').date() + timedelta(days=1)} (~7:00 AM IST)")
        return new_access_token

    except Exception as e:
        logger.error(f"Token exchange failed. Check your API_SECRET or request_token: {e}")
        sys.exit(1)

# --------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------

def now_ist():
    # Returns normalized (midnight) IST datetime object
    return datetime.now(INDIAN_TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)

def market_open_now():
    t = datetime.now(INDIAN_TIMEZONE).time()
    return MARKET_OPEN <= t <= MARKET_CLOSE

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(spot, strike, rate_annual, days_to_expiry, vol_annual, option_type="CE"):
    if days_to_expiry <= 0:
        call_delta = 1.0 if spot > strike else 0.0
        return call_delta if option_type == "CE" else (call_delta - 1.0)
    T = max(days_to_expiry / 365.0, 1e-6)
    sigma = max(vol_annual, 1e-6)
    try:
        d1 = (math.log(spot / strike) + (rate_annual + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    except Exception:
        return 0.0 if option_type == "CE" else 0.0
    call_delta = norm_cdf(d1)
    return call_delta if option_type == "CE" else (call_delta - 1.0)

def annualized_vol_from_histogram(prices):
    try:
        logrets = np.log(prices / prices.shift(1)).dropna()
        if len(logrets) < 2:
            return FALLBACK_IV
        daily_sigma = np.std(logrets, ddof=1)
        annual_sigma = daily_sigma * math.sqrt(252)
        return float(max(annual_sigma, 0.01))
    except Exception as e:
        logger.warning(f"Vol estimation failed: {e}, using fallback {FALLBACK_IV}")
        return FALLBACK_IV

# --------------------------------------------------------------------------
# KITE HELPERS
# --------------------------------------------------------------------------

def kite_init(token):
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(token)
    return kite

def fetch_all_nfo_instruments(kite):
    try:
        instruments = kite.instruments("NFO")
        df = pd.DataFrame(instruments)
        if not df.empty and 'expiry' in df.columns:
            # FIX 1: Enforce conversion to datetime; 'coerce' turns bad dates into NaT
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df['expiry'] = df['expiry'].dt.tz_localize(None) # Remove time zone for consistency
        return df
    except Exception as e:
        logger.error(f"Failed to fetch instruments: {e}")
        return pd.DataFrame()

def fetch_spot_ltp(kite):
    try:
        l = kite.ltp([INDEX_SYMBOL_KITE])
        if isinstance(l, dict) and INDEX_SYMBOL_KITE in l:
            return float(l[INDEX_SYMBOL_KITE]["last_price"])
        for v in l.values():
            return float(v["last_price"])
    except Exception as e:
        if "Invalid `api_key` or `access_token`" in str(e):
             logger.error(f"Failed to fetch spot LTP: {e}")
             raise e
        logger.error(f"Failed to fetch spot LTP: {e}")
    return None

def fetch_historical_spot_close(kite, from_date, to_date):
    try:
        data = kite.historical(INDEX_SYMBOL_KITE, from_date, to_date, "day")
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        return df['close']
    except Exception as e:
        logger.warning(f"Failed to fetch historical spot: {e}")
        return pd.Series(dtype=float)

def ltp_for_symbols(kite, tradingsymbols):
    results = {}
    try:
        keys = [f"NFO:{s}" for s in tradingsymbols]
        resp = kite.ltp(keys)
        for k, v in resp.items():
            symbol = k.split(":", 1)[1]
            last_price = float(v.get("last_price", 0.0))
            oi = float(v.get("oi", 0.0))
            results[symbol] = {"last_price": last_price, "oi": oi}
    except Exception as e:
        logger.warning(f"Error fetching LTP for symbols: {e}")
    return results

def place_limit_order(kite, tradingsymbol, qty, price, transaction_type="BUY", product="NRML", variety="regular"):
    try:
        order = kite.place_order(
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            transaction_type=transaction_type,
            quantity=qty,
            order_type="LIMIT",
            product=product,
            price=price,
            variety=variety
        )
        logger.info(f"Placed LIMIT {transaction_type} order {order} {tradingsymbol} qty {qty} @ {price}")
        return order
    except Exception as e:
        logger.error(f"Limit order failed for {tradingsymbol}: {e}")
        return None

def place_market_order(kite, tradingsymbol, qty, transaction_type="SELL", product="NRML", variety="regular"):
    try:
        order = kite.place_order(
            tradingsymbol=tradingsymbol,
            exchange="NFO",
            transaction_type=transaction_type,
            quantity=qty,
            order_type="MARKET",
            product=product,
            variety=variety
        )
        logger.info(f"Placed MARKET {transaction_type} order {order} {tradingsymbol} qty {qty}")
        return order
    except Exception as e:
        logger.error(f"Market order failed for {tradingsymbol}: {e}")
        return None

# --------------------------------------------------------------------------
# WEEKLY EXPIRY / CHAIN SELECTION
# --------------------------------------------------------------------------

def compute_next_tuesday_after_today():
    today = now_ist().date() # Use normalized date
    days_ahead = (1 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)

def find_next_weekly_expiry_date(df_instruments):
    if df_instruments is None or df_instruments.empty:
        return None
    df = df_instruments.copy()
    if 'expiry' not in df.columns:
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
    df['expiry_date'] = df['expiry'].dt.date
    target = compute_next_tuesday_after_today()
    if target in set(df['expiry_date'].unique()):
        return target
    candidates = sorted([d for d in df['expiry_date'].unique() if d > target])
    return candidates[0] if candidates else None

def find_weekly_chain_for_expiry(df_instruments, expiry_date):
    df = df_instruments.copy()
    if 'expiry' not in df.columns:
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
    df['expiry_date'] = df['expiry'].dt.date
    chain = df[
        (df['name'] == SYMBOL_PREFIX) &
        (df['instrument_type'].isin(['CE', 'PE'])) &
        (df['expiry_date'] == expiry_date)
    ].copy()
    return chain

def select_option_by_delta_and_oi(kite, chain_df, spot, target_delta_low=DELTA_MIN, target_delta_high=DELTA_MAX, rate_annual=0.07):
    if chain_df is None or chain_df.empty:
        return None, None, {}
    chain = chain_df.copy()
    
    # NEW FINAL FIX: FORCE EXPIRY CONVERSION, normalize date components, then calculate Timedelta.
    chain['expiry'] = pd.to_datetime(chain['expiry'], errors='coerce')
    chain = chain[pd.notna(chain['expiry'])].copy()
    
    # Calculate Timedelta (difference) between normalized expiry date and normalized today's date
    delta = chain['expiry'].dt.normalize() - now_ist()
    # Extract total days from the Timedelta object
    chain['days_to_expiry'] = delta.dt.days
    
    strikes = sorted(chain['strike'].unique())
    end = now_ist().date()
    start = end - timedelta(days=30)
    hist = fetch_historical_spot_close(kite, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    vol_est = annualized_vol_from_histogram(hist) if not hist.empty else FALLBACK_IV
    symbols_to_query = chain['tradingsymbol'].unique().tolist()
    ltp_map = ltp_for_symbols(kite, symbols_to_query)
    ce_candidates = []
    pe_candidates = []
    for strike in strikes:
        row_ce = chain[(chain['strike'] == strike) & (chain['instrument_type'] == 'CE')]
        row_pe = chain[(chain['strike'] == strike) & (chain['instrument_type'] == 'PE')]
        if not row_ce.empty:
            sym = row_ce['tradingsymbol'].iloc[0]
            dte = int(row_ce['days_to_expiry'].iloc[0])
            delta = bs_delta(spot, strike, rate_annual, dte, vol_est, option_type="CE")
            oi = ltp_map.get(sym, {}).get("oi", 0.0)
            last = ltp_map.get(sym, {}).get("last_price", 0.0)
            ce_candidates.append({"symbol": sym, "strike": strike, "delta": delta, "oi": oi, "ltp": last, "dte": dte})
        if not row_pe.empty:
            sym = row_pe['tradingsymbol'].iloc[0]
            dte = int(row_pe['days_to_expiry'].iloc[0])
            delta = abs(bs_delta(spot, strike, rate_annual, dte, vol_est, option_type="PE"))
            oi = ltp_map.get(sym, {}).get("oi", 0.0)
            last = ltp_map.get(sym, {}).get("last_price", 0.0)
            pe_candidates.append({"symbol": sym, "strike": strike, "delta": delta, "oi": oi, "ltp": last, "dte": dte})
    def choose_best(cands):
        if not cands:
            return None
        within = [c for c in cands if target_delta_low <= c['delta'] <= target_delta_high]
        pool = within if within else cands
        target_mid = (target_delta_low + target_delta_high) / 2.0
        pool_sorted = sorted(pool, key=lambda x: (abs(x['delta'] - target_mid), -x['oi']))
        return pool_sorted[0]
    best_ce = choose_best(ce_candidates)
    best_pe = choose_best(pe_candidates)
    meta = {"vol_est": vol_est, "ce_candidates": ce_candidates, "pe_candidates": pe_candidates}
    return (best_ce['symbol'] if best_ce else None), (best_pe['symbol'] if best_pe else None), meta

# --------------------------------------------------------------------------
# STRATEGY HOOKS
# --------------------------------------------------------------------------

def run_wave_strategy(kite, price, symbol, qty):
    target_sell_price = round(max(1, price + WAVE_GAP), 2)
    logger.info(f"🌊 [WAVE ACTIVE] GRID - SELL LIMIT {symbol} qty {qty} @ {target_sell_price}")
    place_limit_order(kite, tradingsymbol=symbol, qty=qty, price=target_sell_price, transaction_type="SELL")
    target_buy_price = round(max(1, price - WAVE_GAP), 2)
    logger.info(f"🌊 [WAVE ACTIVE] GRID - BUY LIMIT {symbol} qty {qty} @ {target_buy_price}")
    place_limit_order(kite, tradingsymbol=symbol, qty=qty, price=target_buy_price, transaction_type="BUY")

def run_survivor_strategy(kite, price, symbol, qty):
    if PLACE_MARKET_FOR_SURVIVOR:
        logger.warning(f"🚀 [SURVIVOR ACTIVE] TREND - MARKET SELL {symbol} qty {qty}")
        place_market_order(kite, tradingsymbol=symbol, qty=qty, transaction_type="SELL")
    else:
        limit_price = round(max(1, price - 0.5), 2)
        logger.warning(f"🚀 [SURVIVOR ACTIVE] TREND - SELL LIMIT {symbol} qty {qty} @ {limit_price}")
        place_limit_order(kite, tradingsymbol=symbol, qty=qty, price=limit_price, transaction_type="SELL")

# --------------------------------------------------------------------------
# HYBRID SUPERVISOR CLASS
# --------------------------------------------------------------------------

class HybridSupervisorOptions:
    def __init__(self, token):
        logger.info("🔑 Init kite...")
        self.kite = kite_init(token)
        self.chain_df = pd.DataFrame()
        self.ce_symbol = None
        self.pe_symbol = None
        self.chain_expiry = None
        self.vol_est = None
        self.qty = int(NUM_LOTS * LOT_SIZE)
        self.roll_to_next_tuesday()

    def roll_to_next_tuesday(self):
        df = fetch_all_nfo_instruments(self.kite)
        if df.empty:
            logger.error("No NFO instruments fetched.")
            return
        target_expiry = find_next_weekly_expiry_date(df)
        if not target_expiry:
            logger.error("Could not compute next weekly expiry (Tuesday).")
            return
        chain = find_weekly_chain_for_expiry(df, target_expiry)
        if chain.empty:
            future_dates = sorted([d for d in df['expiry'].dt.date.unique() if d > target_expiry])
            if not future_dates:
                logger.error("No future weekly expiries found.")
                return
            fallback = future_dates[0]
            chain = find_weekly_chain_for_expiry(df, fallback)
            if chain.empty:
                logger.error("Fallback chain empty.")
                return
            target_expiry = fallback

        spot = fetch_spot_ltp(self.kite)
        if spot is None:
            logger.error("Spot LTP fetch failed. Cannot pick strikes.")
            return

        ce, pe, meta = select_option_by_delta_and_oi(self.kite, chain, spot)
        if ce and pe:
            self.chain_df = chain
            self.ce_symbol = ce
            self.pe_symbol = pe
            self.chain_expiry = pd.to_datetime(chain['expiry'].iloc[0]).date()
            self.vol_est = meta.get("vol_est", FALLBACK_IV)
            logger.warning(f"🔄 OPTION ROLLED (next Tue): CE={self.ce_symbol}, PE={self.pe_symbol}, expiry={self.chain_expiry}, vol_est={self.vol_est:.2%}")
        else:
            logger.error("Failed to pick CE/PE for next-Tuesday chain.")

    def maybe_roll_if_needed(self):
        if self.chain_df is None or self.chain_df.empty or self.chain_expiry is None:
            self.roll_to_next_tuesday()
            return
        today = now_ist().date()
        if self.chain_expiry <= today:
            logger.info("Current chain expiry is today or passed -> rolling to next week's Tuesday.")
            self.roll_to_next_tuesday()
            return

    def decide(self):
        if not market_open_now():
            logger.info("💤 Market closed. waiting for open.")
            return

        self.maybe_roll_if_needed()

        if not self.ce_symbol or not self.pe_symbol:
            logger.warning("CE/PE not selected yet; attempting roll.")
            self.roll_to_next_tuesday()
            if not self.ce_symbol or not self.pe_symbol:
                logger.error("No CE/PE available after roll. Skipping this tick.")
                return

        spot = fetch_spot_ltp(self.kite)
        if spot is None:
            logger.warning("Spot missing. skip.")
            return

        try:
            median_strike = float(self.chain_df['strike'].median())
        except Exception:
            median_strike = spot

        chosen = self.ce_symbol if spot >= median_strike else self.pe_symbol

        end = now_ist().date()
        start = end - timedelta(days=10)
        hist = fetch_historical_spot_close(self.kite, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        short_vol = np.std(np.log(hist / hist.shift(1)).dropna()) * math.sqrt(252) if not hist.empty else self.vol_est

        is_trend = (short_vol * 100) >= ADX_THRESHOLD or short_vol >= 0.30
        logger.info(f"Deciding mode: short_vol={short_vol:.2%} -> {'SURVIVOR' if is_trend else 'WAVE'}")

        if is_trend:
            run_survivor_strategy(self.kite, spot, chosen, self.qty)
        else:
            run_wave_strategy(self.kite, spot, chosen, self.qty)

    def run_loop(self, sleep_sec=15):
        logger.info("🤖 Hybrid Options Bot (next-Tue) Ready. Entering main loop.")
        while True:
            try:
                self.decide()
                time.sleep(sleep_sec)
            except KeyboardInterrupt:
                logger.info("User requested shutdown.")
                break
            except Exception as e:
                logger.exception(f"Error in main loop: {e}")
                time.sleep(5)

# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Get a fresh, valid ACCESS_TOKEN by forcing the user interaction
    final_token = get_live_access_token()

    # 2. Start the bot with the valid token
    bot = HybridSupervisorOptions(final_token)
    bot.run_loop(sleep_sec=15)