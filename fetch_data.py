import logging
from kiteconnect import KiteConnect
import pandas as pd
import datetime

# --- YOUR CREDENTIALS ---
api_key = "jnte5vc1eukpl9ex"
access_token = "XvhNE0d7MKA0JMdQk9qk82xyau6yLnl3"

# Initialize Kite
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def get_current_nifty_future_token():
    print("🔍 Searching for the current Nifty Futures token...")
    # Download all NFO instruments
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    
    # Filter for NIFTY Futures
    nifty_futs = df[
        (df["name"] == "NIFTY") & 
        (df["instrument_type"] == "FUT")
    ]
    
    # Sort by expiry date to get the nearest one (Current Month)
    nifty_futs = nifty_futs.sort_values(by="expiry")
    
    # Get the first one (Nearest Expiry)
    current_future = nifty_futs.iloc[0]
    print(f"✅ Found Contract: {current_future['tradingsymbol']} (ID: {current_future['instrument_token']})")
    return current_future['instrument_token']

def fetch_data():
    try:
        # 1. Get the correct token automatically
        token = get_current_nifty_future_token()
        
        # 2. Define dates (Last 5 days)
        to_date = datetime.datetime.now().date()
        from_date = to_date - datetime.timedelta(days=5)
        
        print(f"📥 Downloading data from {from_date} to {to_date}...")
        
        # 3. Fetch Minute Data
        records = kite.historical_data(
            token, 
            from_date, 
            to_date, 
            "minute"
        )
        
        # 4. Save to CSV
        df = pd.DataFrame(records)
        if not df.empty:
            # Rename 'date' to 'timestamp' to match our backtester
            df = df.rename(columns={"date": "timestamp"})
            # Keep only what we need
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            filename = "nifty_real_data.csv"
            df.to_csv(filename, index=False)
            print(f"\nSUCCESS! 🚀")
            print(f"Saved {len(df)} candles to '{filename}'")
        else:
            print("⚠️ No data found. (Market might be closed or holiday?)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fetch_data()
