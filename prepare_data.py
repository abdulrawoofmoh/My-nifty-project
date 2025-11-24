import pandas as pd
import glob
import os

def clean_and_save():
    # 1. Find the file automatically (looks for any CSV starting with NIFTY)
    files = glob.glob("NIFTY*.csv")
    
    # Filter out the target file if it already exists so we don't read it by mistake
    files = [f for f in files if "real_data" not in f]
    
    if not files:
        print("❌ Could not find the Nifty CSV file. Make sure it is in this folder!")
        return

    input_filename = files[0] # Pick the first one found
    print(f"📂 Found file: {input_filename}")
    
    # 2. Load Data
    df = pd.read_csv(input_filename)
    
    # 3. Clean Column Names (Date -> timestamp, Close -> close)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'date': 'timestamp'})
    
    # 4. Clean Date Format
    # Turns "Fri Nov 21 2025 11:59:00 GMT+0300..." into "2025-11-21 11:59:00"
    try:
        # Split by " GMT" to remove the timezone junk at the end
        df['timestamp'] = df['timestamp'].astype(str).apply(lambda x: x.split(' GMT')[0])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %Y %H:%M:%S')
    except Exception as e:
        print(f"⚠️ Warning: Could not auto-format dates. keeping original. ({e})")

    # 5. Save as 'nifty_real_data.csv'
    output_filename = "nifty_real_data.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"✅ SUCCESS! Converted {len(df)} rows.")
    print(f"💾 Saved to: {output_filename}")
    print("🚀 You are ready to run the backtest!")

if __name__ == "__main__":
    clean_and_save()
