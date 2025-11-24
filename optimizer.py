import logging
import pandas as pd
import numpy as np
import itertools
import os
import glob

# --- SETTINGS TO TEST ---
# We will test ALL combinations of these numbers
PARAM_GRID = {
    'wave_gap': [15, 20, 25, 30, 35, 40],        # Test wider gaps
    'adx_threshold': [20, 25, 30, 40],           # Test different trend sensitivities
    'survivor_breakout': [20, 30, 40]            # Test stricter breakouts
}

FILENAME = "data.csv"

# --- ENGINE (Simplified for Speed) ---
def calculate_adx(df, period=14):
    df = df.copy()
    df['h-l'] = df['High'] - df['Low']
    df['h-c'] = abs(df['High'] - df['Close'].shift(1))
    df['l-c'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['h-l', 'h-c', 'l-c']].max(axis=1)
    df['up'] = df['High'] - df['High'].shift(1)
    df['down'] = df['Low'].shift(1) - df['Low']
    df['pos_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['neg_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    df['tr_s'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['pos_di'] = 100 * (df['pos_dm'].ewm(alpha=1/period, adjust=False).mean() / df['tr_s'])
    df['neg_di'] = 100 * (df['neg_dm'].ewm(alpha=1/period, adjust=False).mean() / df['tr_s'])
    df['dx'] = 100 * abs(df['pos_di'] - df['neg_di']) / (df['pos_di'] + df['neg_di'])
    return df['dx'].ewm(alpha=1/period, adjust=False).mean()

def run_single_backtest(df, wave_gap, adx_thresh, survivor_brk):
    cash = 1000000.0
    position = 0
    avg_price = 0.0
    active_regime = "NEUTRAL"
    
    # Strategy Memory
    wave_buy = None
    wave_sell = None
    surv_ref = None
    
    qty = 50
    
    for i, row in df.iterrows():
        price = row['Close']
        adx = row['adx']
        oi_chg = row['oi_pct']
        
        # 1. SUPERVISOR
        is_trending = (adx > adx_thresh) or (oi_chg < -5)
        new_regime = "SURVIVOR" if is_trending else "WAVE"
        
        if new_regime != active_regime:
            active_regime = new_regime
            wave_buy = price - wave_gap
            wave_sell = price + wave_gap
            surv_ref = price

        # 2. EXECUTE
        if active_regime == "WAVE":
            # Buy Dip
            if price <= wave_buy:
                cash -= price * qty
                position += qty
                wave_buy = price - wave_gap
                wave_sell = price + wave_gap
            # Sell Rip
            elif price >= wave_sell:
                cash += price * qty
                position -= qty
                wave_buy = price - wave_gap
                wave_sell = price + wave_gap
                
            # Trail
            if price > wave_sell: wave_sell=price+wave_gap; wave_buy=price-wave_gap
            if price < wave_buy: wave_buy=price-wave_gap; wave_sell=price+wave_gap

        elif active_regime == "SURVIVOR":
            # Buy Breakout
            if price > (surv_ref + survivor_brk):
                if position <= 0:
                    buy_qty = qty if position == 0 else qty*2
                    cash -= price * buy_qty
                    position += buy_qty
                surv_ref = price
            # Sell Breakdown
            elif price < (surv_ref - survivor_brk):
                if position >= 0:
                    sell_qty = qty if position == 0 else qty*2
                    cash += price * sell_qty
                    position -= sell_qty
                surv_ref = price

    # Final P&L
    final_val = cash + (position * df.iloc[-1]['Close'])
    return final_val

# --- MAIN OPTIMIZER ---
if __name__ == "__main__":
    print("⏳ Loading Data...")
    
    # Find File
    if os.path.exists(FILENAME): filename = FILENAME
    else: filename = max(glob.glob("NIFTY*.csv"), key=os.path.getsize)
    
    # Load
    df = pd.read_csv(filename)
    df.columns = [c.strip() for c in df.columns] # Clean names
    # Standardize names
    col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open','high','low','close']}
    df = df.rename(columns=col_map)
    if 'Date' in df.columns: df = df.rename(columns={'Date': 'timestamp'})
    
    # Handle OI
    oi_col = next((c for c in df.columns if 'OI' in c or 'Interest' in c), None)
    if oi_col: df = df.rename(columns={oi_col: 'oi'})
    else: df['oi'] = 100000
    
    # Indicators
    df['adx'] = calculate_adx(df)
    df['oi_pct'] = df['oi'].pct_change() * 100
    df = df.dropna().reset_index(drop=True)
    
    print(f"🚀 OPTIMIZING on {len(df)} candles...")
    print("Testing different combinations of Gap, Trend Threshold, and Breakout...")
    print("-" * 60)
    print(f"{'WAVE GAP':<10} | {'ADX THRESH':<10} | {'SURV BRK':<10} | {'FINAL EQUITY':<15} | {'ROI':<10}")
    print("-" * 60)
    
    best_roi = -100
    best_params = {}
    
    # Generate combinations
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in combinations:
        final_equity = run_single_backtest(df, params['wave_gap'], params['adx_threshold'], params['survivor_breakout'])
        roi = ((final_equity - 1000000) / 1000000) * 100
        
        print(f"{params['wave_gap']:<10} | {params['adx_threshold']:<10} | {params['survivor_breakout']:<10} | ₹ {final_equity:,.0f}    | {roi:.2f}%")
        
        if roi > best_roi:
            best_roi = roi
            best_params = params

    print("-" * 60)
    print("🏆 WINNING SETTINGS FOUND:")
    print(f"   Wave Gap:          {best_params['wave_gap']}")
    print(f"   ADX Threshold:     {best_params['adx_threshold']}")
    print(f"   Survivor Breakout: {best_params['survivor_breakout']}")
    print(f"   💰 MAX PROFIT:     ₹ {best_roi * 10000:,.0f} (ROI: {best_roi:.2f}%)")
    print("-" * 60)
