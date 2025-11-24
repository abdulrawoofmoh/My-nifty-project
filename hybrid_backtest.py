import logging
import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
CONFIG = {
    'initial_capital': 1500000.0,  # 15 Lakhs
    
    # POSITION SIZING
    'num_lots': 1,                 # How many lots to trade
    'lot_size': 75,                # Current Nifty Lot Size
    'margin_per_lot': 150000.0,    # Approx margin to sell 1 lot

    # REGIME THRESHOLDS
    'adx_period': 14,
    'adx_threshold': 40,
    'oi_drop_threshold': -5,

    # STRATEGY PARAMS
    'wave_gap': 15,
    'survivor_breakout': 40
}

logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- INDICATORS ---
def calculate_adx(df, period=14):
    df = df.copy()
    if 'High' not in df.columns: df = df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close'})
    
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

class HybridBacktest:
    def __init__(self):
        # 1. Load & Clean
        if os.path.exists("data.csv"):
            filename = "data.csv"
        else:
            files = glob.glob("NIFTY*.csv")
            if not files:
                print("❌ ERROR: No CSV file found!")
                exit()
            filename = max(files, key=os.path.getsize)
            
        print(f"📂 Using File: {filename}")
        df = pd.read_csv(filename)
        
        df.columns = [c.strip() for c in df.columns]
        col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open','high','low','close','volume']}
        df = df.rename(columns=col_map)
        if 'Date' in df.columns: df = df.rename(columns={'Date': 'timestamp'})
        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        
        oi_col = next((c for c in df.columns if c.upper() == 'OI' or 'Open Interest' in c), None)
        if oi_col:
            df = df.rename(columns={oi_col: 'oi'})
            print("✅ Found Open Interest Data!")
        else:
            df['oi'] = 100000 

        try:
            df['timestamp'] = df['timestamp'].astype(str).apply(lambda x: x.split(' GMT')[0])
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except: pass

        self.data = df.sort_values('timestamp').reset_index(drop=True)
        self.data['adx'] = calculate_adx(self.data, CONFIG['adx_period'])
        self.data['oi_pct'] = self.data['oi'].pct_change() * 100
        self.data = self.data.dropna().reset_index(drop=True)

        # State
        self.cash = CONFIG['initial_capital']
        self.position = 0
        self.avg_price = 0.0
        self.pnl_realized = 0.0
        self.active_regime = "NEUTRAL"
        self.wave_buy_target = None
        self.wave_sell_target = None
        self.survivor_ref_price = None
        self.trades_count = 0
        self.skipped_trades = 0
        
        # DYNAMIC QUANTITY CALCULATION
        self.trade_qty = CONFIG['num_lots'] * CONFIG['lot_size']

    def run(self):
        print(f"🚀 STARTING HYBRID TEST ({len(self.data)} candles)...")
        print(f"💰 Capital: ₹{CONFIG['initial_capital']:,.0f} | Lots: {CONFIG['num_lots']} (Qty: {self.trade_qty})")

        for i, row in self.data.iterrows():
            price = row['Close']
            adx = row['adx']
            oi_chg = row['oi_pct']
            time_str = str(row['timestamp'])

            is_trending = (adx > CONFIG['adx_threshold']) or (oi_chg < CONFIG['oi_drop_threshold'])
            new_regime = "SURVIVOR" if is_trending else "WAVE"

            if new_regime != self.active_regime:
                self.active_regime = new_regime
                self.wave_buy_target = price - CONFIG['wave_gap']
                self.wave_sell_target = price + CONFIG['wave_gap']
                self.survivor_ref_price = price

            if self.active_regime == "WAVE":
                self._run_wave(price, time_str)
            elif self.active_regime == "SURVIVOR":
                self._run_survivor(price, time_str)

        self._report()

    def _run_wave(self, price, time_str):
        gap = CONFIG['wave_gap']
        if price <= self.wave_buy_target:
            self._trade("BUY", price, self.trade_qty, time_str, "WAVE")
            self.wave_buy_target = price - gap
            self.wave_sell_target = price + gap
        elif price >= self.wave_sell_target:
            self._trade("SELL", price, self.trade_qty, time_str, "WAVE")
            self.wave_buy_target = price - gap
            self.wave_sell_target = price + gap
            
        if price > self.wave_sell_target: 
            self.wave_sell_target = price + gap
            self.wave_buy_target = price - gap
        if price < self.wave_buy_target:
            self.wave_buy_target = price - gap
            self.wave_sell_target = price + gap

    def _run_survivor(self, price, time_str):
        breakout = CONFIG['survivor_breakout']
        if price > (self.survivor_ref_price + breakout):
            # Double quantity if reversing (Flip), Single if Flat
            qty = self.trade_qty * 2 if self.position != 0 else self.trade_qty
            if self.position <= 0: self._trade("BUY", price, qty, time_str, "SURVIVOR")
            self.survivor_ref_price = price
        elif price < (self.survivor_ref_price - breakout):
            qty = self.trade_qty * 2 if self.position != 0 else self.trade_qty
            if self.position >= 0: self._trade("SELL", price, qty, time_str, "SURVIVOR")
            self.survivor_ref_price = price

    def _check_margin(self, side, qty, price):
        is_opening = False
        if side == "BUY" and self.position >= 0: is_opening = True 
        if side == "SELL" and self.position <= 0: is_opening = True 
        
        if not is_opening: return True 
        
        # Calculate Margin based on LOTS
        lots = qty / CONFIG['lot_size']
        margin_needed = lots * CONFIG['margin_per_lot']
        
        available_funds = CONFIG['initial_capital'] + self.pnl_realized
        
        if available_funds < margin_needed:
            return False
        return True

    def _trade(self, side, price, qty, time_str, tag):
        if not self._check_margin(side, qty, price):
            self.skipped_trades += 1
            return

        val = price * qty
        closing_qty = 0
        if (self.position > 0 and side == "SELL") or (self.position < 0 and side == "BUY"):
            closing_qty = min(abs(self.position), qty)
            
        if closing_qty > 0:
            pnl = (price - self.avg_price)*closing_qty if self.position > 0 else (self.avg_price - price)*closing_qty
            self.pnl_realized += pnl
            
        if side == "BUY":
            if self.position >= 0:
                self.avg_price = ((self.position * self.avg_price) + val) / (self.position + qty)
            else:
                if (qty - abs(self.position)) > 0: self.avg_price = price
                elif (qty - abs(self.position)) == 0: self.avg_price = 0
            self.position += qty
            self.cash -= val
        else:
            if self.position <= 0:
                self.avg_price = ((abs(self.position) * self.avg_price) + val) / (abs(self.position) + qty)
            else:
                if (qty - self.position) > 0: self.avg_price = price
                elif (qty - self.position) == 0: self.avg_price = 0
            self.position -= qty
            self.cash += val
            
        self.trades_count += 1

    def _report(self):
        last_price = self.data.iloc[-1]['Close']
        unrealized = (last_price - self.avg_price)*self.position if self.position > 0 else (self.avg_price - last_price)*abs(self.position)
        total_equity = self.cash + (self.position * last_price)
        roi = ((total_equity - CONFIG['initial_capital']) / CONFIG['initial_capital']) * 100
        
        print("\n" + "="*40)
        print(f"HYBRID BACKTEST REPORT")
        print("="*40)
        print(f"Data File:      {self.data.shape[0]} candles")
        print(f"Trades Executed:{self.trades_count}")
        print(f"Trades SKIPPED: {self.skipped_trades} (Margin Limit)")
        print("-" * 40)
        print(f"Realized P&L:   ₹ {self.pnl_realized:,.2f}")
        print(f"Unrealized P&L: ₹ {unrealized:,.2f}")
        print("-" * 40)
        print(f"TOTAL EQUITY:   ₹ {total_equity:,.2f}")
        print(f"ROI:            {roi:.2f}%")
        print("="*40)

if __name__ == "__main__":
    bot = HybridBacktest()
    bot.run()
