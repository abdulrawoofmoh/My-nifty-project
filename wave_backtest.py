import logging
import pandas as pd
import os

# --- CONFIGURATION ---
CONFIG = {
    'base_gap': 15,       # Adjusted to 15 (Sweet spot between 10 and 20)
    'qty': 50,            # 1 Lot
    'lot_size': 50,       
    'initial_capital': 1000000.0,
    'force_entry': True 
}

FILENAME = "data.csv"
logging.basicConfig(level=logging.INFO, format='%(message)s')

class BacktestEngine:
    def __init__(self, config):
        if not os.path.exists(FILENAME):
            print("❌ ERROR: 'data.csv' not found.")
            exit()
            
        df = pd.read_csv(FILENAME)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' in df.columns: df = df.rename(columns={'date': 'timestamp'})
        try:
            df['timestamp'] = df['timestamp'].astype(str).apply(lambda x: x.split(' GMT')[0])
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except: pass
        
        self.data = df.sort_values('timestamp')
        self.config = config
        
        # Portfolio State
        self.cash = config['initial_capital']
        self.position = 0     
        self.avg_price = 0.0  
        self.pnl_realized = 0.0
        
        # Strategy State
        self.active_buy_target = None
        self.active_sell_target = None
        self.trades_count = 0

    def _get_gaps(self, current_qty):
        return self.config['base_gap'], self.config['base_gap']

    def run(self):
        print(f"🚀 BACKTEST STARTING | Gap: {self.config['base_gap']} | Capital: {self.config['initial_capital']}")
        
        for index, row in self.data.iterrows():
            price = row['close']
            time_str = str(row['timestamp'])
            
            # 1. FORCE START
            if self.config['force_entry'] and index == 0:
                self._execute_trade("BUY", price, self.config['qty'], time_str, "FORCE")
                self.active_buy_target = price - self.config['base_gap']
                self.active_sell_target = price + self.config['base_gap']
                continue

            # 2. CHECK EXECUTION
            traded = False
            if self.active_sell_target and price >= self.active_sell_target:
                self._execute_trade("SELL", price, self.config['qty'], time_str, "PROFIT")
                traded = True
            elif self.active_buy_target and price <= self.active_buy_target:
                self._execute_trade("BUY", price, self.config['qty'], time_str, "DIP")
                traded = True

            # 3. UPDATE TARGETS
            buy_gap, sell_gap = self._get_gaps(self.position)
            theoretical_buy = price - buy_gap
            theoretical_sell = price + sell_gap
            
            # Grid Logic: Always center around current price
            self.active_buy_target = theoretical_buy
            self.active_sell_target = theoretical_sell

        self._report()

    def _execute_trade(self, side, price, qty, time_str, tag=""):
        trade_val = price * qty
        qty_delta = qty if side == "BUY" else -qty
        
        # --- FIFO ACCOUNTING LOGIC ---
        # 1. Check if we are closing an existing position
        closing_qty = 0
        opening_qty = 0
        
        if (self.position > 0 and side == "SELL") or (self.position < 0 and side == "BUY"):
            # We are reducing risk / closing
            if abs(self.position) >= qty:
                closing_qty = qty
            else:
                closing_qty = abs(self.position)
                opening_qty = qty - closing_qty
        else:
            # We are adding to position or opening new
            opening_qty = qty

        # 2. Calculate P&L on Closing Portion
        if closing_qty > 0:
            # Profit = (Sell Price - Avg Cost) if Long
            # Profit = (Avg Cost - Buy Price) if Short
            if self.position > 0: # Closing Long
                pnl = (price - self.avg_price) * closing_qty
            else: # Closing Short
                pnl = (self.avg_price - price) * closing_qty
            
            self.pnl_realized += pnl

        # 3. Update Average Price on Opening Portion
        if opening_qty > 0:
            new_val = opening_qty * price
            # If we flipped from 0 to position
            if self.position == 0 or (self.position > 0 and side == "BUY") or (self.position < 0 and side == "SELL"):
                current_val = abs(self.position) * self.avg_price
                total_val = current_val + new_val
                total_qty = abs(self.position) + opening_qty
                self.avg_price = total_val / total_qty
            else:
                # This handles the flip case (e.g. -10 to +10). 
                # The closing part handled above resets pos to 0. The opening part sets new avg.
                self.avg_price = price

        # 4. Update State
        self.position += qty_delta
        if side == "BUY": self.cash -= trade_val
        else: self.cash += trade_val
        
        # Safety reset
        if self.position == 0: self.avg_price = 0
            
        self.trades_count += 1
        # print(f"[{time_str}] {side} {qty} @ {price:.2f} | Pos: {self.position} | PnL: {self.pnl_realized:.0f}")

    def _report(self):
        last_price = self.data.iloc[-1]['close']
        
        # Calculate Unrealized P&L correctly
        unrealized = 0
        if self.position > 0:
            unrealized = (last_price - self.avg_price) * abs(self.position)
        elif self.position < 0:
            unrealized = (self.avg_price - last_price) * abs(self.position)
            
        # Total Equity = Cash + (Pos * Price) is simpler, but PnL method is good for checking
        # Let's use the Cash Balance method for final truth
        liquidation_value = self.position * last_price
        total_equity = self.cash + liquidation_value
        
        roi = ((total_equity - self.config['initial_capital']) / self.config['initial_capital']) * 100
        
        print("\n" + "="*40)
        print(f"FINAL REPORT (Gap: {self.config['base_gap']})")
        print("="*40)
        print(f"Total Trades:   {self.trades_count}")
        print(f"Final Position: {self.position}")
        print(f"Avg Entry Price:{self.avg_price:.2f}")
        print("-" * 40)
        print(f"Realized P&L:   ₹ {self.pnl_realized:,.2f}")
        print(f"Unrealized P&L: ₹ {unrealized:,.2f}")
        print("-" * 40)
        print(f"TOTAL EQUITY:   ₹ {total_equity:,.2f}")
        print(f"ROI:            {roi:.2f}%")
        print("="*40)

if __name__ == "__main__":
    engine = BacktestEngine(CONFIG)
    engine.run()
