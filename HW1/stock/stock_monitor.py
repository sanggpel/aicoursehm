import yfinance as yf
from datetime import datetime
import pandas as pd
import time
import os
import logging
import requests
from typing import Dict, List

class StockPriceMonitor:
    def __init__(self, csv_file: str):
        """
        Initialize the stock price monitor
        
        Args:
            csv_file (str): Path to CSV file containing stock symbols and stop loss prices
        """
        print("Initializing StockPriceMonitor...")
        self.csv_file = csv_file
        self.stocks_data = self.load_stock_data()
        self.alerted_stocks = {}  # Track stocks alerts with timestamps and count
        self.max_daily_alerts = 3  # Maximum number of alerts per stock per day
        
        # Telegram configuration
        self.telegram_config = {
            'bot_token': 'YOUR_BOT_TOKEN_HERE',
            'chat_id': 'YOUR_CHAT_ID_HERE'
        }
        
        self.setup_logging()
        self.logger.info(f"Monitoring {len(self.stocks_data)} stocks for stop loss triggers")

    def setup_logging(self):
        """Set up logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = f"logs/stock_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging

    def load_stock_data(self) -> Dict[str, float]:
        """
        Load stock data from CSV file
        
        Returns:
            Dict[str, float]: Dictionary of stock symbols and their stop loss prices
        """
        try:
            df = pd.read_csv(self.csv_file)
            return dict(zip(df['symbol'], df['stop_loss']))
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")

    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all monitored stocks
        
        Returns:
            Dict[str, float]: Dictionary of stock symbols and their current prices
        """
        current_prices = {}
        # Fetch all symbols at once for better performance
        symbols_str = " ".join(self.stocks_data.keys())
        
        try:
            # Use yfinance download function for multiple symbols
            data = yf.download(symbols_str, period="1d", interval="1m", progress=False)
            
            # Get the latest prices from the data
            if len(self.stocks_data) == 1:
                # Handle single stock case
                symbol = list(self.stocks_data.keys())[0]
                if not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    if not pd.isna(latest_price):
                        current_prices[symbol] = latest_price
                    else:
                        self.logger.warning(f"No valid price found for {symbol}")
            else:
                # Handle multiple stocks case
                latest_prices = data['Close'].iloc[-1]
                for symbol in self.stocks_data.keys():
                    if symbol in latest_prices.index and not pd.isna(latest_prices[symbol]):
                        current_prices[symbol] = latest_prices[symbol]
                    else:
                        self.logger.warning(f"No valid price found for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}")
            
        if not current_prices:
            self.logger.warning("No valid prices fetched in this iteration")
            
        return current_prices

    def send_telegram_message(self, message: str) -> bool:
        """
        Send alert message via Telegram
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            # Verify Telegram configuration
            if not self.telegram_config['bot_token'] or not self.telegram_config['chat_id']:
                self.logger.error("Telegram configuration is incomplete")
                return False
                
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            params = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            self.logger.info(f"Sending Telegram message to chat_id: {self.telegram_config['chat_id']}")
            response = requests.post(url, params=params, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram notification sent successfully: {response.json()}")
                return True
            else:
                self.logger.error(f"Error sending Telegram message. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error("Telegram request timed out")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error sending Telegram message: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending Telegram message: {str(e)}")
            return False

    def can_send_alert(self, symbol: str) -> bool:
        """
        Check if we can send an alert for this stock based on daily limits
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            bool: True if we can send an alert, False otherwise
        """
        now = datetime.now()
        today = now.date()
        
        if symbol not in self.alerted_stocks:
            self.alerted_stocks[symbol] = {'date': today, 'count': 0, 'last_alert': None}
            return True
            
        stock_alerts = self.alerted_stocks[symbol]
        
        # Reset count if it's a new day
        if stock_alerts['date'] != today:
            stock_alerts['date'] = today
            stock_alerts['count'] = 0
            stock_alerts['last_alert'] = None
            return True
            
        # Check if we've hit the daily limit
        if stock_alerts['count'] >= self.max_daily_alerts:
            self.logger.info(f"Daily alert limit reached for {symbol}")
            return False
            
        # Ensure at least 1 hour between alerts for the same stock
        if stock_alerts['last_alert']:
            time_since_last = now - stock_alerts['last_alert']
            if time_since_last.total_seconds() < 3600:  # 1 hour in seconds
                return False
                
        return True

    def send_stop_loss_alert(self, symbol: str, current_price: float, stop_loss: float):
        """
        Send stop loss alert for a specific stock
        
        Args:
            symbol (str): Stock symbol
            current_price (float): Current price
            stop_loss (float): Stop loss price
        """
        if not self.can_send_alert(symbol):
            return
            
        current_time = datetime.now()
        alert_count = self.alerted_stocks[symbol]['count'] + 1
        
        message = (
            f"🔔 STOP LOSS ALERT! (Alert {alert_count}/{self.max_daily_alerts}) 🔔\n\n"
            f"Stock: {symbol}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Stop Loss: ${stop_loss:.2f}\n"
            f"Time: {current_time.strftime('%I:%M %p')}\n\n"
            f"⚠️ Stock has fallen below stop loss level!"
        )
        
        if self.send_telegram_message(message):
            self.alerted_stocks[symbol]['count'] += 1
            self.alerted_stocks[symbol]['last_alert'] = current_time
            self.logger.info(f"Stop loss alert sent for {symbol} (Alert {alert_count}/{self.max_daily_alerts})")
        else:
            self.logger.error(f"Failed to send stop loss alert for {symbol}")

    def monitor_prices(self, check_interval: int = 60):
        """
        Main monitoring loop
        
        Args:
            check_interval (int): Time between price checks in seconds
        """
        self.logger.info(f"Starting price monitoring. Checking every {check_interval} seconds...")
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                # Check if it's within market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
                now = datetime.now()
                if now.weekday() >= 5:  # Saturday or Sunday
                    self.logger.info("Market is closed (weekend). Waiting for market hours...")
                    time.sleep(3600)  # Sleep for an hour
                    continue
                
                current_hour = now.hour
                current_minute = now.minute
                market_time = current_hour * 100 + current_minute  # Convert to HHMM format
                
                if market_time < 930 or market_time > 1600:
                    self.logger.info("Market is closed. Waiting for market hours...")
                    time.sleep(300)  # Sleep for 5 minutes
                    continue
                
                current_prices = self.get_current_prices()
                
                if current_prices:  # Only process if we got valid prices
                    consecutive_errors = 0  # Reset error counter on success
                    
                    for symbol, current_price in current_prices.items():
                        stop_loss = self.stocks_data[symbol]
                        
                        self.logger.info(f"{symbol}: Current: ${current_price:.2f}, Stop Loss: ${stop_loss:.2f}")
                        
                        # Check if price has hit stop loss and hasn't been alerted yet
                        if current_price <= stop_loss and symbol not in self.alerted_stocks:
                            self.send_stop_loss_alert(symbol, current_price, stop_loss)
                    
                    self.logger.info("-" * 50)
                else:
                    consecutive_errors += 1
                    self.logger.warning(f"No valid prices received (attempt {consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Maximum consecutive errors reached. Waiting for 5 minutes before retrying...")
                        time.sleep(300)  # Wait 5 minutes before retrying
                        consecutive_errors = 0  # Reset counter
                        continue
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("\nStopping monitor...")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Maximum consecutive errors reached. Waiting for 5 minutes before retrying...")
                    time.sleep(300)  # Wait 5 minutes before retrying
                    consecutive_errors = 0
                else:
                    time.sleep(check_interval)

def main():
    # Configuration
    csv_file = "stocks.csv"  # CSV should have columns: symbol, stop_loss
    check_interval = 60      # Seconds between price checks
    
    try:
        print(f"Starting stock price monitor...")
        monitor = StockPriceMonitor(csv_file)
        
        print(f"Starting price monitoring (checking every {check_interval} seconds)...")
        monitor.monitor_prices(check_interval=check_interval)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()