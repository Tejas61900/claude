import asyncio
import json
import os
import sys
import ssl
import websockets
import requests
import pandas as pd
import numpy as np
import ta
import datetime as dt
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from google.protobuf.json_format import MessageToDict
from threading import Thread, Lock
import time
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import quote


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("trading_bot.log", maxBytes=5*1024*1024, backupCount=3),  # 5MB per file, keeps 3 backups
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import MarketDataFeedV3_pb2 as pb
except ImportError:
    logger.error("MarketDataFeedV3_pb2 module not found. Please generate it from the Upstox protobuf definition.")
    logger.info("You can generate it using: protoc --python_out=. MarketDataFeedV3.proto")
    raise

class UpstoxTradingBot:
    def __init__(self, access_token, use_ai=False):
        self.access_token = access_token
        self.trades = []
        self.instrument_df = None
        self.initial_capital = 20000
        self.use_ai = use_ai
        self.current_capital = self.initial_capital

        # Fixed parameters
        self.fixed_quantity = 75
        self.fixed_sl = 10
        self.fixed_tp = 35
        self.trailing_sl = 20
        self.angle_threshold = 5.00
        self.risk_per_trade = 0.05

        # Storage for real-time data
        self.data_lock = Lock()
        self.ohlc_buffer = {}
        self.last_ltp = {}
        self.subscribed_instruments = set()
        self.candle_storage = {}
        self.max_storage_days = 3  # Only store 3 days of data

        # Trade tracking
        self.active_position = None
        self.trade_entry_time = None
        self.max_trade_duration = timedelta(minutes=45)
        self.prev_trend = None

        # AI model components
        self.model = None
        self.scaler = None

        self.active_websocket = None
        self.websocket_lock = Lock()
        self.websocket_ready = False
        # Connection tracking
        self.last_successful_connection = time.time()
        self.max_disconnect_time = 300  # 5 minutes

        # Initialize
        logger.info("Initializing Upstox Trading Bot...")
        self.load_instruments()

    def check_connection_health(self):
        """Check if we've been disconnected for too long"""
        current_time = time.time()
        with self.websocket_lock:
            if not self.websocket_ready and (current_time - self.last_successful_connection > self.max_disconnect_time):
                logger.error("Disconnected for too long. Exiting.")
                self.shutdown()
                return False
        return True

    def shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Performing graceful shutdown...")

        # Exit any active trade
        if self.active_position:
            try:
                current_price = self.get_last_ltp(self.active_position["instrument_key"])
                if current_price:
                    self.exit_trade(current_price, "Emergency Exit - Connection Lost")
                else:
                    logger.warning("Can't exit trade gracefully - no price data")
            except Exception as e:
                logger.error(f"Error during emergency exit: {e}")

        # Export results
        self.export_results()
        self.visualize_performance()

        sys.exit(0)

    def load_instruments(self):
        """Load and filter instruments from Upstox"""
        try:
            logger.info("Loading instrument data...")
            df = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
            df = df[(df['name'] == "NIFTY") & (df['instrument_type'] == "OPTIDX")]
            df['expiry'] = pd.to_datetime(df['expiry'])
            df.reset_index(drop=True, inplace=True)
            self.instrument_df = df
            logger.info(f"Loaded {len(df)} NIFTY options instruments")
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            raise

    def get_historical_data(self, instrument_key, from_date=None, to_date=None):
        """Fetch historical data from Upstox API"""
        if not from_date:
            from_date = (dt.datetime.now() - dt.timedelta(days=3)).date()
        if not to_date:
            to_date = dt.datetime.now().date()

        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

        try:
            response = requests.get(url, headers=headers)
            res = response.json()

            if 'data' not in res or 'candles' not in res['data']:
                logger.warning(f"Error fetching data for {instrument_key}: {res.get('errors', 'Unknown error')}")
                return pd.DataFrame()

            candle_data = res['data']['candles']
            df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            df.sort_values('timestamp', inplace=True)

            # Store candles in memory
            with self.data_lock:
                self.candle_storage[instrument_key] = df

            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {instrument_key}: {e}")
            return pd.DataFrame()

    def get_intraday_missing_data(self, instrument_key):
        """Fetch intraday data for mid-session start"""
        try:
            encoded_key = quote(instrument_key)
            url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"
            headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

            response = requests.get(url, headers=headers)
            res = response.json()

            if 'data' not in res or 'candles' not in res['data']:
                logger.warning(f"Error fetching intraday data for {instrument_key}: {res.get('errors', 'Unknown error')}")
                return pd.DataFrame()

            candle_data = res['data']['candles']
            df = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            df.sort_values('timestamp', inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {instrument_key}: {e}")
            return pd.DataFrame()

    def merge_historical_with_intraday(self, instrument_key):
        """Combine historical and intraday data"""
        historical_df = self.get_historical_data(instrument_key)
        intraday_df = self.get_intraday_missing_data(instrument_key)

        if historical_df.empty and intraday_df.empty:
            logger.warning(f"No data available for {instrument_key}")
            return pd.DataFrame()

        if historical_df.empty:
            return intraday_df

        if intraday_df.empty:
            return historical_df

        # Combine and remove duplicates
        combined_df = pd.concat([historical_df, intraday_df])
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        # Update storage
        with self.data_lock:
            self.candle_storage[instrument_key] = combined_df

        return combined_df

    def clean_old_data(self):
        """Clean up old data to manage memory usage"""
        with self.data_lock:
            current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            cutoff_time = current_time - timedelta(days=self.max_storage_days)

            for key in list(self.candle_storage.keys()):
                if not self.candle_storage[key].empty:
                    # Filter data newer than cutoff time
                    self.candle_storage[key] = self.candle_storage[key][
                        self.candle_storage[key]['timestamp'] > cutoff_time
                    ]

                    # If all data was removed, drop the key entirely
                    if self.candle_storage[key].empty:
                        del self.candle_storage[key]

    def clean_ohlc_buffer(self):
        """Clean up old data from OHLC buffer to prevent memory leaks"""
        with self.data_lock:
            current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            cutoff_ms = int((current_time - timedelta(hours=3)).timestamp() * 1000)

            for key in list(self.ohlc_buffer.keys()):
                if self.ohlc_buffer[key]:
                    # Keep only recent candles
                    self.ohlc_buffer[key] = [
                        candle for candle in self.ohlc_buffer[key] 
                        if candle["timestamp"] > cutoff_ms
                    ]

    def calculate_indicators(self, df):
        """Calculate technical indicators for trading decisions"""
        # Ensure DataFrame is sorted by timestamp
        df = df.sort_values(by='timestamp').copy()

        # Calculate moving averages
        df['EMA9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['EMA15'] = ta.trend.ema_indicator(df['close'], window=15)

        # Calculate other indicators
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['RSI_SMA_14'] = df['RSI'].rolling(window=14).mean()
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['close'])
        df['EMA_Angle'] = self.calculate_ema_angle(df)

        # Calculate ATR for dynamic targets and stops
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

        # Calculate candle strength
        df['candle_body'] = abs(df['close'] - df['open'])
        high_low_diff = df['high'] - df['low']
        df['candle_strength'] = np.where(high_low_diff > 0, df['candle_body'] / high_low_diff, 0)

        # Market trend strength
        df['Close_SMA50'] = df['close'].rolling(window=50).mean()
        df['Market_Trend'] = np.where(df['close'] > df['Close_SMA50'], 1, -1)

        # Handle missing values
        df.bfill(inplace=True)

        return df

    def calculate_ema_angle(self, df):
        """Calculate angle between EMA lines to determine trend strength"""
        ema_diff = df['EMA9'] - df['EMA15']
        ema_diff_change = ema_diff.diff()
        ema_angle = np.degrees(np.arctan(ema_diff_change))
        return ema_angle

    def is_valid_trading_time(self, timestamp):
        """Check if current time is valid for trading"""
        # Check if market is open (9:15 AM to 3:30 PM)
        if timestamp.hour < 9 or (timestamp.hour == 9 and timestamp.minute < 15):
            return False
        if timestamp.hour > 15 or (timestamp.hour == 15 and timestamp.minute > 30):
            return False

        # Avoid trading in the first 5 minutes (market volatility settling)
        if timestamp.hour == 9 and timestamp.minute < 20:
            return False

        return True

    def get_best_option(self, nifty_price, signal, expiry_date=None):
        """Select the best option based on strike price and expiry"""
        # Calculate base strike price (ATM)
        atm_strike = round(nifty_price / 50) * 50

        # For slightly OTM options (better risk-reward)
        if signal == "BULLISH":
            strike_candidates = [atm_strike, atm_strike + 50]  # ATM and OTM
            option_type = 'CE'
        else:  # BEARISH
            strike_candidates = [atm_strike, atm_strike - 50]  # ATM and OTM
            option_type = 'PE'

        # Get expiry date - if not provided, use nearest expiry
        if not expiry_date:
            current_date = datetime.now().date()
            available_expiries = self.instrument_df['expiry'].unique()
            future_expiries = [exp for exp in available_expiries if exp.date() >= current_date]
            if not future_expiries:
                logger.error("No future expiries available")
                return None, None, None
            expiry_date = min(future_expiries)
        else:
            expiry_date = pd.to_datetime(expiry_date)

        expiry_filtered_df = self.instrument_df[self.instrument_df['expiry'] == expiry_date]

        if expiry_filtered_df.empty:
            logger.warning(f"No options available for expiry {expiry_date}")
            return None, None, None

        # Find the best strike from candidates
        for strike in strike_candidates:
            option = expiry_filtered_df[(expiry_filtered_df['strike'] == strike) & 
                                      (expiry_filtered_df['option_type'] == option_type)]

            if not option.empty:
                instrument_key = option.iloc[0]['instrument_key']
                trading_symbol = option.iloc[0]['tradingsymbol']
                strike_price = option.iloc[0]['strike']
                return instrument_key, trading_symbol, strike_price

        return None, None, None

    def calculate_position_size(self, entry_price, stop_loss, current_capital):
        """Calculate position size based on risk management"""
        return self.fixed_quantity  # Using fixed quantity as specified

    def test_conditions(self, df, index, prev_trend):
        """Test trading conditions based on technical indicators"""
        if index not in df.index:
            return None

        if not self.is_valid_trading_time(index):
            return None

        row = df.loc[index]

        # Only use AI if enabled
        if self.use_ai:
            if self.model is None:
                self.train_ai_model(df)
            if self.model is not None:
                ai_signal = self.get_ai_prediction(df, df.index.get_loc(index))
                if not ai_signal:
                    return None

        # Determine trend direction based on EMA9 vs EMA15
        trend_direction = 'bullish' if row['EMA9'] > row['EMA15'] else 'bearish'

        # Check for Fresh EMA Crossover
        fresh_crossover = (prev_trend != trend_direction)

        # Check Market Trend Confirmation
        if trend_direction == 'bullish' and row['Market_Trend'] != 1:
            return None  # Skip bullish trades in bearish market
        if trend_direction == 'bearish' and row['Market_Trend'] != -1:
            return None  # Skip bearish trades in bullish market

        # Check EMA angle condition (should be ≥ threshold degrees)
        if abs(row['EMA_Angle']) < self.angle_threshold:
            return None

        # Dynamic Price Retest Condition (Re-Entry Condition)
        lower_bound = row['EMA9'] - 5
        upper_bound = row['EMA9'] + 5
        price_retest = lower_bound <= row['close'] <= upper_bound

        # Check candle strength condition
        if row['candle_strength'] <= 0.7:
            return None

        # Check for Bullish Signal
        if (trend_direction == 'bullish' and 
            row['MACD'] > row['MACD_Signal'] and 
            30 <= row['RSI_SMA_14'] <= 60 and 20 <= row['ADX'] <= 40):  # Avoiding overbought zone

            # Fresh Crossover or Re-Entry near EMA9
            if fresh_crossover or price_retest:
                return "BULLISH"

        # Check for Bearish Signal
        elif (trend_direction == 'bearish' and 
              row['MACD'] < row['MACD_Signal'] and 
              40 <= row['RSI_SMA_14'] <= 70 and 20 <= row['ADX'] <= 40):  # Avoiding oversold zone

            # Fresh Crossover or Re-Entry near EMA9
            if fresh_crossover or price_retest:
                return "BEARISH"

        return None

    def prepare_features(self, df):
        """Prepare features for AI model"""
        features = pd.DataFrame()
        features['ema_ratio'] = df['EMA9'] / df['EMA15']
        features['rsi'] = df['RSI']
        features['adx'] = df['ADX']
        features['macd_diff'] = df['MACD'] - df['MACD_Signal']
        features['market_trend'] = df['Market_Trend']
        return features

    def train_ai_model(self, df):
        """Train AI model using historical data"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            # Prepare features and target
            features = self.prepare_features(df)
            target = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 for price increase

            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | pd.isna(target))
            features = features[valid_idx]
            target = target[valid_idx]

            if len(features) < 2:  # Not enough data
                logger.warning("Not enough data to train AI model")
                return

            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)

            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(scaled_features, target)
            logger.info("AI model trained successfully")
        except Exception as e:
            logger.error(f"Error training AI model: {e}")

    def get_ai_prediction(self, df, current_idx):
        """Get prediction from AI model"""
        if self.model is None or current_idx < 0:
            return True  # Default to True if no model

        features = self.prepare_features(df)
        if features.iloc[current_idx].isna().any():
            return True

        scaled_features = self.scaler.transform(features.iloc[[current_idx]])
        prediction = self.model.predict(scaled_features)[0]
        return bool(prediction)

    # WebSocket Market Data Functions
    async def connect_market_data(self):
        """Connect to the Upstox WebSocket for market data"""
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Use exponential backoff for reconnection
        retry_delay = 1
        max_retry_delay = 60

        while True:
            try:
                # Get WebSocket authorization
                url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
                headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

                response = requests.get(url, headers=headers)
                res = response.json()

                if 'data' not in res or 'authorizedRedirectUri' not in res['data']:
                    logger.error(f"Failed to get WebSocket authorization: {res.get('errors', 'Unknown error')}")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                    continue

                websocket_url = res['data']['authorizedRedirectUri']

                async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
                    with self.websocket_lock:
                        self.active_websocket = websocket
                        self.websocket_ready = True

                    logger.info("✅ WebSocket connection established")

                    # Reset retry delay on successful connection
                    retry_delay = 1

                    # Subscribe to Nifty 50 index initially
                    await self.subscribe_instrument(websocket, "NSE_INDEX|Nifty 50")

                    # Continuously receive and process market data
                    await self.process_market_data(websocket)

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")

                with self.websocket_lock:
                    self.active_websocket = None
                    self.websocket_ready = False

                # Use exponential backoff for reconnection
                logger.info(f"Reconnecting in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    async def subscribe_instrument(self, websocket, instrument_key):
        """Subscribe to an instrument for market data"""
        try:
            if instrument_key in self.subscribed_instruments:
                return

            data = {
                "guid": f"sub_{instrument_key}_{int(time.time())}",
                "method": "sub",
                "data": {
                    "mode": "full",
                    "instrumentKeys": [instrument_key]
                }
            }

            binary_data = json.dumps(data).encode("utf-8")
            await websocket.send(binary_data)
            logger.info(f"Subscribed to {instrument_key}")

            with self.data_lock:
                self.subscribed_instruments.add(instrument_key)

        except Exception as e:
            logger.error(f"Error subscribing to {instrument_key}: {e}")

    async def unsubscribe_instrument(self, websocket, instrument_key):
        """Unsubscribe from an instrument"""
        try:
            if instrument_key not in self.subscribed_instruments:
                return

            data = {
                "guid": f"unsub_{instrument_key}_{int(time.time())}",
                "method": "unsub",
                "data": {
                    "instrumentKeys": [instrument_key]
                }
            }

            binary_data = json.dumps(data).encode("utf-8")
            await websocket.send(binary_data)
            logger.info(f"Unsubscribed from {instrument_key}")

            with self.data_lock:
                self.subscribed_instruments.remove(instrument_key)

        except Exception as e:
            logger.error(f"Error unsubscribing from {instrument_key}: {e}")

    async def process_market_data(self, websocket):
        """Process incoming market data from WebSocket"""
        while True:
            try:
                message = await websocket.recv()
                feed_response = self.decode_protobuf(message)
                data_dict = MessageToDict(feed_response)

                if "feeds" in data_dict:
                    for instrument_key, feed_data in data_dict["feeds"].items():
                        if "fullFeed" in feed_data:
                            # Process index data
                            if "indexFF" in feed_data["fullFeed"]:
                                index_data = feed_data["fullFeed"]["indexFF"]
                                self.process_index_data(instrument_key, index_data)

                            # Process option data
                            elif "ltpc" in feed_data["fullFeed"]:
                                self.process_option_data(instrument_key, feed_data["fullFeed"])

            except Exception as e:
                logger.error(f"Error processing market data: {e}")
                # Try to reconnect if connection lost
                break

    def get_last_ltp(self, instrument_key, default=None):
        """Thread-safe getter for last_ltp"""
        with self.data_lock:
            return self.last_ltp.get(instrument_key, default)

    def set_last_ltp(self, instrument_key, value):
        """Thread-safe setter for last_ltp"""
        with self.data_lock:
            self.last_ltp[instrument_key] = value

    def process_index_data(self, instrument_key, index_data):
        """Process index data from WebSocket feed"""
        with self.data_lock:
            # Extract LTP
            if "ltpc" in index_data and "ltp" in index_data["ltpc"]:
                self.set_last_ltp(instrument_key, float(index_data["ltpc"]["ltp"]))

            # Extract OHLC (Only 1-Minute Interval)
            if "marketOHLC" in index_data and "ohlc" in index_data["marketOHLC"]:
                ohlc_data = index_data["marketOHLC"]["ohlc"]
                # Filter only "1minute" (Updated from "I1")
                ohlc_1m_entries = [entry for entry in ohlc_data if entry["interval"] == "1minute"]
                if ohlc_1m_entries:
                    latest_ohlc_1m = max(ohlc_1m_entries, key=lambda x: int(x["ts"]))

                    # Create new candle data
                    new_candle = {
                        "timestamp": int(latest_ohlc_1m["ts"]),
                        "open": float(latest_ohlc_1m["open"]),
                        "high": float(latest_ohlc_1m["high"]),
                        "low": float(latest_ohlc_1m["low"]),
                        "close": float(latest_ohlc_1m["close"]),
                        "volume": 0,  # Not available for index
                        "oi": 0       # Not available for index
                    }

                    # Update OHLC buffer
                    if instrument_key not in self.ohlc_buffer:
                        self.ohlc_buffer[instrument_key] = []

                    # Replace existing candle if timestamp matches, else append new
                    for i, candle in enumerate(self.ohlc_buffer[instrument_key]):
                        if candle["timestamp"] == new_candle["timestamp"]:
                            self.ohlc_buffer[instrument_key][i] = new_candle
                            break
                    else:
                        self.ohlc_buffer[instrument_key].append(new_candle)

                    # Update candle storage
                    if instrument_key in self.candle_storage:
                        df = self.candle_storage[instrument_key]
                        timestamp = pd.to_datetime(new_candle["timestamp"], unit='ms', utc=True).tz_convert('Asia/Kolkata')

                        # Update existing row if timestamp exists, else append
                        if timestamp in df['timestamp'].values:
                            df.loc[df['timestamp'] == timestamp, ['open', 'high', 'low', 'close']] = [
                                new_candle["open"], new_candle["high"], new_candle["low"], new_candle["close"]
                            ]
                        else:
                            new_row = pd.DataFrame({
                                'timestamp': [timestamp],
                                'open': [new_candle["open"]],
                                'high': [new_candle["high"]],
                                'low': [new_candle["low"]],
                                'close': [new_candle["close"]],
                                'volume': [0],
                                'oi': [0]
                            })
                            df = pd.concat([df, new_row]).sort_values('timestamp')
                            self.candle_storage[instrument_key] = df

    def process_option_data(self, instrument_key, option_data):
        """Process option data from WebSocket feed"""
        with self.data_lock:
            # Extract LTP for option
            if "ltpc" in option_data and "ltp" in option_data["ltpc"]:
                self.set_last_ltp(instrument_key, float(option_data["ltpc"]["ltp"]))

                # Get the latest LTP safely
                current_price = self.get_last_ltp(instrument_key)
                if current_price is None:
                    return  # Avoid further execution if LTP is not available

                # If this is our active position, check for exit conditions
                if self.active_position and self.active_position["instrument_key"] == instrument_key:
                    self.check_exit_conditions(instrument_key)

    def print_market_status(self):
        """Print indicators, LTP and trade data to terminal"""
        try:
            # Get Nifty data
            nifty_key = "NSE_INDEX|Nifty 50"
            nifty_ltp = self.get_last_ltp(nifty_key)
            nifty_data = self.get_latest_indicator_data(nifty_key)

            if nifty_data.empty or nifty_ltp is None:
                print("⚠️ No Nifty data available")
                return

            latest = nifty_data.iloc[-1]

            # Print market status
            print("\n" + "="*50)
            print(f"📊 {dt.datetime.now().strftime('%H:%M:%S')} | Nifty: ₹{nifty_ltp:.2f}")
            print(f"📈 EMA9: {latest['EMA9']:.2f} | EMA15: {latest['EMA15']:.2f}")
            print(f"📐 Angle: {latest['EMA_Angle']:.2f}° | RSI: {latest['RSI']:.2f}")
            print(f"📉 MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f}")
            print(f"📏 ADX: {latest['ADX']:.2f}")

            # Print active trade if exists
            if self.active_position:
                pos = self.active_position
                option_ltp = self.get_last_ltp(pos['instrument_key'])
                pnl = (option_ltp - pos['entry_price']) * pos['quantity'] if option_ltp else 0

                print(f"\n🔄 ACTIVE TRADE: {pos['tradingsymbol']} ({pos['direction']})")
                print(f"💰 Entry: ₹{pos['entry_price']:.2f} | Current: ₹{option_ltp:.2f}")
                print(f"📊 P&L: ₹{pnl:.2f} | SL: ₹{pos['stop_loss']:.2f} | TP: ₹{pos['target']:.2f}")

            print("="*50 + "\n")
        except Exception as e:
            logger.error(f"Error printing market status: {e}")

    def check_exit_conditions(self, instrument_key):
        """Check if exit conditions are met for active position"""
        if not self.active_position:
            return

        current_price = self.last_ltp[instrument_key]
        position = self.active_position

        # Get latest Nifty data for EMA check
        nifty_df = self.get_latest_indicator_data("NSE_INDEX|Nifty 50")
        if nifty_df.empty:
            return

        latest_row = nifty_df.iloc[-1]

        exit_price = None
        reason = None

        # Target check
        if current_price >= position['target']:
            exit_price = current_price
            reason = "Target Hit"
        else:
            # Update trailing stop-loss if profit >= 20 points
            profit = current_price - position['entry_price']
            if profit >= 20 and position['trailing_sl'] is None:  
                position['trailing_sl'] = position['entry_price'] + 15  # Lock SL at 15 points profit

            # Check trailing stop-loss first if active
            if position['trailing_sl'] is not None and current_price <= position['trailing_sl']:
                exit_price = current_price
                reason = "Trailing Stop Loss"
            # Then check EMA-based stop-loss
            elif (position['direction'] == "BULLISH" and latest_row['close'] < latest_row['EMA15']) or \
                 (position['direction'] == "BEARISH" and latest_row['close'] > latest_row['EMA15']):
                exit_price = current_price
                reason = "EMA Stop Loss"

        # Check trade duration
        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        time_in_trade = current_time - self.trade_entry_time
        if time_in_trade > self.max_trade_duration:
            exit_price = current_price
            reason = "Time Exit"

        # Exit trade if conditions met
        if exit_price:
            self.exit_trade(exit_price, reason)

    def exit_trade(self, exit_price, reason):
        """Exit active trade"""
        if not self.active_position:
            return

        position = self.active_position
        pnl = (exit_price - position['entry_price']) * position['quantity']
        self.current_capital += pnl

        duration = datetime.now(pytz.timezone('Asia/Kolkata')) - self.trade_entry_time
        duration_mins = duration.total_seconds() / 60

        trade_record = {
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': datetime.now(pytz.timezone('Asia/Kolkata')),
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'tradingsymbol': position['tradingsymbol'],
            'quantity': position['quantity'],
            'direction': position['direction'],
            'strike': position['strike'],
            'trade_duration_mins': duration_mins
        }

        self.trades.append(trade_record)
        logger.info(f"Exited trade: {trade_record}")

        # Reset position
        self.active_position = None
        self.trade_entry_time = None

        # Save trade to CSV
        self.save_trades()

    def save_trades(self):
        """Save trades to CSV file"""
        try:
            df = pd.DataFrame(self.trades)
            df.to_csv("trades.csv", index=False)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def get_latest_indicator_data(self, instrument_key):
        """Get latest data with indicators calculated"""
        with self.data_lock:
            if instrument_key not in self.candle_storage or self.candle_storage[instrument_key].empty:
                # Fetch historical + intraday data if not available
                df = self.merge_historical_with_intraday(instrument_key)
            else:
                df = self.candle_storage[instrument_key].copy()

        if df.empty:
            return pd.DataFrame()

        # Calculate indicators
        return self.calculate_indicators(df)

    def decode_protobuf(self, buffer):
        """Decode protobuf message"""
        try:
            feed_response = pb.FeedResponse()
            feed_response.ParseFromString(buffer)
            return feed_response
        except Exception as e:
            logger.error(f"Error decoding protobuf message: {e}")
            return None

    def start_trading(self):
        """Main function to start the trading bot"""
        logger.info("Starting Upstox Trading Bot...")

        # Clean old data periodically
        Thread(target=self.periodic_cleanup, daemon=True).start()

        # Start WebSocket connection in a separate thread
        loop = asyncio.new_event_loop()
        websocket_thread = Thread(target=self.run_websocket_loop, args=(loop,), daemon=True)
        websocket_thread.start()

        # Start trading strategy loop
        self.run_trading_strategy()

    def run_websocket_loop(self, loop):
        """Run WebSocket in a separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_market_data())

    def periodic_cleanup(self):
        """Periodically clean up old data"""
        while True:
            time.sleep(1800)  # Clean up every 30 minutes
            self.clean_old_data()
            self.clean_ohlc_buffer()

    def run_trading_strategy(self):
        """Run the main trading strategy loop"""
        logger.info("Starting trading strategy...")

        last_print_time = datetime.now(pytz.timezone('Asia/Kolkata'))

        while True:
            try:
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))

                # Print status every 1 minute
                if (current_time - last_print_time).total_seconds() >= 60:
                    self.print_market_status()
                    last_print_time = current_time

                # Check connection health
                if not self.check_connection_health():
                    logger.error("Connection issue detected. Exiting strategy loop.")
                    break

                # Check if market is open
                if not self.is_valid_trading_time(current_time):
                    if current_time.hour >= 15 and current_time.minute > 30:
                        logger.info("Market closed. Exiting strategy loop.")
                        break

                    time.sleep(60)  # Wait for 1 minute and check again
                    continue

                # If we already have an active position, just monitor it
                if self.active_position:
                    time.sleep(10)  # Check less frequently when in position
                    continue

                # Get latest Nifty 50 data and indicators
                nifty_data = self.get_latest_indicator_data("NSE_INDEX|Nifty 50")
                if nifty_data.empty:
                    logger.warning("No Nifty data available. Waiting...")
                    time.sleep(60)
                    continue

                latest_row = nifty_data.iloc[-1]
                signal = self.test_conditions(nifty_data, nifty_data.index[-1], self.prev_trend)

                # Update previous trend
                self.prev_trend = 'bullish' if latest_row['EMA9'] > latest_row['EMA15'] else 'bearish'

                if signal:
                    logger.info(f"Signal detected: {signal}")

                    # Get current Nifty price safely
                    nifty_price = self.get_last_ltp("NSE_INDEX|Nifty 50")
                    if nifty_price is None:
                        logger.warning("No Nifty LTP available")
                        time.sleep(10)
                        continue

                    # Get the best option for the signal
                    instrument_key, trading_symbol, strike = self.get_best_option(nifty_price, signal)

                    if not instrument_key:
                        logger.warning("No suitable option found")
                        time.sleep(10)
                        continue

                    # Subscribe to the option's market data
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    subscribe_task = loop.create_task(self.subscribe_instrument_helper(instrument_key))
                    loop.run_until_complete(subscribe_task)

                    # Wait for initial data
                    for _ in range(5):
                        if self.get_last_ltp(instrument_key) is not None:
                            break
                        time.sleep(1)

                    if self.get_last_ltp(instrument_key) is None:
                        logger.warning(f"No LTP data available for {instrument_key}")
                        time.sleep(10)
                        continue

                    # Enter trade
                    entry_price = self.get_last_ltp(instrument_key)

                    # Calculate stop loss and target
                    stop_loss = entry_price - self.fixed_sl if signal == "BULLISH" else entry_price + self.fixed_sl
                    target = entry_price + self.fixed_tp if signal == "BULLISH" else entry_price - self.fixed_tp

                    # Calculate position size
                    quantity = self.calculate_position_size(entry_price, stop_loss, self.current_capital)

                    # Create the position
                    self.active_position = {
                        'instrument_key': instrument_key,
                        'tradingsymbol': trading_symbol,
                        'direction': signal,
                        'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'trailing_sl': None,
                        'quantity': quantity,
                        'strike': strike
                    }

                    self.trade_entry_time = datetime.now(pytz.timezone('Asia/Kolkata'))

                    logger.info(f"Entered trade: {self.active_position}")

                else:
                    # No signal, wait and check again
                    time.sleep(10)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request error: {e}")
                time.sleep(10)  # Wait and retry
            except pd.errors.EmptyDataError as e:
                logger.error(f"No data available: {e}")
                time.sleep(30)  # Wait longer for data to become available
            except ValueError as e:
                logger.error(f"Value error in trading strategy: {e}")
                time.sleep(5)  # Quick retry for value errors
            except Exception as e:
                logger.error(f"Unexpected error in trading strategy: {e}")
                logger.exception("Stack trace:")  # Log the full stack trace
                time.sleep(30)  # Wait and retry

    async def subscribe_instrument_helper(self, instrument_key):
        """Helper method to subscribe to an instrument"""
        with self.websocket_lock:
            if not self.websocket_ready or self.active_websocket is None:
                logger.warning("WebSocket not ready. Waiting for connection.")
                return False

        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Get WebSocket authorization
        url = "https://api.upstox.com/v2/feed/market-data-feed/authorize"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self.access_token}'}

        try:
            response = requests.get(url, headers=headers)
            res = response.json()

            if 'data' not in res or 'authorizedRedirectUri' not in res['data']:
                logger.error(f"Failed to get WebSocket authorization: {res.get('errors', 'Unknown error')}")
                return False

            websocket_url = res['data']['authorizedRedirectUri']

            async with websockets.connect(websocket_url, ssl=ssl_context) as websocket:
                self.active_websocket = websocket
                self.websocket_ready = True
                await self.subscribe_instrument(websocket, instrument_key)
                return True

        except Exception as e:
            logger.error(f"Error subscribing to instrument: {e}")
            return False

    def visualize_performance(self):
        """Visualize trading performance"""
        if not self.trades:
            logger.warning("No trades to visualize")
            return

        try:
            df = pd.DataFrame(self.trades)

            # Calculate cumulative PnL
            df['cumulative_pnl'] = df['pnl'].cumsum()

            # Plot cumulative PnL
            plt.figure(figsize=(12, 6))
            plt.plot(df['exit_time'], df['cumulative_pnl'])
            plt.title('Cumulative Profit & Loss')
            plt.xlabel('Time')
            plt.ylabel('PnL (INR)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('pnl_chart.png')

            # Calculate statistics
            win_rate = (df['pnl'] > 0).mean() * 100
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if any(df['pnl'] > 0) else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if any(df['pnl'] < 0) else 0

            logger.info(f"Performance: Win Rate: {win_rate:.2f}%, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}")

        except Exception as e:
            logger.error(f"Error visualizing performance: {e}")

    def export_results(self):
        """Export trading results and statistics"""
        if not self.trades:
            logger.warning("No trades to export")
            return

        try:
            df = pd.DataFrame(self.trades)

            # Basic statistics
            total_trades = len(df)
            winning_trades = sum(df['pnl'] > 0)
            losing_trades = sum(df['pnl'] < 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            total_profit = df[df['pnl'] > 0]['pnl'].sum()
            total_loss = df[df['pnl'] < 0]['pnl'].sum()
            net_pnl = total_profit + total_loss

            avg_win = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            avg_trade_duration = df['trade_duration_mins'].mean()

            # Export statistics to CSV
            stats = {
                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate (%)', 
                          'Total Profit', 'Total Loss', 'Net PnL', 'Avg Win', 'Avg Loss', 
                          'Avg Trade Duration (mins)', 'Initial Capital', 'Final Capital', 'Return (%)'],
                'Value': [total_trades, winning_trades, losing_trades, win_rate, 
                         total_profit, total_loss, net_pnl, avg_win, avg_loss, 
                         avg_trade_duration, self.initial_capital, self.current_capital, 
                         ((self.current_capital - self.initial_capital) / self.initial_capital) * 100]
            }

            stats_df = pd.DataFrame(stats)
            stats_df.to_csv('trading_statistics.csv', index=False)

            # Export trades with more details
            df['roi_percent'] = (df['pnl'] / self.initial_capital) * 100
            df.to_csv('detailed_trades.csv', index=False)

            logger.info(f"Results exported to CSV: trading_statistics.csv and detailed_trades.csv")

        except Exception as e:
            logger.error(f"Error exporting results: {e}")

# Main execution
if __name__ == "__main__":
    # Load access token
    try:
        with open("accessToken.txt", "r") as file:
            access_token = file.read().strip()
    except FileNotFoundError:
        logger.error("Access token file not found. Please create 'accessToken.txt' with your Upstox access token.")
        sys.exit(1)

    # Initialize and start the trading bot
    bot = UpstoxTradingBot(access_token, use_ai=False)

    try:
        bot.start_trading()

        # Wait for trading to complete (end of day)
        # This assumes trading_strategy will exit when market closes

        # Export results and visualize performance
        bot.export_results()
        bot.visualize_performance()

    except KeyboardInterrupt:
        logger.info("Trading bot manually stopped.")

        # Export results on manual stop
        bot.export_results()
        bot.visualize_performance()

    except Exception as e:
        logger.error(f"Fatal error in trading bot: {e}")
        logger.exception("Stack trace:")
