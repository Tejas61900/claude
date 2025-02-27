import asyncio
import json
import ssl
import upstox_client
import websockets
import pandas as pd
import numpy as np
import requests
import pytz
from google.protobuf.json_format import MessageToDict
from threading import Thread
import MarketDataFeedV3_pb2 as pb
from time import sleep, time
import ta
from datetime import datetime, timedelta
from collections import deque
import math

# Read access token
filename = "accessToken.txt"
with open(filename, "r") as file:
    access_token = file.read()

# Global variables for storing market data
ohlc_buffer = {}
ohlc_history = deque(maxlen=100)  # Store last 100 candles for calculations
last_ltp = None
last_print_time = 0
indicators = {}

# Trade tracking variables
current_trade = None
trade_history = []
can_trade = True
IST = pytz.timezone('Asia/Kolkata')

# Define Trade class for paper trading
class Trade:
    def __init__(self, trade_type, entry_price, entry_time, lot_size=1):
        self.trade_type = trade_type  # 'CALL' or 'PUT'
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.lot_size = lot_size
        self.exit_price = None
        self.exit_time = None
        self.initial_stop_loss = None
        self.current_stop_loss = None
        self.target = None
        self.pnl = 0
        self.exit_reason = None
        self.option_details = None

    def set_option_details(self, strike, premium, expiry, instrument_key=None):
        """Set the option contract details"""
        self.option_details = {
            'strike': strike,
            'premium': premium,
            'expiry': expiry,
            'instrument_key': instrument_key
        }

    def set_stop_loss(self, price):
        """Set initial stop loss"""
        self.initial_stop_loss = price
        self.current_stop_loss = price

    def set_target(self, price):
        """Set profit target"""
        self.target = price

    def update_trailing_sl(self, new_sl):
        """Update stop loss for trailing"""
        self.current_stop_loss = new_sl

    def close_trade(self, exit_price, exit_time, reason):
        """Close the trade with exit details"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        # Calculate P&L
        if self.trade_type == 'CALL':
            self.pnl = (exit_price - self.entry_price) * self.lot_size
        else:  # PUT
            self.pnl = (exit_price - self.entry_price) * self.lot_size

    def get_trade_duration(self):
        """Calculate trade duration"""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None

    def to_dict(self):
        """Convert trade to dictionary for logging"""
        return {
            'trade_type': self.trade_type,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M:%S') if self.exit_time else None,
            'initial_stop_loss': self.initial_stop_loss,
            'target': self.target,
            'pnl': self.pnl,
            'exit_reason': self.exit_reason,
            'option_details': self.option_details
        }


def get_market_data_feed_authorize(api_version, configuration):
    """Get authorization for market data feed."""
    api_instance = upstox_client.WebsocketApi(upstox_client.ApiClient(configuration))
    api_response = api_instance.get_market_data_feed_authorize(api_version)
    return api_response


def decode_protobuf(buffer):
    """Decode protobuf message."""
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response


def fetch_historical_candles(instrument_key="NSE_INDEX|Nifty 50", lookback_candles=100):
    """Fetch historical 1-minute candles to initialize indicators"""

    # Set up API headers with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Format the instrument_key for the URL (replace | with %7C)
    instrument_key_encoded = instrument_key.replace("|", "%7C")

    # API endpoint for historical candles
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key_encoded}/1minute"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            candles = data.get("data", {}).get("candles", [])

            # Process candles and add to history
            for candle in candles[-lookback_candles:]:
                timestamp = candle[0]  # Timestamp is first element
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Convert to Kolkata time
                dt_ist = dt.astimezone(IST)

                ohlc = {
                    "timestamp": int(dt.timestamp() * 1000),
                    "datetime": dt_ist,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]) if len(candle) > 5 else 0
                }
                ohlc_history.append(ohlc)

            print(f"✅ Loaded {len(candles)} historical candles")
            return True
        else:
            print(f"❌ Failed to fetch historical data: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception fetching historical data: {e}")
        return False


def calculate_indicators():
    """Calculate technical indicators from OHLC data"""

    global indicators, ohlc_history

    if len(ohlc_history) < 30:
        print(f"⚠️ Not enough data for indicators calculation: {len(ohlc_history)} candles available")
        return False

    # Convert ohlc_history to DataFrame
    df = pd.DataFrame([{
        'timestamp': item['timestamp'],
        'datetime': item.get('datetime', None),
        'open': item['open'],
        'high': item['high'], 
        'low': item['low'],
        'close': item['close'],
        'volume': item.get('volume', 0)
    } for item in ohlc_history])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp')

    # Calculate EMA-9 and EMA-15
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema15'] = ta.trend.ema_indicator(df['close'], window=15)

    # Calculate EMA differences (gradient)
    df['ema_diff'] = df['ema9'] - df['ema15']
    df['ema_diff_prev'] = df['ema_diff'].shift(1)

    # Calculate the angle of difference change (in degrees)
    df['ema_angle'] = np.degrees(np.arctan((df['ema_diff'] - df['ema_diff_prev']) / df['close'].mean() * 100))

    # Calculate RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Calculate ADX
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()

    # Calculate ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Calculate MACD
    macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Calculate candle strength (body size relative to total range)
    df['body_size'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['candle_strength'] = df['body_size'] / df['total_range'].replace(0, 0.001)  # Avoid div by zero

    # Calculate if price is retesting EMA9 (within ±10 points)
    df['ema9_retest'] = (abs(df['close'] - df['ema9']) <= 10)

    # Calculate crossover signals
    df['ema_crossover'] = 0
    for i in range(1, len(df)):
        if df['ema9'].iloc[i-1] <= df['ema15'].iloc[i-1] and df['ema9'].iloc[i] > df['ema15'].iloc[i]:
            df.loc[i, 'ema_crossover'] = 1  # Bullish crossover
        elif df['ema9'].iloc[i-1] >= df['ema15'].iloc[i-1] and df['ema9'].iloc[i] < df['ema15'].iloc[i]:
            df.loc[i, 'ema_crossover'] = -1  # Bearish crossover

    # Store last few rows for strategy calculation
    prev_rows = df.iloc[-5:].copy()

    # Get the latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    indicators = {
        'ema9': latest['ema9'],
        'ema15': latest['ema15'],
        'ema_angle': latest['ema_angle'],
        'rsi': latest['rsi'],
        'adx': latest['adx'],
        'atr': latest['atr'],
        'macd': latest['macd'],
        'macd_signal': latest['macd_signal'],
        'macd_diff': latest['macd_diff'],
        'candle_strength': latest['candle_strength'],
        'ema9_retest': latest['ema9_retest'],
        'ema_crossover': latest['ema_crossover'],
        'prev_rows': prev_rows
    }

    return True


def get_next_expiry_date():
    """
    Get the next Thursday expiry date

    Returns:
        datetime: Next Thursday's date in IST
        str: Formatted date string in YYYY-MM-DD format
    """
    today = datetime.now(IST)
    days_to_thursday = (3 - today.weekday()) % 7

    # If today is Thursday and before market close (3:30 PM IST)
    if days_to_thursday == 0 and today.hour < 15 or (today.hour == 15 and today.minute < 30):
        next_expiry = today
    else:
        # If today is Thursday after market close, or any other day, get next Thursday
        if days_to_thursday == 0:  # Thursday after market close
            days_to_thursday = 7
        next_expiry = today + timedelta(days=days_to_thursday)

    # Format for API request
    expiry_date_str = next_expiry.strftime('%Y-%m-%d')

    return next_expiry, expiry_date_str


def fetch_option_chain(spot_price, trade_type):
    """
    Fetch option chain and find the closest ITM option

    Args:
        spot_price (float): Current spot price of Nifty
        trade_type (str): 'CALL' or 'PUT'

    Returns:
        dict: Selected option details
    """
    next_expiry, expiry_date_str = get_next_expiry_date()

    url = 'https://api.upstox.com/v2/option/chain'
    params = {
        'instrument_key': 'NSE_INDEX|Nifty 50',
        'expiry_date': expiry_date_str
    }
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    try:
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()

            if data.get('status') == 'success' and data.get('data'):
                # Round spot price to nearest 50
                rounded_spot = round(spot_price / 50) * 50

                # Find the closest ITM option
                closest_strike = None
                closest_diff = float('inf')
                selected_option = None

                for option_data in data['data']:
                    strike = option_data.get('strike_price')

                    # For CALL, we want strike below spot (ITM)
                    # For PUT, we want strike above spot (ITM)
                    if trade_type == 'CALL':
                        if strike < spot_price:
                            diff = spot_price - strike
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_strike = strike
                                selected_option = option_data
                    else:  # PUT
                        if strike > spot_price:
                            diff = strike - spot_price
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_strike = strike
                                selected_option = option_data

                if selected_option:
                    option_key = 'call_options' if trade_type == 'CALL' else 'put_options'
                    option_details = selected_option.get(option_key, {})

                    expiry_formatted = next_expiry.strftime('%d-%b-%Y')

                    return {
                        'strike': closest_strike,
                        'premium': option_details.get('market_data', {}).get('ltp', 0),
                        'expiry': expiry_formatted,
                        'instrument_key': option_details.get('instrument_key'),
                        'delta': option_details.get('option_greeks', {}).get('delta', 0),
                        'iv': option_details.get('option_greeks', {}).get('iv', 0),
                        'underlying_spot': spot_price
                    }

                # Fallback if no ideal option found
                print(f"⚠️ No suitable {trade_type} option found, using nearest strike")
                # Get nearest strike (can be OTM if no ITM available)
                available_strikes = [opt.get('strike_price') for opt in data['data']]
                if available_strikes:
                    nearest_strike = min(available_strikes, key=lambda x: abs(x - spot_price))
                    for option_data in data['data']:
                        if option_data.get('strike_price') == nearest_strike:
                            option_key = 'call_options' if trade_type == 'CALL' else 'put_options'
                            option_details = option_data.get(option_key, {})

                            expiry_formatted = next_expiry.strftime('%d-%b-%Y')

                            return {
                                'strike': nearest_strike,
                                'premium': option_details.get('market_data', {}).get('ltp', 0),
                                'expiry': expiry_formatted,
                                'instrument_key': option_details.get('instrument_key'),
                                'delta': option_details.get('option_greeks', {}).get('delta', 0),
                                'iv': option_details.get('option_greeks', {}).get('iv', 0),
                                'underlying_spot': spot_price
                            }

            print(f"⚠️ Couldn't find option data in response: {data}")
            return None
        else:
            print(f"❌ Failed to fetch option chain: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception fetching option chain: {e}")
        return None


def check_entry_conditions():
    """Check if entry conditions are met for a new trade"""

    global indicators, current_trade, can_trade

    if not can_trade or current_trade is not None:
        return None  # Already in a trade or trading disabled

    # Check if we have all the required indicators
    required_keys = ['ema9', 'ema15', 'ema_angle', 'rsi', 
                     'adx', 'macd', 'macd_signal', 'candle_strength', 'ema9_retest']
    if not all(key in indicators for key in required_keys):
        return None

    # Get the most recent candle and indicators
    latest_candle = ohlc_history[-1]
    close_price = latest_candle['close']

    # Initialize signal and reason
    signal = None
    reasons = []

    # 1. Check EMA Crossover (from previous candle)
    crossover = indicators.get('ema_crossover', 0)

    # Try to detect crossover within last 3 candles
    prev_rows = indicators.get('prev_rows', pd.DataFrame())
    recent_crossover = 0

    if len(prev_rows) >= 3:
        for i in range(len(prev_rows) - 1):
            if prev_rows['ema9'].iloc[i] <= prev_rows['ema15'].iloc[i] and prev_rows['ema9'].iloc[i+1] > prev_rows['ema15'].iloc[i+1]:
                recent_crossover = 1  # Bullish crossover
                break
            elif prev_rows['ema9'].iloc[i] >= prev_rows['ema15'].iloc[i] and prev_rows['ema9'].iloc[i+1] < prev_rows['ema15'].iloc[i+1]:
                recent_crossover = -1  # Bearish crossover
                break

    # 2. Check EMA Angle (≥ 15° for strong trend)
    ema_angle = abs(indicators['ema_angle'])

    # 3. Check Price Retest (must retest EMA9 within ±10 points)
    ema9_retest = indicators['ema9_retest']

    # 4. Check Candle Strength (> 0.6 for strong momentum)
    candle_strength = indicators['candle_strength']

    # 5. Check MACD Confirmation
    macd = indicators['macd']
    macd_signal = indicators['macd_signal']
    macd_hist = indicators['macd_diff']

    # 6. Check RSI (avoid overbought/oversold)
    rsi = indicators['rsi']

    # 7. Check ADX (> 20 for strong trend)
    adx = indicators['adx']

    # Evaluate CALL signal
    if (recent_crossover == 1 or crossover == 1) and indicators['ema9'] > indicators['ema15']:
        # Bullish setup
        reasons.append("✅ EMA9 crossed above EMA15")

        if ema_angle >= 15:
            reasons.append("✅ EMA angle >= 5°")
        else:
            reasons.append("❌ EMA angle < 5°")
            return None

        if ema9_retest:
            reasons.append("✅ Price retested EMA9")
        else:
            reasons.append("❌ Price did not retest EMA9")
            return None

        if candle_strength > 0.6:
            reasons.append("✅ Candle strength > 0.6")
        else:
            reasons.append("❌ Candle strength <= 0.6")
            return None

        if macd > macd_signal:
            reasons.append("✅ MACD > Signal line")
        else:
            reasons.append("❌ MACD < Signal line")
            return None

        if 30 <= rsi <= 70:
            reasons.append("✅ RSI between 30-70")
        else:
            reasons.append("❌ RSI outside 30-70 range")
            return None

        if adx > 20:
            reasons.append("✅ ADX > 20")
        else:
            reasons.append("❌ ADX <= 20")
            return None

        signal = "CALL"

    # Evaluate PUT signal
    elif (recent_crossover == -1 or crossover == -1) and indicators['ema9'] < indicators['ema15']:
        # Bearish setup
        reasons.append("✅ EMA9 crossed below EMA15")

        if ema_angle >= 5:
            reasons.append("✅ EMA angle >= 5°")
        else:
            reasons.append("❌ EMA angle < 5°")
            return None

        if ema9_retest:
            reasons.append("✅ Price retested EMA9")
        else:
            reasons.append("❌ Price did not retest EMA9")
            return None

        if candle_strength > 0.6:
            reasons.append("✅ Candle strength > 0.6")
        else:
            reasons.append("❌ Candle strength <= 0.6")
            return None

        if macd < macd_signal:
            reasons.append("✅ MACD < Signal line")
        else:
            reasons.append("❌ MACD > Signal line")
            return None

        if 30 <= rsi <= 70:
            reasons.append("✅ RSI between 30-70")
        else:
            reasons.append("❌ RSI outside 30-70 range")
            return None

        if adx > 20:
            reasons.append("✅ ADX > 20")
        else:
            reasons.append("❌ ADX <= 20")
            return None

        signal = "PUT"

    # If signal is valid, create new trade
    if signal:
        current_time = latest_candle['datetime'] if 'datetime' in latest_candle else datetime.now(IST)

        # Create a new trade
        new_trade = Trade(signal, close_price, current_time)

        # Set stop loss at EMA15
        stop_loss = indicators['ema15']

        # Set target at entry price ±30 points
        if signal == "CALL":
            target = close_price + 30
        else:  # PUT
            target = close_price + 30

        new_trade.set_stop_loss(stop_loss)
        new_trade.set_target(target)

        # Fetch option contract details from option chain
        option_details = fetch_option_chain(close_price, signal)

        if option_details:
            # Set option details with actual market data
            new_trade.set_option_details(
                option_details['strike'],
                option_details['premium'],
                option_details['expiry'],
                option_details['instrument_key']
            )

            # Log the trade entry
            entry_log = {
                "time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                "action": f"ENTRY: {signal}",
                "price": close_price,
                "reasons": reasons,
                "stop_loss": stop_loss,
                "target": target,
                "option": f"{signal} {option_details['strike']} {option_details['expiry']}"
            }

            print("\n" + "="*60)
            print(f"🚀 NEW TRADE: {signal} at {close_price}")
            for reason in reasons:
                print(reason)
            print(f"📍 Stop Loss: {stop_loss}")
            print(f"🎯 Target: {target}")
            print(f"🎟️ Option: {signal} {option_details['strike']} {option_details['expiry']}")
            print(f"💰 Premium: {option_details['premium']}")
            print(f"🔑 Instrument Key: {option_details['instrument_key']}")
            print(f"📊 IV: {option_details.get('iv', 'N/A')}%  | Delta: {option_details.get('delta', 'N/A')}")
            print("="*60 + "\n")

            return new_trade
        else:
            print(f"⚠️ Could not fetch option details, using estimated values")

            # Mock option selection details as fallback
            strike = round(close_price / 50) * 50  # Round to nearest 50
            if signal == "CALL":
                strike -= 50  # Slightly ITM
            else:
                strike += 50  # Slightly ITM

            # Mock premium
            premium = round(indicators['atr'] * 0.8, 1)

            # Mock expiry
            next_expiry, _ = get_next_expiry_date()
            expiry = next_expiry.strftime('%d-%b-%Y')

            new_trade.set_option_details(strike, premium, expiry)

            print("\n" + "="*60)
            print(f"🚀 NEW TRADE: {signal} at {close_price}")
            for reason in reasons:
                print(reason)
            print(f"📍 Stop Loss: {stop_loss}")
            print(f"🎯 Target: {target}")
            print(f"🎟️ Option: {signal} {strike} {expiry} (ESTIMATED)")
            print(f"💰 Premium: {premium} (ESTIMATED)")
            print("="*60 + "\n")

            return new_trade

    return None


def check_exit_conditions(latest_candle):
    """Check if exit conditions are met for the current trade"""
    global current_trade, can_trade

    if not current_trade:
        return None

    close_price = latest_candle['close']
    current_time = latest_candle['datetime'] if 'datetime' in latest_candle else datetime.now(IST)
    exit_reason = None

    # Check if target hit
    if current_trade.trade_type == 'CALL' and close_price >= current_trade.target:
        exit_reason = "TARGET HIT"
    elif current_trade.trade_type == 'PUT' and close_price <= current_trade.target:
        exit_reason = "TARGET HIT"

    # Check if stop loss hit
    elif current_trade.trade_type == 'CALL' and close_price <= current_trade.current_stop_loss:
        exit_reason = "STOP LOSS HIT"
    elif current_trade.trade_type == 'PUT' and close_price >= current_trade.current_stop_loss:
        exit_reason = "STOP LOSS HIT"

    # Update trailing stop loss if applicable
    if current_trade.trade_type == 'CALL' and close_price >= (current_trade.entry_price + 20):
        new_sl = max(current_trade.current_stop_loss, current_trade.entry_price + 15)
        if new_sl > current_trade.current_stop_loss:
            current_trade.update_trailing_sl(new_sl)
            print(f"🔄 Updated Trailing SL: {new_sl}")

    elif current_trade.trade_type == 'PUT' and close_price <= (current_trade.entry_price - 20):
        new_sl = min(current_trade.current_stop_loss, current_trade.entry_price - 15)
        if new_sl < current_trade.current_stop_loss:
            current_trade.update_trailing_sl(new_sl)
            print(f"🔄 Updated Trailing SL: {new_sl}")

    # If exit condition met, close the trade
    if exit_reason:
        current_trade.close_trade(close_price, current_time, exit_reason)

        # Log the exit
        pnl = current_trade.pnl
        pnl_emoji = "✅" if pnl > 0 else "❌"

        print("\n" + "="*60)
        print(f"{pnl_emoji} TRADE CLOSED: {current_trade.trade_type} at {close_price}")
        print(f"📊 Entry: {current_trade.entry_price} | Exit: {close_price}")
        print(f"💰 P&L: {pnl:.2f} points")
        print(f"⏱️ Duration: {current_trade.get_trade_duration()}")
        print(f"📝 Reason: {exit_reason}")
        print("="*60 + "\n")

        # Add to trade history
        trade_history.append(current_trade.to_dict())

        # Reset current trade and block re-entry temporarily
        temp_trade = current_trade
        current_trade = None

        # Allow re-entry after trade closure
        can_trade = True

        return temp_trade

    return None


async def fetch_market_data():
    """Fetch market data using WebSocket and store updates."""

    global ohlc_buffer, last_ltp, ohlc_history, current_trade, can_trade

    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Configure OAuth2 access token
    configuration = upstox_client.Configuration()
    api_version = "3.0"
    configuration.access_token = access_token

    # Get WebSocket authorization
    response = get_market_data_feed_authorize(api_version, configuration)

    async with websockets.connect(response.data.authorized_redirect_uri, ssl=ssl_context) as websocket:
        print("✅ Connection established")

        await asyncio.sleep(1)  # Wait briefly

        # Subscribe to Nifty 50 index data
        data = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_INDEX|Nifty 50"]
            }
        }

        # Convert to binary and send over WebSocket
        binary_data = json.dumps(data).encode("utf-8")
        await websocket.send(binary_data)

        # Variable to track the last candle timestamp
        last_candle_timestamp = None

        # Continuously receive and store market data
        while True:
            message = await websocket.recv()
            decoded_data = decode_protobuf(message)
            data_dict = MessageToDict(decoded_data)

            if "feeds" in data_dict:
                try:
                    nifty_data = data_dict["feeds"].get("NSE_INDEX|Nifty 50", {}).get("fullFeed", {}).get("indexFF", {})

                    # Extract LTP
                    if isinstance(nifty_data.get("ltpc"), dict) and "ltp" in nifty_data["ltpc"]:
                        last_ltp = float(nifty_data["ltpc"]["ltp"])

                    # Extract OHLC (Only 1-Minute Interval)
                    ohlc_data = nifty_data.get("marketOHLC", {}).get("ohlc", [])
                    if ohlc_data:
                        # Filter only "I1" (1-minute) OHLC data
                        ohlc_1m_entries = [entry for entry in ohlc_data if entry["interval"] == "I1"]
                        if ohlc_1m_entries:
                            latest_ohlc_1m = max(ohlc_1m_entries, key=lambda x: int(x["ts"]))

                            # Convert timestamp to datetime with IST timezone
                            ts = int(latest_ohlc_1m["ts"])
                            dt = datetime.fromtimestamp(ts/1000)
                            dt_ist = dt.astimezone(IST)

                            # Update the ohlc_buffer
                            ohlc_buffer["open"] = float(latest_ohlc_1m["open"])
                            ohlc_buffer["high"] = float(latest_ohlc_1m["high"])
                            ohlc_buffer["low"] = float(latest_ohlc_1m["low"])
                            ohlc_buffer["close"] = float(latest_ohlc_1m["close"])
                            ohlc_buffer["timestamp"] = ts
                            ohlc_buffer["datetime"] = dt_ist

                            # If this is a new candle, add to history
                            if last_candle_timestamp != ohlc_buffer["timestamp"]:
                                ohlc_history.append(ohlc_buffer.copy())
                                last_candle_timestamp = ohlc_buffer["timestamp"]

                                # Calculate indicators on new candle
                                if calculate_indicators():
                                    # Check for exit conditions first
                                    exit_trade = check_exit_conditions(ohlc_buffer)

                                    # If no current trade, check for entry
                                    if current_trade is None and can_trade:
                                        new_trade = check_entry_conditions()
                                        if new_trade:
                                            current_trade = new_trade

                except KeyError as e:
                    print(f"⚠️ KeyError: {e}")
            else:
                print("⏳ Waiting for market data...")


def run_websocket():
    """Run WebSocket in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_market_data())


def log_trade_summary():
    """Log a summary of all completed trades"""

    if not trade_history:
        print("No completed trades to summarize.")
        return

    total_trades = len(trade_history)
    winning_trades = sum(1 for trade in trade_history if trade['pnl'] > 0)
    losing_trades = total_trades - winning_trades

    total_profit = sum(trade['pnl'] for trade in trade_history if trade['pnl'] > 0)
    total_loss = sum(trade['pnl'] for trade in trade_history if trade['pnl'] < 0)
    net_pnl = total_profit + total_loss

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    print("\n" + "="*80)
    print("📊 TRADING SESSION SUMMARY")
    print("="*80)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: {total_profit:.2f} points")
    print(f"Total Loss: {total_loss:.2f} points")
    print(f"Net P&L: {net_pnl:.2f} points")
    print("="*80)
    print("\nTrade Details:")

    for i, trade in enumerate(trade_history, 1):
        print(f"\nTrade #{i}:")
        print(f"Type: {trade['trade_type']}")
        print(f"Entry: {trade['entry_price']} at {trade['entry_time']}")
        print(f"Exit: {trade['exit_price']} at {trade['exit_time']}")
        print(f"P&L: {trade['pnl']:.2f} points")
        print(f"Reason: {trade['exit_reason']}")
        print(f"Option: {trade.get('option_details', 'N/A')}")

    print("="*80 + "\n")


def main():
    # Fetch historical data first to initialize indicators
    print("📜 Fetching historical candles to initialize indicators...")
    fetch_historical_candles()
    # Calculate initial indicators
    calculate_indicators()
    # Start WebSocket in a separate thread
    print("🔌 Starting WebSocket connection...")
    websocket_thread = Thread(target=run_websocket)
    websocket_thread.daemon = True  # Thread will exit when main thread exits
    websocket_thread.start()
    # Print indicators and OHLC every minute
    sleep(5)  # Give time for connection to establish
    last_print_time = 0
    try:
        while True:
            sleep(1)
            current_time = time()
            if current_time - last_print_time >= 60:  # Print once per minute
                last_print_time = current_time
                if last_ltp is not None and ohlc_buffer and indicators:
                    # Format timestamp as readable date/time
                    if 'datetime' in ohlc_buffer:
                        dt_str = ohlc_buffer['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        dt_str = "N/A"
                    print("\n" + "-"*50)
                    print(f"⏱️ Time: {dt_str}")
                    print(f"💹 Nifty 50: {last_ltp:.2f}")
                    print(f"📊 OHLC: O={ohlc_buffer['open']:.2f} H={ohlc_buffer['high']:.2f} L={ohlc_buffer['low']:.2f} C={ohlc_buffer['close']:.2f}")
                    print(f"📈 EMA9: {indicators['ema9']:.2f} | EMA15: {indicators['ema15']:.2f}")
                    print(f"📐 EMA Angle: {indicators['ema_angle']:.2f}°")
                    print(f"🔋 RSI: {indicators['rsi']:.2f} | ADX: {indicators['adx']:.2f}")
                    print(f"📶 MACD: {indicators['macd']:.2f} | Signal: {indicators['macd_signal']:.2f}")
                    if current_trade:
                        print(f"\n🔴 ACTIVE TRADE: {current_trade.trade_type}")
                        print(f"⏬ Entry: {current_trade.entry_price:.2f}")
                        print(f"🛑 Stop Loss: {current_trade.current_stop_loss:.2f}")
                        print(f"🎯 Target: {current_trade.target:.2f}")
                        current_pnl = 0
                        if current_trade.trade_type == 'CALL':
                            current_pnl = ohlc_buffer['close'] - current_trade.entry_price
                        else:
                            current_pnl = current_trade.entry_price - ohlc_buffer['close']
                        pnl_emoji = "✅" if current_pnl > 0 else "❌"
                        print(f"{pnl_emoji} Current P&L: {current_pnl:.2f} points")
                    else:
                        print("\n⏳ No active trade")
                    print("-"*50)
    except KeyboardInterrupt:
        print("\n👋 User interrupted program. Closing...")
        log_trade_summary()
        print("🔌 Disconnecting from WebSocket...")
        print("✅ Done!")

if __name__ == "__main__":
    main()
