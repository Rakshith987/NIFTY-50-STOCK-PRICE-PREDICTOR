import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Load NIFTY 50 stock details from CSV
@st.cache_data
def load_stock_info():
    try:
        df = pd.read_csv("nifty50_stocks_data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Make sure 'nifty50_stocks_data.csv' is in the correct directory.")
        return pd.DataFrame()

# Fetch historical stock prices from Yahoo Finance
@st.cache_data
def fetch_historical_data(symbol):
    try:
        stock = yf.download(symbol, start="2014-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        return stock if not stock.empty else pd.DataFrame()  # Ensure it returns a DataFrame
    except Exception as e:
        st.warning(f"⚠️ Failed to load data for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame instead of None

# Streamlit App Config
st.set_page_config(page_title="NIFTY 50 Stock Price Prediction", layout="wide")
st.title("📈 NIFTY 50 Stock Price Prediction")

# Load stock details
df = load_stock_info()

# If CSV failed to load, stop execution
if df.empty:
    st.stop()

# Sidebar for Stock Selection
st.sidebar.header("Select Stock")
stock_symbol = st.sidebar.selectbox("Choose a stock", df["Symbol"].unique())
forecast_days = st.sidebar.number_input("Forecast Days", min_value=1, max_value=30, step=1, value=7)

# Fetch historical data for selected stock
nse_symbol = stock_symbol + ".NS"  # Convert to Yahoo Finance format
stock_data = fetch_historical_data(nse_symbol)

# Display Stock Info
st.subheader(f"Stock Details: {stock_symbol}")
stock_info = df[df["Symbol"] == stock_symbol]
st.dataframe(stock_info)  # Show stock details from CSV

# If data failed to load, stop execution
if stock_data.empty:
    st.error("❌ Could not fetch historical data. Please try a different stock.")
    st.stop()

st.subheader(f"📊 Historical Data for {stock_symbol}")
st.dataframe(stock_data.tail(10))  # Show last 10 days of stock data

# Train the SVR Model
def train_model(data, days):
    if len(data) <= days:
        st.warning("⚠️ Not enough historical data to predict. Try a lower forecast period.")
        return None, None  # Fix: Return tuple instead of a single None

    X = np.array(data["Close"]).reshape(-1, 1)
    y = np.array(data["Close"].shift(-days))

    # Remove NaN values
    X, y = X[:-days], y[:-days]
    
    if len(X) < 2:
        st.warning("⚠️ Dataset is too small for training.")
        return None, None  # Fix: Return tuple

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVR(kernel="rbf", C=1000.0, gamma=0.0001)
    model.fit(X_train, y_train)

    # Generate future dates
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    future_prices = model.predict(X[-days:].reshape(-1, 1))

    return future_dates, future_prices  # Fix: Now properly returns both values

# Predict and Display Results
st.subheader("📊 Predicted Future Stock Prices")
future_dates, predicted_prices = train_model(stock_data, forecast_days)

if predicted_prices is not None:
    future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})
    st.dataframe(future_df)

    # Plot Stock Price & Moving Average
    st.subheader("📈 Stock Price and Moving Average Graph")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual stock prices
    ax.plot(stock_data.index, stock_data["Close"], label="Stock Price", color="blue", linestyle="-")

    # Moving Average (e.g., 10-day average)
    stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
    ax.plot(stock_data.index, stock_data["MA10"], label="10-Day Moving Avg", color="red", linestyle="--")

    # Plot predictions
    ax.plot(future_dates, predicted_prices, label="Predicted Prices", color="green", linestyle="--", marker="o")

    # Labels & Legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (₹)")
    ax.set_title(f"{stock_symbol} Price Prediction & Moving Average")
    ax.legend()
    ax.grid()

    # Display Plot
    st.pyplot(fig)

else:
    st.info("🔹 Try selecting a different stock or reducing forecast days.")

st.info("🔹 This is a basic model for educational purposes. Not for financial decisions.")
