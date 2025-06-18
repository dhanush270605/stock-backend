from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI warnings
import matplotlib.pyplot as plt
import io
import base64
from datetime import date
import pandas as pd
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS

def predict(company, days):
    try:
        # Validate inputs
        if not company:
            raise ValueError("Company ticker is required.")
        if not isinstance(days, int):
            raise ValueError("Days must be an integer.")
        if days <= 0 or days > 365:
            raise ValueError("Days must be between 1 and 365.")

        # Download stock data
        df = yf.download(company, start='2015-01-01', end=date.today().strftime('%Y-%m-%d'), auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data found for ticker {company}.")
        if len(df) < 60:
            raise ValueError(f"Insufficient data for ticker {company}. At least 60 days of data required.")

        data = df[['Close']].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        window = 60

        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i - window:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Build and train model
        model = Sequential([
            Input(shape=(window, 1)),  # Use Input layer
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # Make predictions
        last_in = scaled[-window:].reshape(1, window, 1)
        preds = []
        for _ in range(days):
            p = model.predict(last_in, verbose=0)[0]
            preds.append(p)
            last_in = np.append(last_in[:, 1:, :], [[p]], axis=1)

        future = scaler.inverse_transform(preds).flatten().tolist()
        last_date = data.index[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
        # Keep dates as datetime objects (no strftime)

        return future, dates, data

    except Exception as e:
        print(f"Prediction error for {company}: {str(e)}")
        raise e

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        body = request.get_json()
        if not body:
            return jsonify({'error': 'Request body is empty.'}), 400

        company = body.get('company')
        days = int(body.get('days', 30))
        print(f"Processing prediction for {company} over {days} days")  # Debug log

        op, dates, data = predict(company, days)

        # Create plot
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        ax.plot(data.index[-100:], data['Close'][-100:], label='Actual')
        ax.plot(dates, op, label='Forecast', color='green')
        ax.set_title(f'{company} Forecast for {days} days')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
        buf.close()

        # Convert dates to strings for JSON serialization
        dates_str = dates.strftime('%Y-%m-%d').tolist()

        return jsonify({
            'company': company,
            'days': days,
            'op': op,
            'dates': dates_str,
            'image': img
        })

    except ValueError as ve:
        print(f"Value error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)