Data Science Intermediate Project
# Stock Price Prediction using LSTM and MLP

This project predicts future stock prices using Deep Learning (LSTM) and Machine Learning (MLPRegressor) models. It uses historical stock market data (for example, AAPL.csv) along with metadata from the Kaggle Stock Market Dataset.

## Overview
The project includes:
- Data preprocessing using MinMax scaling
- Model training with both LSTM (deep learning) and MLP (machine learning)
- Interactive Streamlit web app for prediction and visualization
- Future forecasting (next-day and 30-day ahead)

Although the model here was trained on Apple Inc. (AAPL) data, you can easily replace the dataset with any other stock’s CSV file from the same source.

## Dataset
Dataset source: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?select=stocks

You will use two files from the dataset:
1. symbols_valid_meta.csv — contains metadata for all listed companies
2. <SYMBOL>.csv — the specific stock data (for example, AAPL.csv, MSFT.csv, etc.)

Example structure:
AAPL.csv
symbols_valid_meta.csv
models/
   ├─ lstm_aapl_model.h5
   ├─ mlp_aapl_model.joblib
   └─ close_scaler.save
app.py
stock_price_prediction.ipynb

## How It Works
1. Model Training  
Run the Jupyter notebook stock_price_prediction.ipynb to:
   - Load AAPL.csv and symbols_valid_meta.csv
   - Train the LSTM and MLP models
   - Save trained models and scalers in the models/ folder

2. Streamlit Web App  
Run the following command in the terminal:
   streamlit run app.py
Then, open your browser to view the web interface.

The app allows you to:
- Select any stock symbol (default: AAPL)
- View its company information
- See historical closing prices
- Predict the next-day closing price
- Generate a 30-day forecast chart

## Model Details
Model | Framework | Purpose
------|------------|---------
LSTM | TensorFlow / Keras | Sequential deep learning model for time series
MLPRegressor | Scikit-learn | Lightweight fallback model when TensorFlow isn't available

Both models are trained using 60-day lookback windows of normalized closing prices.

## Example Output
Next-Day Predicted Close: 203.47  
30-Day Forecast Chart: Displays predicted price trend overlayed on historical prices.

## Using Other Stocks
To use another stock (for example, MSFT or AMZN):
1. Download the desired <SYMBOL>.csv file from the Kaggle dataset.
2. Place it in the same directory as the app.
3. Run the app again — it will automatically detect and use that CSV.

## Requirements
Install dependencies:
   pip install pandas numpy scikit-learn tensorflow streamlit matplotlib joblib
If TensorFlow fails on your system, the app will automatically switch to the MLPRegressor model.

## Project Files
File | Description
------|--------------
stock_price_prediction.ipynb | Model training and evaluation
app.py | Streamlit dashboard for interactive predictions
symbols_valid_meta.csv | Company metadata
<SYMBOL>.csv | Historical stock data
models/ | Saved models and scaler

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)
