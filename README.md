# LSTM-Based Stock Price Prediction for NIFTY 50 & HUL

## 📌 Project Overview
Stock market prediction is a crucial yet challenging task due to market volatility and numerous influencing factors. This project leverages **Long Short-Term Memory (LSTM)** networks to predict stock prices for **NIFTY 50** (a benchmark index of India's top 50 companies) and **Hindustan Unilever Limited (HUL)** (a leading FMCG company). The deep learning model captures sequential dependencies in stock price movements, enhancing prediction accuracy.

## 🎯 Objectives
- Develop an **LSTM-based model** to predict stock prices using historical data.
- Compare stock trends for **NIFTY 50 (market index)** and **HUL (sectoral leader in FMCG)**.
- Evaluate the impact of **macroeconomic factors, technical indicators, and financial sentiment**.
- Provide data-driven insights for **investors, traders, and portfolio managers**.

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍
- **Libraries:**
  - `numpy`, `pandas` (Data Handling)
  - `matplotlib`, `seaborn` (Data Visualization)
  - `yfinance` (Stock Data Retrieval)
  - `tensorflow.keras` (LSTM Model Development)
  - `scikit-learn` (Performance Evaluation)

## 📂 Project Structure
```
├── data/                     # Stock market datasets
├── notebooks/                # Jupyter Notebooks for model development
├── src/                      # Python scripts for model training and evaluation
│   ├── data_preprocessing.py # Data cleaning & normalization
│   ├── lstm_model.py         # LSTM model architecture
│   ├── train_model.py        # Model training script
│   ├── evaluate_model.py     # Performance evaluation
│   ├── visualization.py      # Graphs & insights
├── results/                  # Prediction outputs & model performance metrics
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## 🔍 Data Collection & Preprocessing
- Stock price data is sourced from **Yahoo Finance (`yfinance`)**.
- **Features include:** Open, High, Low, Close, Volume.
- Data normalization using **MinMaxScaler**.
- Train-Test split (80-20) and reshaping into a **3D format (samples, time steps, features)** for LSTM compatibility.

## 🔧 Model Development
- **Architecture:**
  - LSTM layers to capture sequential dependencies.
  - Dropout layers to reduce overfitting.
  - Dense layers with a linear activation function for continuous price prediction.
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

## 📊 Performance Evaluation
- **Metrics Used:**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- **Visualization:**
  - Predicted vs. Actual stock price graphs
  - Error analysis

## 💡 Key Insights & Recommendations
- The model **effectively captures price trends** but may struggle with sudden market fluctuations.
- **HUL’s price movements** are less volatile than NIFTY 50, making it a stable defensive stock.
- Integrating **macroeconomic indicators & sentiment analysis** could improve accuracy.
- **Hybrid AI models (LSTM + GRU + XGBoost)** may enhance predictive performance.

## 🔥 Future Enhancements
- Implement **reinforcement learning** for dynamic investment strategies.
- Use **tick-level data** for high-frequency trading applications.
- Enhance interpretability using **SHAP & LIME** for AI-driven financial decisions.
- Improve risk assessment using **Sharpe Ratio & Sortino Ratio**.

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/lstm-stock-prediction.git
cd lstm-stock-prediction
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run Model Training
```bash
python src/train_model.py
```
### 4️⃣ Evaluate Model Performance
```bash
python src/evaluate_model.py
```

## 📢 Contributing
Contributions are welcome! Feel free to fork the repository, submit pull requests, or open issues.

## 📜 License
This project is open-source under the **APACHE License 2.0**.

---

