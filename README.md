# Transformer-Based Limit Order Book Forecasting for Bitcoin Price Direction Prediction

## Overview
Deep learning model using Transformer architecture to predict Bitcoin price direction (UP/DOWN/STATIONARY) from limit order book microstructure data. Achieved 54.7% directional accuracy on validation set, demonstrating the model's ability to learn tradable patterns from order book dynamics.

## Key Results
- **Validation Performance:** 54.7% directional accuracy (trade win rate)
- **Test Performance:** 42.6% directional accuracy (temporal distribution shift observed)
- **Dataset:** 31 days of Bitcoin tick-level order book data (1.8M ticks → 44K 1-minute bars)
- **Features:** 53 engineered microstructure features
- **Model Size:** ~700K parameters

## Methodology

### Data Processing
1. Collected 31 days of Bitcoin LOB data (10 levels of depth, Nov 27 - Dec 27, 2024)
2. Sampled 5% of ticks to reduce noise
3. Aggregated to 1-minute bars using OHLC methodology
4. Engineered 13 microstructure features:
   - Order book imbalance at 5 levels
   - Spread in basis points
   - Price momentum (1, 5, 10 periods)
   - Volume-weighted metrics
   - Rolling volatility (10, 30 periods)

### Model Architecture
- **Type:** Transformer Encoder
- **Layers:** 3 encoder layers
- **Attention Heads:** 8
- **Hidden Dimension:** 128
- **Feedforward Dimension:** 512
- **Dropout:** 0.1
- **Input:** 100-timestep windows (53 features each)
- **Output:** 3-class classification (DOWN/STATIONARY/UP)

### Training Strategy
- **Loss Function:** Weighted Cross-Entropy (handles class imbalance)
- **Optimizer:** AdamW (lr=5e-5, weight_decay=1e-4)
- **Gradient Clipping:** Max norm = 1.0
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Early Stopping:** Patience = 7 epochs
- **Optimization Target:** Directional accuracy (not overall accuracy)

### Evaluation
- **Split:** Temporal 70/15/15 (train/val/test by date)
- **Primary Metric:** Directional accuracy (trade win rate when actually trading)
- **Secondary Metrics:** UP/DOWN precision, trade frequency, coverage

## Technical Stack
- **Framework:** PyTorch 2.0
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab with GPU (NVIDIA T4)

## Key Findings

### Strengths
- Model successfully learns order book patterns (54.7% validation vs 50% random)
- UP/DOWN precision balanced (52.6% UP, 55.5% DOWN on validation)
- Reasonable trade frequency (42% of days)

### Challenges
- **Temporal distribution shift:** Test accuracy dropped to 42.6%
- Late December market regime differed from training period
- Model increased trade frequency to 78% on test set (overfitting indicator)

### Learnings
- Importance of temporal cross-validation in financial ML
- Bitcoin high-frequency data is inherently noisy and non-stationary
- Model calibration is critical for trading systems
- Overfitting is a major challenge even with regularization

## Project Structure
```
Project-1-LOB-Forecasting/
├── notebook/
│   └── LOB_Forecasting.ipynb          # Complete implementation
├── results/
│   ├── training_history.png           # Training curves
│   ├── confusion_matrix.png           # Test set confusion matrix
│   └── confidence_analysis.png        # Model calibration analysis
└── README.md
```

## How to Run
1. Open `notebook/LOB_Forecasting.ipynb` in Google Colab
2. Mount Google Drive (data stored in Drive)
3. Run cells sequentially:
   - Cell 1: Data loading
   - Cell 2: Preprocessing & feature engineering
   - Cell 3: Model architecture & setup
   - Cell 4: Training loop (15-22 epochs, ~20 minutes on GPU)
   - Cell 5: Test evaluation

## Future Improvements
- Implement k-fold temporal cross-validation
- Test on stock market data (less volatile than crypto)
- Explore ensemble methods (combine multiple models)
- Add feature importance analysis
- Implement walk-forward validation
- Try alternative architectures (CNN-LSTM, GRU)

## References
- Attention Is All You Need (Vaswani et al., 2017)
- DeepLOB: Deep Convolutional Neural Networks for Limit Order Books (Zhang et al., 2019)
- FI-2010: A Benchmark Dataset for Financial Market Prediction

## Author
[Your Name]  
[LinkedIn](linkedin.com/in/dasanagh) | [Email](ddas948@gmail.com)

---

*I created this project to demonstrate machine learning applications in quantitative finance.*
