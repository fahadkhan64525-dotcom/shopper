# 🛍️ Shopper Spectrum

Customer Segmentation & Product Recommendations — E-Commerce Analytics

## Quick Start

```bash
pip install -r requirements.txt

# Option A: regenerate data + models from scratch
python generate_data.py
python pipeline.py

# Option B: use pre-built assets (included)

# Launch the app
streamlit run app.py
```

## Project Structure
```
shopper_spectrum/
├── app.py                         # Streamlit application
├── pipeline.py                    # Full ML pipeline
├── generate_data.py               # Synthetic data generator
├── ecommerce_data.csv             # Dataset (2022-2023)
├── shopper_spectrum_notebook.ipynb# Jupyter notebook
├── requirements.txt
└── assets/
    ├── model_bundle.pkl           # Trained models + similarity matrix
    ├── rfm_data.csv               # RFM table with segment labels
    ├── segment_stats.csv          # Per-segment statistics
    └── *.png                      # EDA & cluster visualisations
```

## Methods
- **RFM Analysis**: Recency, Frequency, Monetary features
- **Preprocessing**: log-transform (Frequency, Monetary) + StandardScaler
- **Clustering**: KMeans, k=4, selected via Elbow + Silhouette
- **Segments**: High-Value · Regular · Occasional · At-Risk
- **Recommendations**: Item-based cosine similarity on purchase matrix
