"""
Configuration du projet Sales Kayak Prediction
"""

# =============================================================================
# PARAMÈTRES DU MODÈLE
# =============================================================================

# Fenêtre de prédiction par défaut (en jours)
DEFAULT_PREDICTION_WINDOW = 7

# Options de fenêtre disponibles
PREDICTION_WINDOW_OPTIONS = [7, 14, 21, 30]

# Nombre de fenêtres glissantes pour l'entraînement
N_SLIDING_WINDOWS = 12

# Pas entre chaque fenêtre (en jours)
SLIDING_STEP = 30

# Historique minimum requis (en jours)
MIN_HISTORY_DAYS = 90

# =============================================================================
# FEATURES DU MODÈLE
# =============================================================================

FEATURE_COLS = [
    # RFM
    'RECENCY', 'FREQUENCY', 'MONETARY_TOTAL', 'MONETARY_MEAN', 'MONETARY_STD',
    'MONETARY_MIN', 'MONETARY_MAX',
    # Quantity & Price
    'QUANTITY_TOTAL', 'QUANTITY_MEAN', 'PRICE_MEAN', 'PRICE_STD',
    # Temporal
    'AVG_ORDER_INTERVAL', 'ORDER_REGULARITY', 'RECENT_ORDERS_90D',
    # Product/Geo encoded
    'TOP_PRODUCTLINE_encoded', 'COUNTRY_encoded', 'TERRITORY_encoded',
    'PREFERRED_DEALSIZE_encoded', 'PRODUCT_DIVERSITY',
    # Derived
    'VALUE_PER_ORDER', 'CHURN_RISK_SCORE', 'IS_ACTIVE_RECENTLY'
]

CAT_COLS = ['TOP_PRODUCTLINE', 'COUNTRY', 'TERRITORY', 'PREFERRED_DEALSIZE']

# =============================================================================
# CHEMINS
# =============================================================================

DATA_PATH = "data/sales_data_sample.csv"
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"

# =============================================================================
# DASHBOARD
# =============================================================================

TOP_N_CLIENTS = 20
SEGMENTS = {
    'high': ('Très probable', 0.7, 1.0, '#2ecc71'),
    'medium': ('Probable', 0.5, 0.7, '#3498db'),
    'low': ('Possible', 0.3, 0.5, '#f39c12'),
    'very_low': ('Peu probable', 0.0, 0.3, '#e74c3c')
}

# Couleurs du thème
COLORS = {
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'background': '#ecf0f1'
}
