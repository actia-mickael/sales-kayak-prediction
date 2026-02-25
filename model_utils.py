"""
Utilitaires pour le modèle de prédiction Sales Kayak
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

from config import (
    FEATURE_COLS, CAT_COLS, N_SLIDING_WINDOWS, SLIDING_STEP, 
    MIN_HISTORY_DAYS, DATA_PATH, MODEL_PATH, SCALER_PATH, ENCODERS_PATH
)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_customer_features(df_hist, reference_date):
    """Crée les features client basées sur l'historique"""
    if len(df_hist) == 0:
        return pd.DataFrame()
    
    customer_features = df_hist.groupby('CUSTOMERNAME').agg({
        'ORDERDATE': lambda x: (reference_date - x.max()).days,
        'ORDERNUMBER': 'nunique',
        'SALES': ['sum', 'mean', 'std', 'min', 'max'],
        'QUANTITYORDERED': ['sum', 'mean'],
        'PRICEEACH': ['mean', 'std'],
    }).reset_index()
    
    customer_features.columns = [
        'CUSTOMERNAME', 'RECENCY', 'FREQUENCY',
        'MONETARY_TOTAL', 'MONETARY_MEAN', 'MONETARY_STD', 'MONETARY_MIN', 'MONETARY_MAX',
        'QUANTITY_TOTAL', 'QUANTITY_MEAN',
        'PRICE_MEAN', 'PRICE_STD'
    ]
    
    return customer_features


def add_temporal_features(df_hist, customer_df, reference_date):
    """Ajoute des features temporelles avancées"""
    if len(customer_df) == 0:
        return customer_df
    
    # Intervalle moyen entre commandes
    order_intervals = df_hist.groupby('CUSTOMERNAME')['ORDERDATE'].apply(
        lambda x: x.sort_values().diff().mean().days if len(x) > 1 else np.nan
    ).reset_index()
    order_intervals.columns = ['CUSTOMERNAME', 'AVG_ORDER_INTERVAL']
    
    # Régularité des commandes
    quarterly_orders = df_hist.groupby(['CUSTOMERNAME', 'QTR_ID']).size().unstack(fill_value=0)
    quarterly_orders['ORDER_REGULARITY'] = quarterly_orders.std(axis=1)
    quarterly_orders = quarterly_orders[['ORDER_REGULARITY']].reset_index()
    
    # Commandes récentes (90 jours)
    recent_date = reference_date - timedelta(days=90)
    recent_orders = df_hist[df_hist['ORDERDATE'] > recent_date].groupby('CUSTOMERNAME').size().reset_index()
    recent_orders.columns = ['CUSTOMERNAME', 'RECENT_ORDERS_90D']
    
    # Merge
    customer_df = customer_df.merge(order_intervals, on='CUSTOMERNAME', how='left')
    customer_df = customer_df.merge(quarterly_orders, on='CUSTOMERNAME', how='left')
    customer_df = customer_df.merge(recent_orders, on='CUSTOMERNAME', how='left')
    
    # Remplir NaN
    customer_df['AVG_ORDER_INTERVAL'] = customer_df['AVG_ORDER_INTERVAL'].fillna(999)
    customer_df['RECENT_ORDERS_90D'] = customer_df['RECENT_ORDERS_90D'].fillna(0)
    customer_df['ORDER_REGULARITY'] = customer_df['ORDER_REGULARITY'].fillna(0)
    
    return customer_df


def add_product_geo_features(df_hist, customer_df):
    """Ajoute des features produits et géographiques"""
    if len(customer_df) == 0:
        return customer_df
    
    # Ligne de produit préférée
    top_product = df_hist.groupby('CUSTOMERNAME')['PRODUCTLINE'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    top_product.columns = ['CUSTOMERNAME', 'TOP_PRODUCTLINE']
    
    # Diversité produits
    product_diversity = df_hist.groupby('CUSTOMERNAME')['PRODUCTLINE'].nunique().reset_index()
    product_diversity.columns = ['CUSTOMERNAME', 'PRODUCT_DIVERSITY']
    
    # Géographie
    customer_geo = df_hist.groupby('CUSTOMERNAME').agg({
        'COUNTRY': 'first',
        'TERRITORY': 'first'
    }).reset_index()
    
    # Deal size préféré
    top_deal = df_hist.groupby('CUSTOMERNAME')['DEALSIZE'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    top_deal.columns = ['CUSTOMERNAME', 'PREFERRED_DEALSIZE']
    
    # Merge
    customer_df = customer_df.merge(top_product, on='CUSTOMERNAME', how='left')
    customer_df = customer_df.merge(product_diversity, on='CUSTOMERNAME', how='left')
    customer_df = customer_df.merge(customer_geo, on='CUSTOMERNAME', how='left')
    customer_df = customer_df.merge(top_deal, on='CUSTOMERNAME', how='left')
    
    return customer_df


def add_derived_features(customer_df):
    """Ajoute des features dérivées"""
    if len(customer_df) == 0:
        return customer_df
    
    customer_df['MONETARY_STD'] = customer_df['MONETARY_STD'].fillna(0)
    customer_df['PRICE_STD'] = customer_df['PRICE_STD'].fillna(0)
    customer_df['VALUE_PER_ORDER'] = customer_df['MONETARY_TOTAL'] / customer_df['FREQUENCY']
    customer_df['CHURN_RISK_SCORE'] = customer_df['RECENCY'] / (customer_df['AVG_ORDER_INTERVAL'] + 1)
    customer_df['IS_ACTIVE_RECENTLY'] = (customer_df['RECENCY'] <= 30).astype(int)
    
    return customer_df


def build_dataset_for_window(df, reference_date, prediction_window):
    """Construit le dataset pour une fenêtre donnée"""
    df_history = df[df['ORDERDATE'] <= reference_date].copy()
    df_future = df[(df['ORDERDATE'] > reference_date) & 
                   (df['ORDERDATE'] <= reference_date + timedelta(days=prediction_window))].copy()
    
    if len(df_history) == 0:
        return pd.DataFrame()
    
    customer_df = create_customer_features(df_history, reference_date)
    customer_df = add_temporal_features(df_history, customer_df, reference_date)
    customer_df = add_product_geo_features(df_history, customer_df)
    customer_df = add_derived_features(customer_df)
    
    # TARGET
    customers_ordered = df_future['CUSTOMERNAME'].unique()
    customer_df['TARGET'] = customer_df['CUSTOMERNAME'].isin(customers_ordered).astype(int)
    customer_df['WINDOW_REF_DATE'] = reference_date
    
    return customer_df


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def load_data(data_path=DATA_PATH):
    """Charge et prépare les données"""
    df = pd.read_csv(data_path, encoding='latin-1')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    return df


def build_training_dataset(df, prediction_window):
    """Construit le dataset d'entraînement avec sliding windows"""
    DATE_MAX = df['ORDERDATE'].max()
    DATE_MIN = df['ORDERDATE'].min()
    
    # Générer les dates de référence
    reference_dates = []
    for i in range(N_SLIDING_WINDOWS):
        ref_date = DATE_MAX - timedelta(days=prediction_window + i * SLIDING_STEP)
        if ref_date - timedelta(days=MIN_HISTORY_DAYS) >= DATE_MIN:
            reference_dates.append(ref_date)
    
    # Construire le dataset
    all_windows_data = []
    for ref_date in reference_dates:
        window_df = build_dataset_for_window(df, ref_date, prediction_window)
        if len(window_df) > 0:
            all_windows_data.append(window_df)
    
    return pd.concat(all_windows_data, ignore_index=True), reference_dates


def encode_features(customer_df, label_encoders=None, fit=True):
    """Encode les variables catégorielles"""
    if label_encoders is None:
        label_encoders = {}
    
    for col in CAT_COLS:
        if col in customer_df.columns:
            if fit:
                le = LabelEncoder()
                customer_df[f'{col}_encoded'] = le.fit_transform(customer_df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le:
                    # Gérer les nouvelles catégories
                    customer_df[f'{col}_encoded'] = customer_df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
    
    return customer_df, label_encoders


def train_model(df, prediction_window):
    """Entraîne le modèle complet"""
    # Construire dataset
    full_dataset, reference_dates = build_training_dataset(df, prediction_window)
    
    # Encoder
    full_dataset, label_encoders = encode_features(full_dataset, fit=True)
    
    # Préparer X, y
    available_features = [f for f in FEATURE_COLS if f in full_dataset.columns]
    X = full_dataset[available_features].copy()
    y = full_dataset['TARGET']
    
    # Nettoyer
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraîner
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Sauvegarder
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    joblib.dump(available_features, 'models/feature_cols.pkl')
    
    # Métriques
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, model.predict(X_test)),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'positive_rate': y.mean()
    }
    
    return model, label_encoders, metrics


# =============================================================================
# PREDICTION PIPELINE
# =============================================================================

def load_model():
    """Charge le modèle entraîné"""
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load('models/feature_cols.pkl')
    return model, label_encoders, feature_cols


def predict_customers(df, prediction_window, model=None, label_encoders=None, feature_cols=None):
    """Prédit la probabilité de commande pour tous les clients"""
    
    # Charger modèle si nécessaire
    if model is None:
        model, label_encoders, feature_cols = load_model()
    
    # Date de référence = aujourd'hui (ou max du dataset pour démo)
    reference_date = df['ORDERDATE'].max()
    
    # Construire features pour la dernière période
    df_history = df[df['ORDERDATE'] <= reference_date].copy()
    
    customer_df = create_customer_features(df_history, reference_date)
    customer_df = add_temporal_features(df_history, customer_df, reference_date)
    customer_df = add_product_geo_features(df_history, customer_df)
    customer_df = add_derived_features(customer_df)
    
    # Encoder
    customer_df, _ = encode_features(customer_df, label_encoders, fit=False)
    
    # Préparer X
    available_features = [f for f in feature_cols if f in customer_df.columns]
    X = customer_df[available_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Prédire
    probas = model.predict_proba(X)[:, 1]
    
    # Construire résultat
    result_df = customer_df[['CUSTOMERNAME', 'COUNTRY', 'TERRITORY', 'RECENCY', 
                             'FREQUENCY', 'MONETARY_TOTAL', 'TOP_PRODUCTLINE']].copy()
    result_df['PROBA_COMMANDE'] = probas
    result_df['RANK'] = result_df['PROBA_COMMANDE'].rank(ascending=False).astype(int)
    
    # Segmentation
    def segment_probability(prob):
        if prob >= 0.7:
            return 'Très probable'
        elif prob >= 0.5:
            return 'Probable'
        elif prob >= 0.3:
            return 'Possible'
        else:
            return 'Peu probable'
    
    result_df['SEGMENT'] = result_df['PROBA_COMMANDE'].apply(segment_probability)
    result_df = result_df.sort_values('PROBA_COMMANDE', ascending=False)
    
    # Calculer CA potentiel
    result_df['CA_POTENTIEL'] = result_df['MONETARY_TOTAL'] / result_df['FREQUENCY'] * result_df['PROBA_COMMANDE']
    
    return result_df


def get_dashboard_metrics(result_df, top_n=20):
    """Calcule les métriques pour le dashboard"""
    top_clients = result_df.head(top_n)
    
    metrics = {
        'total_clients': len(result_df),
        'ca_potentiel_total': result_df['CA_POTENTIEL'].sum(),
        'ca_potentiel_top': top_clients['CA_POTENTIEL'].sum(),
        'proba_moyenne_top': top_clients['PROBA_COMMANDE'].mean(),
        'ca_historique_top': top_clients['MONETARY_TOTAL'].sum(),
        'segments': {
            'tres_probable': (result_df['SEGMENT'] == 'Très probable').sum(),
            'probable': (result_df['SEGMENT'] == 'Probable').sum(),
            'possible': (result_df['SEGMENT'] == 'Possible').sum(),
            'peu_probable': (result_df['SEGMENT'] == 'Peu probable').sum()
        }
    }
    
    return metrics


# =============================================================================
# UTILS
# =============================================================================

def get_week_dates(week_offset=0):
    """Retourne les dates de début/fin de semaine"""
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week.date(), end_of_week.date()


def format_currency(value):
    """Formate un nombre en devise"""
    return f"${value:,.0f}"


def format_percentage(value):
    """Formate un nombre en pourcentage"""
    return f"{value:.1%}"
