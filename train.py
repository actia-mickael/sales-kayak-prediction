#!/usr/bin/env python3
"""
🛶 Sales Kayak - Script d'entraînement du modèle
Exécuter ce script pour (ré)entraîner le modèle de prédiction
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH, DEFAULT_PREDICTION_WINDOW
from model_utils import load_data, train_model

def main():
    print("=" * 60)
    print("🛶 SALES KAYAK - ENTRAÎNEMENT DU MODÈLE")
    print("=" * 60)
    
    # Paramètres
    prediction_window = DEFAULT_PREDICTION_WINDOW
    print(f"\n⚙️ Configuration:")
    print(f"   • Fenêtre de prédiction: {prediction_window} jours")
    print(f"   • Données: {DATA_PATH}")
    
    # Charger données
    print("\n📊 Chargement des données...")
    df = load_data(DATA_PATH)
    print(f"   ✅ {len(df)} transactions chargées")
    print(f"   ✅ {df['CUSTOMERNAME'].nunique()} clients uniques")
    print(f"   ✅ Période: {df['ORDERDATE'].min().date()} → {df['ORDERDATE'].max().date()}")
    
    # Entraîner
    print("\n🧠 Entraînement du modèle...")
    model, label_encoders, metrics = train_model(df, prediction_window)
    
    # Résultats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("=" * 60)
    print(f"""
   • Accuracy: {metrics['accuracy']:.2%}
   • ROC-AUC: {metrics['roc_auc']:.2%}
   • Observations train: {metrics['n_train']}
   • Observations test: {metrics['n_test']}
   • Taux de positifs: {metrics['positive_rate']:.2%}
    """)
    
    print("✅ Modèle sauvegardé dans models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
