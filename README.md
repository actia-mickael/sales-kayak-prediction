# 🛶 Sales Kayak - Prédiction Client ML

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange.svg)](https://gradio.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-yellow.svg)](https://huggingface.co/spaces/Aligator722/sales-kayak-prediction)

Dashboard de prédiction de probabilité de commande client avec intégration météo en temps réel.

![Sxreen shot API](images/Scr_shot_API.png)

![Dashboard Preview](https://img.shields.io/badge/Status-Production-brightgreen)

## 🎯 Fonctionnalités

- **Prédiction ML** : Probabilité de commande par client sur 7 jours
- **Sélection de semaine** : Prédictions pour S+0, S+1, S+2... jusqu'à S+7
- **Intégration météo** : Prévisions Open-Meteo injectées dans le modèle
- **Segmentation** : Clients classés par quartiles (Top 25%, 50%, 75%)
- **Dashboard interactif** : KPIs, graphiques Plotly, tableau Top 20

## 🧠 Architecture ML

### Approche Transaction (Client × Jour)

Le modèle prédit au niveau **transaction** (1 client × 1 jour) puis agrège:

```
P(commande semaine) = 1 - ∏(1 - P(jour_i))
```

### Features

| Catégorie | Features |
|-----------|----------|
| **RFM** | Recency, Frequency, Monetary (total, mean, std, min, max) |
| **Temporel** | AVG_ORDER_INTERVAL, ORDER_REGULARITY, RECENT_ORDERS_90D |
| **Produit/Geo** | TOP_PRODUCTLINE, COUNTRY, TERRITORY, PRODUCT_DIVERSITY |
| **Saisonnalité** | DAY_OF_YEAR_SIN/COS, MONTH_SIN/COS, IS_WEEKEND |
| **Météo** | TEMPERATURE, PRECIPITATION, LATITUDE, LONGITUDE |

### Encodage Cyclique

Les features temporelles utilisent un encodage sin/cos pour capturer la périodicité:

```python
DAY_OF_YEAR_SIN = sin(2π × day / 365)
DAY_OF_YEAR_COS = cos(2π × day / 365)
```

## 📁 Structure du Projet

```
sales_kayak_app/
├── hf_space/                    # Submodule HuggingFace Spaces
│   ├── app.py                   # Dashboard Gradio
│   ├── model_utils.py           # Pipeline ML complet
│   ├── config.py                # Configuration features
│   ├── train.py                 # Script d'entraînement
│   ├── requirements.txt         # Dépendances
│   ├── data/
│   │   └── sales_data_enriched.csv
│   └── models/
│       ├── best_model.pkl
│       ├── label_encoders.pkl
│       └── feature_cols.pkl
├── notebooks/                   # Notebooks d'analyse
│   ├── EDA_Sales.ipynb
│   └── Sales_Kayak_ML_Prediction.ipynb
└── README.md
```

## 🚀 Déploiement

### HuggingFace Spaces (Production)

🔗 **[Démo Live](https://huggingface.co/spaces/Aligator722/sales-kayak-prediction)**

```bash
cd hf_space
git add . && git commit -m "Update" && git push
```

### Local (Docker)

```bash
cd hf_space
docker-compose up -d
# Accès: http://localhost:7860
```

### Local (Python)

```bash
cd hf_space
pip install -r requirements.txt
python train.py      # Entraîner le modèle
python app.py        # Lancer le dashboard
```

## 📊 Métriques du Modèle

| Métrique | Valeur |
|----------|--------|
| Accuracy | ~88% |
| ROC-AUC | ~79% |
| Precision | ~100% |
| Recall | ~7% |

> Le modèle est conservateur : haute précision, recall faible (B2B = peu de commandes).

## 🌤️ Intégration Météo

Les prévisions météo sont récupérées via [Open-Meteo API](https://open-meteo.com/):

- Prévisions jusqu'à 16 jours
- Par pays (coordonnées des capitales)
- Features: TEMPERATURE, PRECIPITATION
- Catégorisation: Cold/Cool/Warm/Hot, Clear/Rain/Heavy_Rain

## 🛠️ Technologies

- **ML**: scikit-learn, Random Forest
- **Dashboard**: Gradio 5.x, Plotly
- **Data**: Pandas, NumPy
- **API Météo**: Open-Meteo (gratuit, sans clé)
- **Déploiement**: HuggingFace Spaces, Docker

## 📈 Évolutions Futures

- [ ] Ajout de XGBoost/LightGBM
- [ ] Feature importance interactive
- [ ] Export CSV des prédictions
- [ ] Alertes email pour clients prioritaires
- [ ] Historique des prédictions

## 👨‍💻 Auteur

**Mickael** - [ACT-IA](https://github.com/actia-mickael)

- 🔗 LinkedIn: [mickael-moisan](https://www.linkedin.com/in/mickael-moisan-42605774/)
- 📧 Email: contact@act-ia.fr

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Développé avec ❤️ et ☕ en Bretagne*
