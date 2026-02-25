# 🛶 Sales Kayak - Customer Prediction Dashboard

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/sales-kayak-prediction)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)](https://gradio.app)

**Dashboard interactif de prédiction de probabilité de commande client.**

Identifiez les clients les plus susceptibles de commander dans les prochains jours et optimisez vos actions commerciales.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Sales+Kayak+Dashboard)

---

## 🎯 Fonctionnalités

- **📊 Prédiction ML** : Modèle Random Forest avec features RFM avancées
- **📅 Fenêtre paramétrable** : 7, 14, 21 ou 30 jours de prédiction
- **🏆 Top 20 clients** : Liste priorisée des clients à contacter
- **💰 CA potentiel** : Estimation du chiffre d'affaires potentiel par segment
- **📈 Visualisations** : Charts interactifs avec Plotly
- **🔄 Mise à jour temps réel** : Actualisation instantanée des prédictions

---

## 🚀 Démarrage rapide

### Option 1 : Docker (Recommandé)

```bash
# Cloner le repo
git clone https://github.com/YOUR_USERNAME/sales-kayak-prediction.git
cd sales-kayak-prediction

# Lancer avec Docker Compose
docker-compose up -d

# Accéder au dashboard
open http://localhost:7860
```

### Option 2 : Installation locale

```bash
# Cloner le repo
git clone https://github.com/YOUR_USERNAME/sales-kayak-prediction.git
cd sales-kayak-prediction

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

### Option 3 : Hugging Face Spaces

🔗 [Accéder au dashboard en ligne](https://huggingface.co/spaces/YOUR_USERNAME/sales-kayak-prediction)

---

## 📁 Structure du projet

```
sales-kayak-prediction/
├── app.py                 # Application Gradio principale
├── config.py              # Configuration et paramètres
├── model_utils.py         # Fonctions ML et feature engineering
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Image Docker
├── docker-compose.yml    # Orchestration Docker
├── .dockerignore         # Fichiers exclus du build
├── data/
│   └── sales_data_sample.csv  # Dataset de ventes
├── models/
│   ├── best_model.pkl         # Modèle entraîné
│   ├── label_encoders.pkl     # Encodeurs
│   └── feature_cols.pkl       # Liste des features
└── README.md
```

---

## ⚙️ Configuration

Les paramètres sont modifiables dans `config.py` :

```python
# Fenêtre de prédiction par défaut
DEFAULT_PREDICTION_WINDOW = 7  # jours

# Options disponibles
PREDICTION_WINDOW_OPTIONS = [7, 14, 21, 30]

# Nombre de fenêtres glissantes (entraînement)
N_SLIDING_WINDOWS = 12

# Top clients à afficher
TOP_N_CLIENTS = 20
```

---

## 🧠 Modèle ML

### Features utilisées

| Catégorie | Features |
|-----------|----------|
| **RFM** | Recency, Frequency, Monetary (total, mean, std, min, max) |
| **Temporel** | Intervalle moyen, régularité, commandes récentes |
| **Produit** | Ligne préférée, diversité, deal size |
| **Géographie** | Pays, territoire |
| **Dérivées** | Valeur par commande, score churn, activité récente |

### Approche Sliding Window

Pour pallier le déséquilibre des classes, on utilise des fenêtres temporelles glissantes :

```
Fenêtre 1: [------historique------][prédiction]
Fenêtre 2:    [------historique------][prédiction]
Fenêtre 3:       [------historique------][prédiction]
...
```

Cela multiplie les observations d'entraînement et améliore la robustesse du modèle.

---

## 📊 Screenshots

### KPIs et Segmentation
![KPIs](https://via.placeholder.com/400x200?text=KPIs)

### Top Clients
![Top Clients](https://via.placeholder.com/400x300?text=Top+Clients+Chart)

### Distribution
![Distribution](https://via.placeholder.com/400x200?text=Probability+Distribution)

---

## 🐳 Docker

### Build manuel

```bash
docker build -t sales-kayak-dashboard .
docker run -p 7860:7860 sales-kayak-dashboard
```

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `GRADIO_SERVER_NAME` | Adresse du serveur | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Port du serveur | `7860` |

---

## 🤗 Déploiement Hugging Face

1. Créer un nouveau Space sur [Hugging Face](https://huggingface.co/new-space)
2. Sélectionner **Gradio** comme SDK
3. Cloner et pousser le code :

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sales-kayak-prediction
cp -r * sales-kayak-prediction/
cd sales-kayak-prediction
git add .
git commit -m "Initial commit"
git push
```

---

## 📝 Licence

MIT License - voir [LICENSE](LICENSE)

---

## 👨‍💻 Auteur

**Mickael** - [ACT-IA](https://github.com/YOUR_USERNAME)

- 🔗 LinkedIn: [Votre profil]
- 📧 Email: contact@act-ia.fr

---

## 🙏 Remerciements

- Dataset basé sur [Sample Sales Data](https://www.kaggle.com/datasets)
- Dashboard propulsé par [Gradio](https://gradio.app)
- Visualisations avec [Plotly](https://plotly.com)
