# Kayak Travel Analysis (Version provisoire)

> Application de recommandation de destinations de voyage basée sur la météo et les hôtels.

## 🎯 Objectif

Projet réalisé dans le cadre de la formation **JEDHA Fullstack Data Science**.

L'équipe marketing de Kayak souhaite créer une application recommandant les meilleures destinations de vacances. L'objectif est de collecter et analyser des données réelles sur :
- Les conditions météorologiques par destination
- Les hôtels disponibles et leurs évaluations

## 🔄 Pipeline de données
```
Sources              Lac de données        Entrepôt
   │                      │                   │
   ├── API Météo ────────►│                   │
   │                      ├── AWS S3 ────────►├── BigQuery
   ├── Scraping Booking ─►│   (raw data)      │   (clean data)
   │                      │                   │
```

## 📦 Installation
```bash
git clone git@github.com:actia-mickael/kayak-travel-analysis.git
cd kayak-travel-analysis
pip install -r requirements.txt
```

## 🚀 Utilisation
```bash
# 1. Collecte des données météo
python src/weather_scraper.py

# 2. Scraping des hôtels Booking
python src/booking_scraper.py

# 3. Upload vers S3
python src/upload_to_s3.py

# 4. ETL vers BigQuery
python src/etl_bigquery.py
```

## 🏗️ Structure du projet
```
├── data/               # Données locales (non versionnées)
├── notebooks/          # Exploration et analyses
├── src/                # Scripts Python
│   ├── weather_scraper.py
│   ├── booking_scraper.py
│   ├── upload_to_s3.py
│   └── etl_bigquery.py
├── requirements.txt
└── README.md
```

## 🛠️ Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scrapy](https://img.shields.io/badge/Scrapy-60A839?style=flat&logo=scrapy&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS%20S3-569A31?style=flat&logo=amazons3&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-4285F4?style=flat&logo=googlebigquery&logoColor=white)

## 📊 Résultats

Top 5 des destinations recommandées selon le score combiné (météo + hôtels) :

| Rang | Destination | Score Météo | Score Hôtels |
|------|-------------|-------------|--------------|
| 1    | ...         | ...         | ...          |
| 2    | ...         | ...         | ...          |

> *A-Compléter*

## 📝 License

MIT