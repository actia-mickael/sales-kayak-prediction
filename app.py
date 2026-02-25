"""
🛶 Sales Kayak - Dashboard de Prédiction Client
Application Gradio pour visualiser les prédictions de commandes
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

from config import (
    DEFAULT_PREDICTION_WINDOW, PREDICTION_WINDOW_OPTIONS, 
    TOP_N_CLIENTS, COLORS, DATA_PATH
)
from model_utils import (
    load_data, train_model, predict_customers, 
    get_dashboard_metrics, load_model, format_currency, format_percentage
)


# =============================================================================
# VARIABLES GLOBALES
# =============================================================================

df_global = None
model_global = None
label_encoders_global = None
feature_cols_global = None


# =============================================================================
# INITIALISATION
# =============================================================================

def initialize_app():
    """Initialise l'application avec les données et le modèle"""
    global df_global, model_global, label_encoders_global, feature_cols_global
    
    # Charger données
    df_global = load_data(DATA_PATH)
    
    # Vérifier si modèle existe, sinon entraîner
    if not os.path.exists('models/best_model.pkl'):
        print("⏳ Entraînement du modèle...")
        model_global, label_encoders_global, metrics = train_model(df_global, DEFAULT_PREDICTION_WINDOW)
        feature_cols_global = [f for f in pd.read_pickle('models/feature_cols.pkl')]
        print(f"✅ Modèle entraîné - ROC-AUC: {metrics['roc_auc']:.2%}")
    else:
        model_global, label_encoders_global, feature_cols_global = load_model()
        print("✅ Modèle chargé")


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def create_gauge_chart(value, title, max_value=None):
    """Crée un gauge chart pour les KPIs"""
    if max_value is None:
        max_value = value * 1.5
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': COLORS['primary']}},
        number={'font': {'size': 28, 'color': COLORS['primary']}, 'prefix': "$", 'valueformat': ",.0f"},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': COLORS['success']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': COLORS['primary'],
            'steps': [
                {'range': [0, max_value * 0.33], 'color': '#e8f5e9'},
                {'range': [max_value * 0.33, max_value * 0.66], 'color': '#c8e6c9'},
                {'range': [max_value * 0.66, max_value], 'color': '#a5d6a7'}
            ],
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_segments_chart(segments):
    """Crée un donut chart pour les segments"""
    labels = ['Très probable', 'Probable', 'Possible', 'Peu probable']
    values = [segments['tres_probable'], segments['probable'], 
              segments['possible'], segments['peu_probable']]
    colors = [COLORS['success'], COLORS['secondary'], COLORS['warning'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo='label+value',
        textposition='outside',
        textfont=dict(size=12)
    )])
    
    fig.update_layout(
        title=dict(text="Segmentation Clients", font=dict(size=16, color=COLORS['primary'])),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(text='Segments', x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    
    return fig


def create_top_clients_chart(result_df, top_n=20):
    """Crée un bar chart horizontal des top clients"""
    top_df = result_df.head(top_n).copy()
    top_df = top_df.iloc[::-1]  # Inverser pour affichage
    
    # Couleurs selon segment
    color_map = {
        'Très probable': COLORS['success'],
        'Probable': COLORS['secondary'],
        'Possible': COLORS['warning'],
        'Peu probable': COLORS['danger']
    }
    colors = [color_map.get(s, COLORS['secondary']) for s in top_df['SEGMENT']]
    
    fig = go.Figure(go.Bar(
        x=top_df['PROBA_COMMANDE'] * 100,
        y=top_df['CUSTOMERNAME'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in top_df['PROBA_COMMANDE'] * 100],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Probabilité: %{x:.1f}%<br><extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"Top {top_n} Clients à Contacter", font=dict(size=16, color=COLORS['primary'])),
        xaxis_title="Probabilité de commande (%)",
        yaxis_title="",
        height=600,
        margin=dict(l=200, r=50, t=60, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 105], gridcolor='#eee'),
        yaxis=dict(tickfont=dict(size=10))
    )
    
    return fig


def create_ca_distribution_chart(result_df):
    """Crée un histogramme de la distribution des probabilités"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=result_df['PROBA_COMMANDE'] * 100,
        nbinsx=20,
        marker_color=COLORS['secondary'],
        opacity=0.8,
        name='Distribution'
    ))
    
    # Ligne moyenne
    mean_proba = result_df['PROBA_COMMANDE'].mean() * 100
    fig.add_vline(x=mean_proba, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text=f"Moyenne: {mean_proba:.1f}%")
    
    fig.update_layout(
        title=dict(text="Distribution des Probabilités", font=dict(size=16, color=COLORS['primary'])),
        xaxis_title="Probabilité de commande (%)",
        yaxis_title="Nombre de clients",
        height=300,
        margin=dict(l=50, r=20, t=60, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#eee'),
        yaxis=dict(gridcolor='#eee'),
        showlegend=False
    )
    
    return fig


def create_ca_potentiel_chart(result_df):
    """Crée un bar chart du CA potentiel par segment"""
    ca_by_segment = result_df.groupby('SEGMENT')['CA_POTENTIEL'].sum().reset_index()
    
    # Ordre des segments
    segment_order = ['Très probable', 'Probable', 'Possible', 'Peu probable']
    ca_by_segment['SEGMENT'] = pd.Categorical(ca_by_segment['SEGMENT'], categories=segment_order, ordered=True)
    ca_by_segment = ca_by_segment.sort_values('SEGMENT')
    
    color_map = {
        'Très probable': COLORS['success'],
        'Probable': COLORS['secondary'],
        'Possible': COLORS['warning'],
        'Peu probable': COLORS['danger']
    }
    colors = [color_map.get(s, COLORS['secondary']) for s in ca_by_segment['SEGMENT']]
    
    fig = go.Figure(go.Bar(
        x=ca_by_segment['SEGMENT'],
        y=ca_by_segment['CA_POTENTIEL'],
        marker=dict(color=colors),
        text=[f"${v:,.0f}" for v in ca_by_segment['CA_POTENTIEL']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="CA Potentiel par Segment", font=dict(size=16, color=COLORS['primary'])),
        xaxis_title="",
        yaxis_title="CA Potentiel ($)",
        height=300,
        margin=dict(l=50, r=20, t=60, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='#eee'),
        showlegend=False
    )
    
    return fig


# =============================================================================
# FONCTIONS PRINCIPALES
# =============================================================================

def run_prediction(prediction_window):
    """Lance la prédiction et retourne les résultats"""
    global df_global, model_global, label_encoders_global, feature_cols_global
    
    # Prédiction
    result_df = predict_customers(
        df_global, 
        prediction_window, 
        model_global, 
        label_encoders_global, 
        feature_cols_global
    )
    
    # Métriques
    metrics = get_dashboard_metrics(result_df, TOP_N_CLIENTS)
    
    return result_df, metrics


def update_dashboard(prediction_window):
    """Met à jour tous les éléments du dashboard"""
    
    # Lancer prédiction
    result_df, metrics = run_prediction(int(prediction_window))
    
    # KPIs
    kpi_ca_total = f"${metrics['ca_potentiel_total']:,.0f}"
    kpi_ca_top = f"${metrics['ca_potentiel_top']:,.0f}"
    kpi_proba_top = f"{metrics['proba_moyenne_top']:.1%}"
    kpi_clients = f"{metrics['total_clients']}"
    
    # Charts
    chart_segments = create_segments_chart(metrics['segments'])
    chart_top_clients = create_top_clients_chart(result_df, TOP_N_CLIENTS)
    chart_distribution = create_ca_distribution_chart(result_df)
    chart_ca_segment = create_ca_potentiel_chart(result_df)
    
    # Table top 20
    top_20_df = result_df.head(20)[['RANK', 'CUSTOMERNAME', 'PROBA_COMMANDE', 'SEGMENT', 
                                     'COUNTRY', 'CA_POTENTIEL', 'MONETARY_TOTAL', 'RECENCY']].copy()
    top_20_df['PROBA_COMMANDE'] = (top_20_df['PROBA_COMMANDE'] * 100).round(1).astype(str) + '%'
    top_20_df['CA_POTENTIEL'] = top_20_df['CA_POTENTIEL'].apply(lambda x: f"${x:,.0f}")
    top_20_df['MONETARY_TOTAL'] = top_20_df['MONETARY_TOTAL'].apply(lambda x: f"${x:,.0f}")
    top_20_df.columns = ['Rang', 'Client', 'Probabilité', 'Segment', 'Pays', 'CA Potentiel', 'CA Historique', 'Récence (j)']
    
    return (
        kpi_ca_total, kpi_ca_top, kpi_proba_top, kpi_clients,
        chart_segments, chart_top_clients, chart_distribution, chart_ca_segment,
        top_20_df
    )


# =============================================================================
# INTERFACE GRADIO
# =============================================================================

# CSS personnalisé
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.kpi-value {
    font-size: 2.5em;
    font-weight: bold;
    margin: 10px 0;
}
.kpi-label {
    font-size: 0.9em;
    opacity: 0.9;
}
.header-title {
    background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
.section-title {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
"""

def create_dashboard():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(css=custom_css, title="🛶 Sales Kayak - Prédictions", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
                <h1 style="color: white; margin: 0; font-size: 2.5em;">🛶 Sales Kayak Dashboard</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1em;">
                    Prédiction de probabilité de commande client
                </p>
            </div>
        """)
        
        # Sélecteur de fenêtre
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("<h3 style='color: #2c3e50; margin: 0;'>⚙️ Configuration</h3>")
            with gr.Column(scale=1):
                prediction_window = gr.Dropdown(
                    choices=PREDICTION_WINDOW_OPTIONS,
                    value=DEFAULT_PREDICTION_WINDOW,
                    label="📅 Fenêtre de prédiction (jours)",
                    interactive=True
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Actualiser", variant="primary", size="lg")
        
        gr.HTML("<hr style='border: 1px solid #eee; margin: 20px 0;'>")
        
        # KPIs
        with gr.Row():
            with gr.Column():
                gr.HTML("<div style='text-align:center; color:#666; font-size:0.9em;'>💰 CA Potentiel Total</div>")
                kpi_ca_total = gr.Textbox(value="$0", label="", interactive=False, 
                                          elem_classes="kpi-value", show_label=False)
            with gr.Column():
                gr.HTML("<div style='text-align:center; color:#666; font-size:0.9em;'>🎯 CA Potentiel Top 20</div>")
                kpi_ca_top = gr.Textbox(value="$0", label="", interactive=False,
                                        elem_classes="kpi-value", show_label=False)
            with gr.Column():
                gr.HTML("<div style='text-align:center; color:#666; font-size:0.9em;'>📊 Proba Moyenne Top 20</div>")
                kpi_proba_top = gr.Textbox(value="0%", label="", interactive=False,
                                           elem_classes="kpi-value", show_label=False)
            with gr.Column():
                gr.HTML("<div style='text-align:center; color:#666; font-size:0.9em;'>👥 Total Clients</div>")
                kpi_clients = gr.Textbox(value="0", label="", interactive=False,
                                         elem_classes="kpi-value", show_label=False)
        
        gr.HTML("<hr style='border: 1px solid #eee; margin: 20px 0;'>")
        
        # Charts Row 1
        with gr.Row():
            with gr.Column(scale=1):
                chart_segments = gr.Plot(label="")
            with gr.Column(scale=2):
                chart_top_clients = gr.Plot(label="")
        
        # Charts Row 2
        with gr.Row():
            with gr.Column():
                chart_distribution = gr.Plot(label="")
            with gr.Column():
                chart_ca_segment = gr.Plot(label="")
        
        gr.HTML("<hr style='border: 1px solid #eee; margin: 20px 0;'>")
        
        # Table
        gr.HTML("<h3 style='color: #2c3e50;'>📋 Top 20 Clients à Contacter en Priorité</h3>")
        table_top_clients = gr.Dataframe(
            headers=['Rang', 'Client', 'Probabilité', 'Segment', 'Pays', 'CA Potentiel', 'CA Historique', 'Récence (j)'],
            interactive=False,
            wrap=True
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 20px; margin-top: 30px; color: #666; font-size: 0.9em;">
                <p>🛶 Sales Kayak Prediction Dashboard | Powered by Gradio & Random Forest</p>
                <p>📅 Données mises à jour automatiquement | 🔄 Modèle réentraînable</p>
            </div>
        """)
        
        # Events
        outputs = [
            kpi_ca_total, kpi_ca_top, kpi_proba_top, kpi_clients,
            chart_segments, chart_top_clients, chart_distribution, chart_ca_segment,
            table_top_clients
        ]
        
        refresh_btn.click(
            fn=update_dashboard,
            inputs=[prediction_window],
            outputs=outputs
        )
        
        prediction_window.change(
            fn=update_dashboard,
            inputs=[prediction_window],
            outputs=outputs
        )
        
        # Chargement initial
        app.load(
            fn=update_dashboard,
            inputs=[prediction_window],
            outputs=outputs
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🚀 Initialisation de l'application...")
    
    # Créer dossiers
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialiser
    initialize_app()
    
    # Lancer app
    app = create_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
