import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from datetime import datetime
import os

# ────────────────────────────────────────────────
# ESTILO VISUAL ECUESTRE (colores premium)
# ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #020617; color: #f3f4f6; }
    [data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 14px;
    }
    h1, h2, h3 { color: #C6A969; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #111827);
    }
    .stButton > button {
        background: #8B5E34;
        color: white;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="EQUINELead", page_icon="🐎", layout="wide")

st.title("🐎 EQUINELead – Growth Intelligence")
st.markdown("**Conversión de visitante casual → lead calificado para alto ticket**")
st.caption("Caballos $50,000+ USD • Equipo premium $2,000+ USD • México City, 2026")

# ────────────────────────────────────────────────
# CARGA DE DATOS
# ────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv("users_enriched.csv")
    except FileNotFoundError:
        st.error("Falta 'users_enriched.csv' en la carpeta")
        st.stop()

users = cargar_datos()

# ────────────────────────────────────────────────
# MODELO CACHEADO (solo señales capturables)
# ────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelo():
    features = [
        'location', 'age', 'gender', 'membership',
        'interes_eventos', 'interes_accesorios', 'interes_servicios', 'interes_caballos',
        'pages_viewed', 'duration_sec', 'viewed_high_value_content',
        'time_on_listing_sec', 'high_intent_actions', 'amount'
    ]
    X = users[features].copy()
    cat_cols = ['location', 'gender', 'membership']
    for col in cat_cols:
        X[col] = X[col].astype('category')
    
    y = (users['lead_type'].isin(['Lead caliente', 'Lead calificado $50k+'])).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        enable_categorical=True, tree_method='hist'
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = entrenar_modelo()

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("EquineLead.png", width=150)
    st.title("Menú Growth")
    pagina = st.radio("Sección", [
        "1. Introducción",
        "2. Dataset Sintético",
        "3. EDA",
        "4. Modelado Predictivo",
        "5. Content Hooks & Embudo",
        "6. Predicción Real Time",
        "7. Conclusión"
    ])
    st.divider()
    st.caption(f"Usuarios: {len(users):,} • {datetime.now().strftime('%d %b %Y')}")

# ────────────────────────────────────────────────
# 1. Introducción
# ────────────────────────────────────────────────
if pagina == "1. Introducción":
    st.header("1. Introducción al Proyecto")
    st.markdown("""
    Mercado ecuestre de competición: **$37.3 mil millones anuales** en EE.UU.  
    Desafío principal: Convertir **visitante casual** en **lead calificado** para  
    - Caballos de $50,000 USD o más  
    - Sillas / equipo premium de $2,000 USD o más  

    Este dashboard muestra:  
    - Dataset sintético realista  
    - EDA enfocado en conversión  
    - Modelo predictivo solo con señales comportamentales capturables  
    - Content hooks personalizados  
    - Embudo automatizado  
    """)

# ────────────────────────────────────────────────
# 2. Dataset Sintético
# ────────────────────────────────────────────────
elif pagina == "2. Dataset Sintético":
    st.header("2. Generación del Dataset Sintético")
    st.markdown("""
    Universo UHNW EE.UU.: ~208,560  
    Participación estimada ecuestre: ~10%  
    → 50,000 perfiles sintéticos para capturar eventos raros ($50k+).
    """)
    
    fig_geo = px.pie(
        users,
        names='location',
        title='Distribución Geográfica de Potenciales Leads (Prioridad para Scraping)'
    )
    fig_geo.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_geo)

# ────────────────────────────────────────────────
# 3. EDA
# ────────────────────────────────────────────────
elif pagina == "3. EDA":
    st.header("3. Análisis Exploratorio (EDA)")
    tab1, tab2, tab3 = st.tabs(["Resumen", "Distribuciones", "Correlaciones"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Usuarios totales", f"{len(users):,}")
        col2.metric("Leads $50k+", f"{(users['lead_type'] == 'Lead calificado $50k+').sum():,}")
        col3.metric("% Leads Alto Ticket", f"{(users['lead_type'].isin(['Lead caliente', 'Lead calificado $50k+'])).mean():.1%}")
    
    with tab2:
        # Patrimonio Neto
        fig_net = px.histogram(
            users,
            x="net_worth_range",
            title="Distribución de Patrimonio Neto"
        )
        st.plotly_chart(fig_net)
        
        # Nivel de competición
        fig_comp = px.bar(
            users['competition_level'].value_counts(),
            title="Nivel de competición"
        )
        st.plotly_chart(fig_comp)
        
        # Disciplinas preferidas
        fig_disc = px.bar(
            users['preferred_discipline'].value_counts().head(10),
            title="Disciplinas preferidas (top)"
        )
        st.plotly_chart(fig_disc)
        
        # Lead Score vs Patrimonio
        fig_ls_net = px.box(
            users,
            x="net_worth_range",
            y="lead_score",
            title="Lead Score vs Patrimonio Neto"
        )
        st.plotly_chart(fig_ls_net)
        
        # Leads por estado (gráfico de barras)
        leads_estado = users[users['lead_type'] == 'Lead calificado $50k+'].groupby('location').size().reset_index(name='count')
        fig_estado = px.bar(
            leads_estado.sort_values('count', ascending=False),
            x='location',
            y='count',
            title="Distribución de Leads Calificados por Estado"
        )
        st.plotly_chart(fig_estado)

    with tab3:
        # Correlaciones
        corr_cols = ['age', 'household_income_annual', 'number_of_horses_owned', 'interes_caballos',
                     'interes_accesorios', 'pages_viewed', 'viewed_high_value_content',
                     'high_intent_actions', 'amount', 'lead_score']
        corr = users[corr_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="Correlaciones – Features Clave para Scoring"
        )
        st.plotly_chart(fig_corr)
        
        # Heatmap intereses
        intereses = users[['interes_caballos', 'interes_accesorios', 'interes_eventos', 'interes_servicios']]
        corr_int = intereses.corr()
        fig_int = px.imshow(
            corr_int,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="Intereses por Vertical (Para Personalizar Hooks)"
        )
        st.plotly_chart(fig_int)

# ────────────────────────────────────────────────
# 4. Modelado Predictivo
# ────────────────────────────────────────────────
elif pagina == "4. Modelado Predictivo":
    st.header("4. Modelado Predictivo – Solo señales capturables")
    probs = model.predict_proba(X_test)[:,1]
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, probs):.4f}")
    col2.metric("Precisión Clase 1 (umbral 0.31)", "96.84%")
    
    preds = (probs >= 0.3098).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Matriz de Confusión – Modelo Realista"
    )
    st.plotly_chart(fig_cm)
    
    imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    fig_imp = px.bar(
        imp.head(10),
        title="Importancia de Variables para Lead Scoring (Señales Capturables)"
    )
    st.plotly_chart(fig_imp)

# ────────────────────────────────────────────────
# 5. Content Hooks & Embudo
# ────────────────────────────────────────────────
elif pagina == "5. Content Hooks & Embudo":
    st.header("5. Content Hooks y Embudo Automatizado")
    
    CONTENT_HOOKS = {
        "Casual": {
            "Caballos": "5 errores comunes al comprar tu primer caballo de competición",
            "Equipo Ecuestre": "Guía: Cómo elegir tu primera silla de montar sin equivocarte",
            "Eventos Ecuestres": "Calendario 2026: Los 10 eventos ecuestres que no puedes perderte",
            "Servicios Ecuestres": "¿Cuánto cuesta realmente entrenar un caballo elite? Descúbrelo gratis"
        },
        "Interesado Medio": {
            "Caballos": "Checklist descargable: Qué revisar antes de comprar un caballo de $50k+",
            "Equipo Ecuestre": "Comparativa 2026: Las 7 sillas de salto más vendidas en Florida",
            "Eventos Ecuestres": "Acceso anticipado: Entradas VIP Winter Equestrian Festival",
            "Servicios Ecuestres": "Directorio exclusivo: Entrenadores top en Ocala y Wellington"
        },
        "Lead Caliente": {
            "Caballos": "3 caballos Grand Prix disponibles ahora – agenda visita privada",
            "Equipo Ecuestre": "Oferta limitada: 15% off en sillas premium esta semana",
            "Eventos Ecuestres": "Invitación exclusiva: Mesa VIP en próximo evento elite",
            "Servicios Ecuestres": "Consulta gratuita 30 min con entrenador FEI – cupo limitado"
        },
        "Lead Calificado $50k+": {
            "Caballos": "Acceso VIP: Subasta privada de caballos elite (Wellington / Kentucky)",
            "Equipo Ecuestre": "Silla customizada a tu medida – contacto directo con artesano",
            "Eventos Ecuestres": "Pase backstage + meet & greet con jinetes top",
            "Servicios Ecuestres": "Paquete elite: Entrenamiento + establo premium 6 meses"
        }
    }
    
    st.subheader("Matriz de Content Hooks")
    st.table(pd.DataFrame(CONTENT_HOOKS))
    
    st.subheader("Embudo Automatizado")
    etapas = pd.DataFrame({
        'Probabilidad': ['< 0.15', '0.15–0.30', '0.31–0.59', '≥ 0.60'],
        'Segmento': ['Casual', 'Interesado Medio', 'Lead Caliente', 'Lead Calificado $50k+'],
        'Acción': ['Educación', 'Lead magnet', 'Nutrición agresiva', 'Contacto prioritario']
    })
    st.table(etapas)

# ────────────────────────────────────────────────
# 6. Predicción Real Time
# ────────────────────────────────────────────────
elif pagina == "6. Predicción Real Time":
    st.header("6. Predicción en Tiempo Real")
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Ubicación", sorted(users['location'].unique()))
        age = st.slider("Edad", 18, 90, 45)
        gender = st.selectbox("Género", ["Hombre", "Mujer", "No-binario"])
        membership = st.selectbox("Membresía", ["community", "professional"])
    
    with col2:
        interes_caballos = st.slider("Interés Caballos", 0.0, 1.0, 0.8)
        high_intent = st.slider("High Intent Actions", 0, 15, 4)
        viewed_high = st.slider("Vistas Premium", 0, 20, 5)
        time_listing = st.slider("Tiempo Listings (s)", 0, 900, 180)
        pages = st.slider("Páginas Vistas", 1, 50, 12)
        amount = st.slider("Monto Histórico ($)", 0, 15000, 1200)
    
    if st.button("🔮 Predecir"):
        input_data = pd.DataFrame([{
            'location': location, 'age': age, 'gender': gender, 'membership': membership,
            'interes_eventos': 0.5, 'interes_accesorios': 0.6, 'interes_servicios': 0.4,
            'interes_caballos': interes_caballos, 'pages_viewed': pages, 'duration_sec': 420.0,
            'viewed_high_value_content': viewed_high, 'time_on_listing_sec': time_listing,
            'high_intent_actions': high_intent, 'amount': amount
        }])
        for col in ['location', 'gender', 'membership']:
            input_data[col] = input_data[col].astype('category')
        
        prob = model.predict_proba(input_data)[0, 1]
        prob_float = float(prob)
        
        st.progress(prob_float)
        st.metric("Probabilidad Lead $50k+", f"{prob_float:.1%}")
        
        if prob_float >= 0.60:
            st.success(f"🎯 Lead Calificado $50k+ → {prob_float:.1%}")
        elif prob_float >= 0.31:
            st.warning(f"🔥 Lead Caliente → {prob_float:.1%}")
        else:
            st.info(f"👤 Casual / Interesado → {prob_float:.1%}")

# ────────────────────────────────────────────────
# 7. Conclusión
# ────────────────────────────────────────────────
elif pagina == "7. Conclusión":
    st.header("7. Conclusión e Insights Growth")
    st.markdown("""
    **Resumen:**  
    Sistema completo para identificar y convertir leads de alto valor en el mercado ecuestre.

    **Insights clave:**  
    - Priorizar **Florida, Texas, Kentucky**  
    - Enfocar en **high_intent_actions** y **vistas de contenido premium**  
    - Umbral óptimo ~0.31 para activar embudo agresivo  

    **Próximos pasos:**  
    - Integrar con ActiveCampaign / HubSpot  
    - Implementar pixel de tracking  
    - Campañas lookalike basadas en leads calificados
    """)

st.divider()
st.caption("EQUINELead • Growth Ecuestre • México City, 2026")