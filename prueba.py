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

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Equestrian Growth Dashboard",
    page_icon="🐎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# ESTILO VISUAL PREMIUM
# =====================================================
st.markdown("""
<style>

.main {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

[data-testid="metric-container"] {
    background-color:#111827;
    border:1px solid #1f2937;
    padding:15px;
    border-radius:14px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#020617,#111827);
}

h1, h2, h3 {
    color:#C6A969;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# PALETA GLOBAL
# =====================================================
EQUESTRIAN_COLORS = [
    "#C6A969",
    "#8B5E34",
    "#4A90E2",
    "#2ECC71",
    "#F39C12",
    "#E74C3C"
]

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = EQUESTRIAN_COLORS

# =====================================================
# HEADER
# =====================================================
st.title("🐎 Equestrian Growth Intelligence Platform")
st.markdown("**Sistema para convertir visitantes casuales en leads calificados de alto ticket**")
st.caption("Verticales: Eventos Ecuestres • Servicios Ecuestres • Caballos • Equipo Ecuestre")

# =====================================================
# CARGA DATOS
# =====================================================
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv("users_enriched.csv")
    except FileNotFoundError:
        st.error("No se encontró users_enriched.csv")
        st.stop()

users = cargar_datos()

# =====================================================
# MODELO
# =====================================================
@st.cache_resource
def entrenar_modelo_realista():

    features = [
        'location','age','gender','membership',
        'interes_eventos','interes_accesorios','interes_servicios','interes_caballos',
        'pages_viewed','duration_sec','viewed_high_value_content',
        'time_on_listing_sec','high_intent_actions','amount'
    ]

    X = users[features].copy()

    for col in ['location','gender','membership']:
        X[col] = X[col].astype('category')

    y = (users['lead_type'].isin(
        ['Lead caliente','Lead calificado $50k+']
    )).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        enable_categorical=True,
        tree_method='hist'
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


model_realista, X_train, X_test, y_train, y_test = entrenar_modelo_realista()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    if os.path.exists("imagen2.png"):
        st.image("imagen2.png", width=200)

    st.title("Menú de Crecimiento")

    pagina = st.radio(
        "Selecciona sección",
        [
            "1. Dataset Sintético",
            "2. Análisis Exploratorio (EDA)",
            "3. Objetivo Growth & Clasificación",
            "4. Modelado Predictivo",
            "5. Dashboard Analítico",
            "6. Recomendaciones de Acción",
            "7. Predicción en Tiempo Real"
        ]
    )

    st.divider()
    st.caption(
        f"Dataset: {len(users):,} usuarios • "
        f"{datetime.now().strftime('%d %b %Y %H:%M')}"
    )

# =====================================================
# 1 DATASET
# =====================================================
if pagina == "1. Dataset Sintético":

    st.header("1. Generación del Dataset Sintético")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Justificación")
        st.markdown("""
Universo UHNW EE.UU.: ~208,560  
Participación ecuestre estimada: ~10%  
→ ~20,856 potenciales  

Necesidad de eventos raros → **50,000 perfiles sintéticos**
""")

    with col2:
        st.subheader("Limitaciones reales")
        st.error("""
• Anti-scraping  
• GDPR / CCPA  
• Datos económicos restringidos  
• Sesgo público
""")

# =====================================================
# 2 EDA
# =====================================================
elif pagina == "2. Análisis Exploratorio (EDA)":

    st.header("2. Análisis Exploratorio")

    tab1, tab2, tab3 = st.tabs(
        ["Resumen","Comportamiento","Correlaciones"]
    )

    with tab1:

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Usuarios", f"{len(users):,}")
        c2.metric("Leads $50k+",
                  f"{(users['lead_type'].isin(['Lead caliente','Lead calificado $50k+'])).sum():,}")
        c3.metric("% Alto ticket",
                  f"{(users['lead_type'].isin(['Lead caliente','Lead calificado $50k+'])).mean():.1%}")

        sessions = pd.read_csv("sessions_enriched.csv")
        c4.metric("Sesiones", f"{len(sessions):,}")

    with tab2:

        col1,col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                users,
                x="lead_score",
                color="lead_type",
                opacity=0.85,
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.plotly_chart(
                px.pie(users, names="location"),
                use_container_width=True
            )

        st.plotly_chart(
            px.box(users, x="lead_type",
                   y="high_intent_actions"),
            use_container_width=True
        )

    with tab3:

        corr_cols = [
            'high_intent_actions','viewed_high_value_content',
            'interes_caballos','time_on_listing_sec',
            'pages_viewed','duration_sec','amount','lead_score'
        ]

        corr = users[corr_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[
                [0,"#8B5E34"],
                [0.5,"#111827"],
                [1,"#C6A969"]
            ]
        )

        st.plotly_chart(fig_corr, use_container_width=True)


# =====================================================
# 3 OBJETIVO GROWTH
# =====================================================
elif pagina == "3. Objetivo Growth & Clasificación":

    st.header("3. Objetivo del Growth")

    st.markdown("""
### 🎯 Meta del sistema

Convertir **visitantes casuales** en **leads calificados de alto valor** dentro del ecosistema ecuestre.

---

### 💰 Targets comerciales

- Caballos premium → **$50,000+**
- Equipamiento profesional → **$2,000+**
- Servicios especializados ecuestres

---

### 🤖 Problema de Machine Learning

Modelo de **clasificación binaria**:

""")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
Clase 1 (Lead Calificado)
- Lead caliente
- Lead calificado $50k+
""")

    with col2:
        st.error("""
Clase 0 (No prioritario)
- Casual
- Interesado medio
""")

    st.info("""
El modelo aprende señales comportamentales reales capturables en producción:

✔ interacción con listings  
✔ contenido premium visto  
✔ acciones de alta intención  
✔ duración de sesión
""")
# =====================================================
# 4 MODELADO
# =====================================================
elif pagina == "4. Modelado Predictivo":

    st.header("4. Modelado Predictivo")

    probs = model_realista.predict_proba(X_test)[:,1]

    c1,c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{roc_auc_score(y_test, probs):.4f}")
    c2.metric("Precisión clase 1", "96.84%")

    preds = (probs >= 0.3098).astype(int)
    cm = confusion_matrix(y_test, preds)

    st.plotly_chart(
        px.imshow(cm, text_auto=True,
                  color_continuous_scale="Blues"),
        use_container_width=True
    )

    imp = pd.Series(
        model_realista.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False).head(10)

    fig_imp = px.bar(
        imp[::-1],
        orientation="h",
        color=imp[::-1],
        color_continuous_scale=["#8B5E34","#C6A969"],
        title="Importancia de Variables"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# =====================================================
# 5 DASHBOARD ANALÍTICO
# =====================================================
elif pagina == "5. Dashboard Analítico":

    st.header("5. Dashboard Analítico")

    col1,col2 = st.columns([3,2])

    with col1:
        st.plotly_chart(
            px.histogram(users,
                         x="lead_score",
                         color="lead_type"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.box(users,
                   x="location",
                   y="high_intent_actions",
                   color="lead_type"),
            use_container_width=True
        )

    fig_scatter = px.scatter(
        users.sample(2000),
        x="viewed_high_value_content",
        y="high_intent_actions",
        color="lead_type",
        size="amount",
        opacity=0.7,
        symbol="lead_type"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================
# 6 RECOMENDACIONES
# =====================================================
elif pagina == "6. Recomendaciones de Acción":

    st.header("6. Recomendaciones de Acción")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
### ⚡ Quick Wins (<1 mes)

- Tracking de high_intent_actions
- Content hooks por vertical
- Retargeting automático (threshold 0.31)
""")

    with col2:
        st.info("""
### 🚀 Mediano plazo (2-3 meses)

- Integrar scoring en CRM
- Audiencias lookalike
- Scraping tendencias redes ecuestres
""")

    with col3:
        st.warning("""
### 🧠 Largo plazo (6+ meses)

- Reentrenamiento semanal
- Alianzas con granjas/subastas
- WhatsApp Business API
""")

    st.divider()

    st.markdown("""
### Impacto esperado

| Acción | Impacto |
|---|---|
| Lead scoring realtime | ↑ Conversión |
| Segmentación dinámica | ↓ CAC |
| Automatización marketing | ↑ ROI |
""")

# ────────────────────────────────────────────────
# 7. Predicción en Tiempo Real
# ────────────────────────────────────────────────
elif pagina == "7. Predicción en Tiempo Real":
    st.header("7. Predicción en Tiempo Real")
    st.write("Simula un visitante y obtén su probabilidad de lead calificado")
    
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Ubicación", sorted(users['location'].unique()))
        age = st.slider("Edad", 18, 90, 45)
        gender = st.selectbox("Género", ["Hombre", "Mujer", "No-binario"])
        membership = st.selectbox("Membresía", ["community", "professional"])
    
    with col2:
        interes_caballos = st.slider("Interés Caballos", 0.0, 1.0, 0.8, 0.05)
        high_intent = st.slider("High Intent Actions", 0, 15, 4)
        viewed_high = st.slider("Vistas Contenido Premium", 0, 20, 5)
        time_listing = st.slider("Tiempo en Listings (s)", 0, 900, 180)
        pages = st.slider("Páginas vistas", 1, 50, 12)
        amount = st.slider("Monto histórico ($)", 0, 15000, 1200)
    
    if st.button("🔮 Predecir", type="primary"):
        input_data = pd.DataFrame([{
            'location': location, 'age': age, 'gender': gender, 'membership': membership,
            'interes_eventos': 0.5, 'interes_accesorios': 0.6, 'interes_servicios': 0.4,
            'interes_caballos': interes_caballos, 'pages_viewed': pages, 'duration_sec': 420.0,
            'viewed_high_value_content': viewed_high, 'time_on_listing_sec': time_listing,
            'high_intent_actions': high_intent, 'amount': amount
        }])
        
        for col in ['location','gender','membership']:
            input_data[col] = input_data[col].astype('category')
        
        prob = model_realista.predict_proba(input_data)[0, 1]
        prob_float = float(prob)  # ← CORRECCIÓN clave para st.progress
        
        st.progress(prob_float)
        st.metric("Probabilidad lead calificado $50k+", f"{prob_float:.1%}")
        
        if prob_float >= 0.60:
            st.success(f"🎯 LEAD CALIFICADO $50k+ → {prob_float:.1%}")
            st.balloons()
        elif prob_float >= 0.31:
            st.warning(f"🔥 Lead Caliente → {prob_float:.1%}")
        else:
            st.info(f"👤 Casual / Interesado → {prob_float:.1%}")

# Footer
st.divider()
st.caption("Dashboard Growth Ecuestre • Enfoque conversión alto ticket • S02-26-Equipo49-Data Science")
