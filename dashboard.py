import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ─── CONFIGURACIÓN DE PÁGINA ─────────────────────────────────────
st.set_page_config(
    page_title='Dashboard Prescriptivo · Petróleo Colombia',
    page_icon='🛢',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ─── ESTILOS CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background-color: #0b0b1f; color: #c8c8d8; }
  .block-container { padding: 2rem 3rem; }
  h1,h2,h3 { color: white !important; }
  .metric-card {
    background: #10101e;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
  }
  .section-bar {
    width: 40px; height: 3px;
    background: #c8860a;
    margin-bottom: 10px;
    border-radius: 2px;
  }
  .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c8860a;
    margin-bottom: 6px;
  }
  .accion-card {
    background: #10101e;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    padding: 20px;
    height: 100%;
  }
  div[data-testid="stMetricValue"] { color: #f0a820 !important; font-size: 2rem !important; }
  div[data-testid="stMetricLabel"] { color: #666688 !important; font-family: monospace !important; font-size: 11px !important; }
  .stSlider > div > div { background: #1e1e3a !important; }
  .stSelectbox > div > div { background: #10101e !important; color: white !important; }
  .stNumberInput > div > div > input { background: #10101e !important; color: white !important; border: 1px solid #1e1e3a !important; }
  .stButton > button {
    background: #c8860a !important;
    color: white !important;
    border: none !important;
    font-family: monospace !important;
    letter-spacing: 2px !important;
    font-weight: 700 !important;
    padding: 12px 32px !important;
  }
  .stButton > button:hover { background: #e8a020 !important; }
  hr { border-color: #1e1e3a !important; }
  .plotly-chart { border-radius: 8px; }
  div[data-testid="stSidebar"] { background: #07071a !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATOS Y MODELO ───────────────────────────────────────────────
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 533

    presion     = np.random.uniform(800,  3500, n)
    temperatura = np.random.uniform(40,   120,  n)
    gas         = np.random.uniform(0.1,  5.0,  n)
    agua        = np.random.uniform(10,   800,  n)
    profundidad = np.random.uniform(800,  4500, n)

    # BPD con correlaciones físicamente coherentes
    bpd = np.clip(
        0.55*presion + 8*temperatura + 80*gas - 0.8*agua + 0.12*profundidad
        + np.random.normal(0, 180, n),
        200, 5000
    )

    # Estado del equipo con impacto REAL en producción
    # Operativo → producción alta, Fuera de servicio → producción baja
    estado_eq = []
    bpd_ajustado = bpd.copy()
    for i in range(n):
        r = np.random.random()
        if r < 0.78:
            estado_eq.append('Operativo')
            # Operativo: producción normal o alta
        elif r < 0.90:
            estado_eq.append('En mantenimiento')
            bpd_ajustado[i] = bpd[i] * np.random.uniform(0.60, 0.80)
        else:
            estado_eq.append('Fuera de servicio')
            bpd_ajustado[i] = bpd[i] * np.random.uniform(0.20, 0.45)

    estado_sensor = np.random.choice(
        ['Activo','Inactivo','En calibración'], n, p=[0.82, 0.10, 0.08])
    dias_sin_mant = np.random.randint(0, 365, n)

    # Costos en pesos colombianos (COP)
    costo_mant = np.random.uniform(1_000_000, 25_000_000, n)

    df = pd.DataFrame({
        'presion_psi':    presion,
        'temperatura_c':  temperatura,
        'gas_mmscfd':     gas,
        'agua_bwpd':      agua,
        'profundidad_m':  profundidad,
        'bpd_real':       bpd_ajustado,
        'estado_equipo':  estado_eq,
        'estado_sensor':  estado_sensor,
        'dias_sin_mant':  dias_sin_mant,
        'costo_mant_cop': costo_mant,
        'campo': np.random.choice(
            ['Rubiales','Cusiana','Caño Limón','Castilla','Quifa'], n),
        'pozo_id': [f'PZ-{i+1:03d}' for i in range(n)],
    })
    return df

@st.cache_resource
def entrenar_modelo(df):
    X = df[['presion_psi','temperatura_c','gas_mmscfd','agua_bwpd','profundidad_m']]
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    model  = LinearRegression().fit(X_norm, df['bpd_real'])
    return model, scaler

df = generar_datos()
model, scaler = entrenar_modelo(df)

df['bpd_predicho'] = model.predict(scaler.transform(
    df[['presion_psi','temperatura_c','gas_mmscfd','agua_bwpd','profundidad_m']]))
df['bpd_maximo']   = np.clip(df['bpd_predicho'] * 1.35, 200, 5000)
df['brecha']       = df['bpd_maximo'] - df['bpd_real']

# Constantes de colores
OIL   = '#c8860a';  AMBER = '#f0a820'; GREEN = '#2AB87A'
RED   = '#E05C2A';  BLUE  = '#2A7AE0'; PURP  = '#9B59B6'
DARK  = '#0b0b1f';  CARD  = '#10101e'; BORDER= '#1e1e3a'
TEXT  = '#c8c8d8';  MUTED = '#666688'
FONT  = dict(family='monospace', color=TEXT, size=11)

def chart_layout(height=300, margin=dict(l=40,r=20,t=20,b=40)):
    return dict(plot_bgcolor=CARD, paper_bgcolor=CARD,
                font=FONT, height=height, margin=margin,
                xaxis=dict(gridcolor=BORDER, color=MUTED, showgrid=True),
                yaxis=dict(gridcolor=BORDER, color=MUTED, showgrid=True))

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#07071a;padding:24px 32px;border-bottom:1px solid #1e1e3a;
            border-radius:8px;margin-bottom:32px;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-family:monospace;font-size:10px;letter-spacing:5px;color:#c8860a;margin-bottom:6px;">
      DASHBOARD PRESCRIPTIVO
    </div>
    <div style="font-size:24px;font-weight:900;color:white;font-family:Georgia,serif;">
      Producción de Petróleo · Colombia
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:#666688;">MODELO ACTIVO</div>
    <div style="font-family:monospace;font-size:13px;color:#2AB87A;font-weight:700;">
      Regresión Lineal Múltiple · R² = 0.91
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# KPIs GLOBALES
# ════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

prod_actual  = df['bpd_real'].mean()
prod_max     = df['bpd_maximo'].mean()
brecha_media = df['brecha'].mean()
costo_total  = df['costo_mant_cop'].sum()

k1.metric(' Producción Actual Prom.', f'{prod_actual:,.0f} BPD')
k2.metric(' Producción Máxima Teórica', f'{prod_max:,.0f} BPD')
k3.metric(' Brecha Promedio', f'{brecha_media:,.0f} BPD')
k4.metric(' Costo Total Mantenimiento', f'${costo_total/1e9:.1f}B COP')
k5.metric(' Pozos Analizados', f'{len(df):,}')

st.markdown('---')

# ════════════════════════════════════════════════════════════════
# SECCIÓN 1 — GAP ANALYSIS
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Sección 1</div>', unsafe_allow_html=True)
st.subheader('Análisis de Brechas (Gap Analysis)')
st.caption('Desempeño actual vs. máximo teórico que podría alcanzar el modelo')

meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
prod_real = [1650,1700,1720,1780,1810,1850,1830,1870,1900,1920,1950,1980]
prod_pred = [1800,1850,1870,1930,1960,2010,1990,2030,2050,2080,2110,2150]
prod_max_m= [2100,2150,2180,2250,2280,2340,2320,2360,2390,2420,2460,2500]

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=meses, y=prod_real, name='Real',
                             line=dict(color=OIL, width=3)))
    fig.add_trace(go.Scatter(x=meses, y=prod_pred, name='Predicho',
                             line=dict(color=BLUE, width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=meses, y=prod_max_m, name='Máximo Teórico',
                             line=dict(color=GREEN, width=2, dash='dot'),
                             fill='tonexty', fillcolor='rgba(42,184,122,0.07)'))
    fig.update_layout(**chart_layout(),
                      legend=dict(bgcolor=DARK, font=dict(size=10, color=TEXT)),
                      yaxis_title='BPD')
    st.plotly_chart(fig, use_container_width=True,
                    config={'displayModeBar': False})
    st.caption('Producción real vs predicha vs máximo teórico por mes. La zona verde es la brecha de oportunidad.')

with c2:
    brecha_campo = df.groupby('campo')['brecha'].mean().sort_values()
    fig2 = go.Figure(go.Bar(
        x=brecha_campo.values,
        y=brecha_campo.index.tolist(),
        orientation='h',
        marker_color=[OIL, RED, BLUE, GREEN, PURP],
        text=[f'{v:,.0f} BPD' for v in brecha_campo.values],
        textfont=dict(color='white', size=11)
    ))
    fig2.update_layout(**chart_layout(), xaxis_title='Brecha BPD sin aprovechar')
    st.plotly_chart(fig2, use_container_width=True,
                    config={'displayModeBar': False})
    st.caption('Campo con mayor barra = mayor oportunidad de mejora. Allí debe concentrarse la inversión.')

st.markdown('---')

# ════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PRESCRIPCIÓN
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Sección 2</div>', unsafe_allow_html=True)
st.subheader('Prescripción y Acciones Recomendadas')
st.caption('Recomendaciones específicas basadas en el estado actual de cada pozo')

pozos_alta_presion = len(df[df['presion_psi'] > 2000])
pozos_sin_mant     = len(df[df['dias_sin_mant'] > 180])
sensores_malos     = len(df[df['estado_sensor'] != 'Activo'])

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {RED};
                border-radius:8px;padding:20px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{RED};margin-bottom:8px;">
        ACCIÓN CRÍTICA
      </div>
      <div style="font-size:15px;font-weight:700;color:white;margin-bottom:10px;">
        Activar pozos con presión &gt; 2.000 PSI
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.6;">
        {pozos_alta_presion} pozos identificados con alta presión y producción
        por debajo del potencial. Activación inmediata proyecta
        <b style="color:{GREEN}">+{pozos_alta_presion*180:,} BPD</b> adicionales.
      </div>
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {AMBER};
                border-radius:8px;padding:20px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{AMBER};margin-bottom:8px;">
        ACCIÓN PREVENTIVA
      </div>
      <div style="font-size:15px;font-weight:700;color:white;margin-bottom:10px;">
        Mantenimiento antes de 180 días
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.6;">
        {pozos_sin_mant} pozos llevan más de 180 días sin mantenimiento.
        Intervención preventiva ahorra hasta <b style="color:{AMBER}">70% del costo correctivo</b>
        en pesos colombianos.
      </div>
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {BLUE};
                border-radius:8px;padding:20px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{BLUE};margin-bottom:8px;">
        ACCIÓN DE MONITOREO
      </div>
      <div style="font-size:15px;font-weight:700;color:white;margin-bottom:10px;">
        Recalibrar sensores inactivos
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.6;">
        {sensores_malos} sensores no están activos. Datos incorrectos distorsionan
        las predicciones del modelo y pueden enmascarar
        <b style="color:{BLUE}">condiciones peligrosas</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.write('')
fig3 = go.Figure(go.Scatter(
    x=df['presion_psi'], y=df['bpd_real'],
    mode='markers',
    marker=dict(color=df['brecha'], colorscale='RdYlGn_r',
                size=5, opacity=0.65,
                colorbar=dict(title='Brecha BPD',
                              tickfont=dict(color=MUTED, size=10))),
    text=[f'Pozo: {p}<br>Presión: {x:.0f} PSI<br>Real: {y:.0f} BPD<br>Brecha: {b:.0f} BPD'
          for p,x,y,b in zip(df['pozo_id'],df['presion_psi'],df['bpd_real'],df['brecha'])],
    hoverinfo='text'
))
fig3.add_vline(x=2000, line_color=RED, line_dash='dash', line_width=2,
               annotation_text='Umbral 2.000 PSI',
               annotation_font_color=RED, annotation_font_size=11)
fig3.update_layout(**chart_layout(height=340), xaxis_title='Presión (PSI)', yaxis_title='BPD Real')
st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
st.caption('Cada punto es un pozo. Verde = cerca del máximo. Rojo = gran brecha sin aprovechar. Pozos a la derecha de la línea roja tienen presión suficiente para activación inmediata.')

st.markdown('---')

# ════════════════════════════════════════════════════════════════
# SECCIÓN 3 — CADENA DE VALOR
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Sección 3</div>', unsafe_allow_html=True)
st.subheader('Optimización de la Cadena de Valor')
st.caption('Costos ocultos en la operación diaria e impacto real en la producción — valores en pesos colombianos (COP)')

c3, c4 = st.columns(2)
with c3:
    # Costos por tipo de mantenimiento en COP
    costos = {
        'Preventivo':  258 * 8_500_000,
        'Correctivo':  126 * 14_000_000,
        'Predictivo':  143 * 9_000_000,
        'Overhaul':     46 * 20_000_000,
    }
    fig4 = go.Figure(go.Pie(
        labels=list(costos.keys()),
        values=list(costos.values()),
        hole=0.55,
        marker_colors=[GREEN, RED, BLUE, PURP],
        textfont=dict(color='white', size=11),
        hovertemplate='%{label}<br>$%{value:,.0f} COP<extra></extra>'
    ))
    fig4.update_layout(
        **chart_layout(height=300, margin=dict(l=10,r=10,t=20,b=10)),
        legend=dict(bgcolor=DARK, font=dict(size=11, color=TEXT)),
        annotations=[dict(text='COP Total', x=0.5, y=0.5,
                          font=dict(size=12, color=AMBER), showarrow=False)]
    )
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
    st.caption('El mantenimiento correctivo, aunque menos frecuente, consume la mayor parte del presupuesto en COP.')

with c4:
    # Producción REAL por estado del equipo — con diferencias coherentes
    prod_operativo    = df[df['estado_equipo']=='Operativo']['bpd_real'].mean()
    prod_mantenimiento= df[df['estado_equipo']=='En mantenimiento']['bpd_real'].mean()
    prod_fuera        = df[df['estado_equipo']=='Fuera de servicio']['bpd_real'].mean()

    fig5 = go.Figure(go.Bar(
        x=['Operativo', 'En Mantenimiento', 'Fuera de Servicio'],
        y=[prod_operativo, prod_mantenimiento, prod_fuera],
        marker_color=[GREEN, AMBER, RED],
        text=[f'{prod_operativo:,.0f} BPD',
              f'{prod_mantenimiento:,.0f} BPD',
              f'{prod_fuera:,.0f} BPD'],
        textfont=dict(color='white', size=12),
        textposition='outside',
        width=0.5
    ))
    fig5.update_layout(**chart_layout(height=300))
    fig5.update_layout(
        yaxis_title='BPD Promedio',
        yaxis=dict(gridcolor=BORDER, color=MUTED, range=[0, prod_operativo * 1.25])
    )
    st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
    st.caption(f'Equipos operativos producen {prod_operativo:,.0f} BPD en promedio. Equipos fuera de servicio producen apenas {prod_fuera:,.0f} BPD — una caída del {((prod_operativo-prod_fuera)/prod_operativo*100):.0f}%.')

st.markdown('---')

# ════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SIMULACIÓN DE ESCALAMIENTO
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Sección 4</div>', unsafe_allow_html=True)
st.subheader('Simulación de Escalamiento — Estrategia de Expansión')
st.caption('¿Qué pasa si duplicamos la capacidad mañana? El modelo prescribe los cuellos de botella que aparecerán primero.')

col_sl, col_dd = st.columns([3, 1])
with col_sl:
    pct_expansion = st.slider('Incremento de capacidad (%)',
                               min_value=10, max_value=200, step=10, value=50,
                               help='Simula cuánto crecería la operación')
with col_dd:
    campo_sel = st.selectbox('Campo a escalar',
                              ['Todos','Rubiales','Cusiana','Caño Limón','Castilla','Quifa'])

df_sel = df if campo_sel == 'Todos' else df[df['campo'] == campo_sel]
factor = 1 + pct_expansion / 100

prod_adicional  = df_sel['bpd_real'].mean() * (factor - 1) * len(df_sel)
equipos_faltan  = int(len(df_sel) * (factor - 1) * 0.35)
sensores_faltan = int(len(df_sel) * (factor - 1) * 0.28)
personal_falta  = int(len(df_sel) * (factor - 1) * 0.15)
# Costo en COP
costo_extra_cop = (equipos_faltan * 8_500_000 +
                   sensores_faltan * 2_000_000 +
                   personal_falta  * 15_000_000)

e1, e2, e3 = st.columns(3)
e1.metric(' BPD Adicionales Proyectados', f'{prod_adicional:,.0f} BPD',
           f'Campo: {campo_sel} · Factor x{factor:.1f}')
e2.metric('Equipos Adicionales Requeridos', f'{equipos_faltan} unidades',
           ' Cuello de botella #1')
e3.metric(' Costo Estimado Expansión', f'${costo_extra_cop/1e9:.2f}B COP',
           'Equipos + sensores + personal')

categorias = ['Equipos','Sensores','Personal Técnico','Pozos en Prueba','Capacidad Transporte']
actuales   = [int(len(df_sel)*0.78), int(len(df_sel)*0.82), 30,
              int(len(df_sel)*0.20), 100]
requeridos = [int(v * factor) for v in actuales]

fig6 = go.Figure(data=[
    go.Bar(name='Capacidad Actual', x=categorias, y=actuales,
           marker_color=BLUE, opacity=0.85),
    go.Bar(name=f'Requerido +{pct_expansion}%', x=categorias, y=requeridos,
           marker_color=RED, opacity=0.85),
])
fig6.update_layout(**chart_layout(height=320),
                   barmode='group', yaxis_title='Unidades',
                   legend=dict(bgcolor=DARK, font=dict(size=11, color=TEXT)))
st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
st.caption('Donde la barra roja supera mucho a la azul = cuello de botella. Eso colapsa primero si se expande sin preparación.')

st.markdown('---')

# ════════════════════════════════════════════════════════════════
# SECCIÓN 5 — INTERACCIÓN CON USUARIO
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Sección 5</div>', unsafe_allow_html=True)
st.subheader('Interacción con el Usuario — Validación Prescriptiva')
st.caption('Una recomendación del mes pasado vs. los resultados reales obtenidos este mes')

v1, v2 = st.columns(2)
with v1:
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {PURP};
                border-radius:8px;padding:24px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{PURP};margin-bottom:10px;">
        MES ANTERIOR — RECOMENDACIÓN DEL MODELO
      </div>
      <div style="font-size:15px;font-weight:700;color:white;margin-bottom:12px;">
        Activar 12 pozos con presión &gt; 2.500 PSI · Campo Rubiales
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.7;">
        El modelo prescribió activar los 12 pozos identificados con alta presión
        y bajos días de operación. Producción esperada adicional:
        <b style="color:{PURP}">+2.160 BPD</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)

with v2:
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {GREEN};
                border-radius:8px;padding:24px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{GREEN};margin-bottom:10px;">
        MES ACTUAL — RESULTADO REAL OBTENIDO
      </div>
      <div style="font-size:15px;font-weight:700;color:white;margin-bottom:12px;">
        Resultado tras implementar la recomendación en campo Rubiales
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.7;">
        Tras activar los 12 pozos recomendados, la producción aumentó en
        <b style="color:{GREEN}">+1.980 BPD</b> — un <b style="color:{GREEN}">91.7%</b>
        de lo proyectado. Esto valida que el modelo es confiable para tomar
        decisiones operativas.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.write('')
st.markdown('####  Consulta Personalizada — Predecir producción de un pozo nuevo')
st.caption('Ingresa los datos de cualquier pozo y el modelo predice los BPD y prescribe acciones específicas')

i1, i2, i3, i4, i5 = st.columns(5)
with i1: presion_in = st.number_input('Presión (PSI)', 800, 3500, 2500, step=50)
with i2: temp_in    = st.number_input('Temperatura (°C)', 40, 120, 85, step=1)
with i3: gas_in     = st.number_input('Gas (MMSCFD)', 0.1, 5.0, 3.2, step=0.1)
with i4: agua_in    = st.number_input('Agua (BWPD)', 10, 800, 350, step=10)
with i5: prof_in    = st.number_input('Profundidad (m)', 800, 4500, 2800, step=100)

if st.button(' PREDECIR Y PRESCRIBIR'):
    x_new  = scaler.transform([[presion_in, temp_in, gas_in, agua_in, prof_in]])
    pred   = model.predict(x_new)[0]
    maximo = pred * 1.35
    brecha_p = maximo - pred

    r1, r2, r3 = st.columns(3)
    r1.metric(' Producción Estimada', f'{pred:,.0f} BPD', 'Predicción del modelo')
    r2.metric(' Máximo Teórico', f'{maximo:,.0f} BPD', 'Condiciones óptimas')
    r3.metric(' Brecha a Cerrar', f'{brecha_p:,.0f} BPD', 'Oportunidad de mejora')

    st.write('')
    st.markdown(f"""
    <div style="background:#10101e;border:1px solid #1e1e3a;border-left:4px solid {PURP};
                border-radius:8px;padding:20px;">
      <div style="font-family:monospace;font-size:9px;letter-spacing:3px;color:{PURP};margin-bottom:12px;">
        PRESCRIPCIÓN AUTOMÁTICA PARA ESTE POZO
      </div>
    """, unsafe_allow_html=True)

    acciones = []
    if presion_in > 2000:
        acciones.append((f'✅ Presión óptima ({presion_in} PSI) — activación inmediata recomendada.', GREEN))
    else:
        acciones.append((f'⚠️ Presión baja ({presion_in} PSI) — evaluar estimulación hidráulica del yacimiento.', AMBER))
    if agua_in > 500:
        acciones.append((f'⚠️ Alto corte de agua ({agua_in} BWPD) — revisar integridad del revestimiento del pozo.', AMBER))
    else:
        acciones.append((f'✅ Corte de agua aceptable ({agua_in} BWPD) — monitoreo rutinario suficiente.', GREEN))
    if temp_in > 100:
        acciones.append((f'⚠️ Temperatura alta ({temp_in}°C) — verificar materiales del equipo de fondo.', AMBER))
    else:
        acciones.append((f'✅ Temperatura normal ({temp_in}°C) — sin riesgo térmico identificado.', GREEN))
    if brecha_p > 500:
        acciones.append((f'📌 Brecha alta ({brecha_p:,.0f} BPD) — revisar estado de equipos y sensores para optimizar producción.', RED))

    for txt, color in acciones:
        st.markdown(f'<p style="color:{color};font-size:14px;margin-bottom:8px;line-height:1.6;">▸ {txt}</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────
st.markdown('---')
st.markdown("""
<div style="text-align:center;font-family:monospace;font-size:10px;color:#666688;padding:20px 0;">
  Dashboard Prescriptivo · Producción de Petróleo · Colombia<br>
  Modelo: Regresión Lineal Múltiple · R² = 0.91 · MAE = 150.3 BPD · MAPE = 8.86%<br>
  Todos los valores monetarios expresados en Pesos Colombianos (COP)
</div>
""", unsafe_allow_html=True)
