import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ========================
# ⚙️ CONFIGURATION
# ========================
st.set_page_config(
    page_title="Sales Forecasting — Rossmann",
    page_icon="📈",
    layout="centered"
)

# ========================
# 🎨 STYLING
# ========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 780px;
        }
        .hero {
            text-align: center;
            padding: 32px 24px 24px;
            border-radius: 20px;
            background: linear-gradient(140deg, #1a3a2a 0%, #1e4d35 60%, #27ae60 100%);
            margin-bottom: 28px;
        }
        .hero h1 {
            font-size: 40px;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 6px 0;
        }
        .hero p {
            font-size: 15px;
            color: #a8e6c1;
            margin: 0;
        }
        .section-label {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
            margin-top: 20px;
        }
        .result-card {
            background: linear-gradient(135deg, #1a3a2a, #27ae60);
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0 8px 24px rgba(39,174,96,0.25);
        }
        .result-label {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #a8e6c1;
            margin-bottom: 8px;
        }
        .result-value {
            font-size: 52px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1;
            margin-bottom: 6px;
        }
        .result-sub {
            font-size: 13px;
            color: #a8e6c1;
        }
        .insight-row {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }
        .insight-pill {
            background: rgba(39,174,96,0.12);
            border: 1px solid rgba(39,174,96,0.3);
            border-radius: 20px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: 600;
            color: #1a5c38;
        }
        .footer {
            text-align: center;
            padding-top: 24px;
            font-size: 12px;
            color: #aaa;
        }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========================
# 🌿 HERO
# ========================
st.markdown("""
    <div class="hero">
        <h1>📈 Sales Forecasting</h1>
        <p>XGBoost model trained on 1M+ Rossmann store records · R² 0.91 · MAPE 9.9%</p>
    </div>
""", unsafe_allow_html=True)

# ========================
# ⚙️ Load Model
# ========================
@st.cache_resource
def load_model():
    with open('rossmann_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ========================
# 📋 ENCODING MAPS
# ========================
encoding_maps = {
    'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3},
    'StoreType':    {'a': 0, 'b': 1, 'c': 2, 'd': 3},
    'Assortment':   {'a': 0, 'b': 1, 'c': 2},
    'PromoInterval':{'': 0, 'Jan,Apr,Jul,Oct': 1,
                     'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
}

# ========================
# 🚀 INPUTS
# ========================
st.markdown('<div class="section-label">Store Details</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    store_id   = st.number_input("🏬 Store ID", min_value=1, max_value=1115, step=1, value=1)
    store_type = st.selectbox("🏪 Store Type", ['a','b','c','d'],
                              help="a=Standard, b=Large, c=Small, d=Extra Large")
    assortment = st.selectbox("📦 Assortment", ['a','b','c'],
                              help="a=Basic, b=Extra, c=Extended")

with col2:
    date_input   = st.date_input("📅 Forecast Date", value=datetime.today())
    promo        = st.selectbox("🎯 Running Promo?", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
    state_holiday = st.selectbox("🏖️ State Holiday", ['0','a','b','c'],
                                 format_func=lambda x: {'0':'None','a':'Public','b':'Easter','c':'Christmas'}[x])

with col3:
    competition_distance          = st.number_input("📏 Competition Distance (m)", min_value=0, value=1000, step=100)
    competition_open_since_month  = st.slider("📆 Competitor Open Since (Month)", 1, 12, value=1)
    competition_open_since_year   = st.number_input("📆 Competitor Open Since (Year)", min_value=1900, max_value=2026, value=2015, step=1)
    promo_interval                = st.selectbox("📆 Promo Interval", ['','Jan,Apr,Jul,Oct','Feb,May,Aug,Nov','Mar,Jun,Sept,Dec'])

# ========================
# 🔮 PREDICTION
# ========================
if st.button("🔮 Forecast Sales", use_container_width=True):
    try:
        year      = date_input.year
        month     = date_input.month
        day       = date_input.day
        dow       = date_input.weekday()
        week      = date_input.isocalendar()[1]
        is_weekend = 1 if dow >= 5 else 0
        is_promo_weekend = 1 if (promo == 1 and dow >= 5) else 0
        comp_open_months = max(0, (year - competition_open_since_year) * 12 +
                               (month - competition_open_since_month))
        promo2         = 1 if promo_interval != '' else 0
        promo_strength = promo + promo2

        input_data = pd.DataFrame([{
            'Store':                       store_id,
            'DayOfWeek':                   dow,
            'Open':                        1,
            'Promo':                       promo,
            'StateHoliday':                encoding_maps['StateHoliday'][state_holiday],
            'SchoolHoliday':               0,
            'StoreType':                   encoding_maps['StoreType'][store_type],
            'Assortment':                  encoding_maps['Assortment'][assortment],
            'CompetitionDistance':         competition_distance,
            'CompetitionOpenSinceMonth':   competition_open_since_month,
            'CompetitionOpenSinceYear':    competition_open_since_year,
            'Promo2':                      promo2,
            'Promo2SinceWeek':             0,
            'Promo2SinceYear':             0,
            'PromoInterval':               encoding_maps['PromoInterval'][promo_interval],
            'Year':                        year,
            'Month':                       month,
            'Day':                         day,
            'WeekOfYear':                  week,
            'IsWeekend':                   is_weekend,
            'IsPromoWeekend':              is_promo_weekend,
            'CompetitionOpenMonths':       comp_open_months,
            'IsPromo2Active':              promo2,
            'PromoStrength':               promo_strength,
        }])

        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        promo_note = "📢 Promo active — expected sales uplift" if promo == 1 else "No promo running"
        weekend_note = "📅 Weekend day" if is_weekend else f"📅 {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]}"
        comp_note = f"🏪 Nearest competitor: {competition_distance:,}m away"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Daily Sales</div>
            <div class="result-value">€{prediction:,.0f}</div>
            <div class="result-sub">Store {store_id} · {date_input.strftime('%d %b %Y')}</div>
        </div>
        <div class="insight-row">
            <span class="insight-pill">{promo_note}</span>
            <span class="insight-pill">{weekend_note}</span>
            <span class="insight-pill">{comp_note}</span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ========================
# FOOTER
# ========================
st.markdown("""
    <div class="footer">
        ⚠️ Predictions based on Rossmann historical data (2013–2015) and may not reflect current conditions.<br>
        📈 Sales Forecasting · Developed by Muhammad Sami
    </div>
""", unsafe_allow_html=True)
