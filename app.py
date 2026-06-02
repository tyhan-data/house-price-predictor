import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏡 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .main { background-color: #f0f4f8; }
    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #e8edf5 100%); }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(15,52,96,0.3);
        text-align: center;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #a8c6fa;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,179,237,0.2);
        color: #63b3ed;
        border: 1px solid rgba(99,179,237,0.4);
        border-radius: 20px;
        padding: 0.25rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #ebf0f8;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(15,52,96,0.35);
        margin: 1rem 0;
    }
    .result-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #a8c6fa;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
    }
    .result-price {
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .result-range {
        font-size: 0.9rem;
        color: #63b3ed;
        margin-top: 0.8rem;
    }

    /* Stat badges */
    .stat-row {
        display: flex;
        gap: 0.8rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }
    .stat-badge {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        text-align: center;
        min-width: 100px;
    }
    .stat-value { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .stat-key { font-size: 0.7rem; color: #a8c6fa; text-transform: uppercase; letter-spacing: 1px; }

    /* Tip box */
    .tip-box {
        background: linear-gradient(135deg, #ebf8ff, #e6fffa);
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #2c5282;
        margin-top: 1rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .css-1d391kg { background: transparent; }

    /* Button style */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a5276);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(15,52,96,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a4a7a, #2471a3);
        box-shadow: 0 6px 20px rgba(15,52,96,0.45);
        transform: translateY(-1px);
    }

    /* Input styling */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1.5px solid #e2e8f0 !important;
    }

    /* Slider */
    .stSlider { padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("model/cat_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ─── Feature Defaults & Mappings ───────────────────────────────────────────────
NEIGHBORHOOD_MAP = {
    "Bloomington Heights": 0, "Bluestem": 1, "Briardale": 2, "Brookside": 3,
    "Clear Creek": 4, "College Creek": 5, "Crawford": 6, "Edwards": 7,
    "Gilbert": 8, "Greens": 9, "GreenHills": 10, "Iowa DOT/RR": 11,
    "Landmark": 12, "Meadow Village": 13, "Mitchell": 14, "North Ames": 15,
    "Northpark Villa": 16, "Northridge": 17, "Northridge Heights": 18,
    "Northwest Ames": 19, "Old Town": 20, "Sawyer": 21, "Sawyer West": 22,
    "Somerset": 23, "Stone Brook": 24, "Timberland": 25, "Veenker": 26,
}
QUAL_MAP = {"Poor": 0, "Fair": 1, "Below Average": 2, "Average": 3, "Good": 4, "Excellent": 5}
BLDG_TYPE_MAP = {"Single Family": 0, "2-family Conversion": 1, "Duplex": 2, "Townhouse End": 3, "Townhouse Inside": 4}
HOUSE_STYLE_MAP = {"1 Story": 0, "1.5 Fin": 1, "1.5 Unf": 2, "2 Story": 3, "2.5 Fin": 4, "2.5 Unf": 5, "Split Foyer": 6, "Split Level": 7}
ZONING_MAP = {"Agriculture": 0, "Commercial": 1, "Floating Village Res.": 2, "Industrial": 3, "Res. High Density": 4, "Res. Low Density": 5, "Res. Low Density Park": 6, "Res. Medium Density": 7}
FOUNDATION_MAP = {"Brick & Tile": 0, "Cinder Block": 1, "Poured Concrete": 2, "Slab": 3, "Stone": 4, "Wood": 5}
GARAGE_TYPE_MAP = {"None": 0, "2+ Types": 1, "Attached": 2, "Basement": 3, "Built-In": 4, "Car Port": 5, "Detached": 6}
GARAGE_FINISH_MAP = {"None": 0, "Rough Finished": 1, "Finished": 2, "Unfinished": 3}
KITCHEN_QUAL_MAP = {"Poor": 0, "Fair": 1, "Typical": 2, "Good": 3, "Excellent": 4}

# ─── Helper: build full feature vector ─────────────────────────────────────────
def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """Map user inputs to the full 79-feature vector expected by the model."""

    # Derived / computed features
    year_built = inputs["year_built"]
    year_remod = inputs["year_remod"]
    garage_yr = inputs.get("garage_yr_blt", year_built)
    yr_sold = 2010  # typical value

    house_age   = yr_sold - year_built
    remod_age   = yr_sold - year_remod
    garage_age  = yr_sold - garage_yr if garage_yr > 0 else house_age

    total_sf    = inputs["total_bsmt_sf"] + inputs["first_flr_sf"] + inputs["second_flr_sf"]
    total_bath  = inputs["full_bath"] + 0.5 * inputs["half_bath"] + inputs["bsmt_full_bath"] + 0.5 * inputs["bsmt_half_bath"]
    overall_score = inputs["overall_qual"] * inputs["overall_cond"]
    gr_liv_area = inputs["first_flr_sf"] + inputs["second_flr_sf"]

    row = {
        'MS SubClass':    60,
        'MS Zoning':      inputs["ms_zoning"],
        'Lot Frontage':   inputs["lot_frontage"],
        'Lot Area':       inputs["lot_area"],
        'Street':         1,
        'Lot Shape':      3,
        'Land Contour':   3,
        'Utilities':      3,
        'Lot Config':     4,
        'Land Slope':     2,
        'Neighborhood':   inputs["neighborhood"],
        'Condition 1':    2,
        'Condition 2':    2,
        'Bldg Type':      inputs["bldg_type"],
        'House Style':    inputs["house_style"],
        'Overall Qual':   inputs["overall_qual"],
        'Overall Cond':   inputs["overall_cond"],
        'Year Built':     year_built,
        'Year Remod/Add': year_remod,
        'Roof Style':     1,
        'Roof Matl':      1,
        'Exterior 1st':   13,
        'Exterior 2nd':   13,
        'Mas Vnr Area':   inputs["mas_vnr_area"],
        'Exter Qual':     inputs["exter_qual"],
        'Exter Cond':     2,
        'Foundation':     inputs["foundation"],
        'Bsmt Qual':      inputs["bsmt_qual"],
        'Bsmt Cond':      3,
        'Bsmt Exposure':  3,
        'BsmtFin Type 1': 5,
        'BsmtFin SF 1':   inputs["bsmtfin_sf1"],
        'BsmtFin Type 2': 6,
        'BsmtFin SF 2':   0.0,
        'Bsmt Unf SF':    max(0, inputs["total_bsmt_sf"] - inputs["bsmtfin_sf1"]),
        'Total Bsmt SF':  inputs["total_bsmt_sf"],
        'Heating':        1,
        'Heating QC':     3,
        'Central Air':    1,
        'Electrical':     4,
        '1st Flr SF':     inputs["first_flr_sf"],
        '2nd Flr SF':     inputs["second_flr_sf"],
        'Low Qual Fin SF': 0,
        'Gr Liv Area':    gr_liv_area,
        'Bsmt Full Bath': inputs["bsmt_full_bath"],
        'Bsmt Half Bath': inputs["bsmt_half_bath"],
        'Full Bath':      inputs["full_bath"],
        'Half Bath':      inputs["half_bath"],
        'Bedroom AbvGr':  inputs["bedroom_abvgr"],
        'Kitchen AbvGr':  1,
        'Kitchen Qual':   inputs["kitchen_qual"],
        'TotRms AbvGrd':  inputs["tot_rms_abvgrd"],
        'Functional':     6,
        'Fireplaces':     inputs["fireplaces"],
        'Garage Type':    inputs["garage_type"],
        'Garage Yr Blt':  float(garage_yr),
        'Garage Finish':  inputs["garage_finish"],
        'Garage Cars':    float(inputs["garage_cars"]),
        'Garage Area':    float(inputs["garage_area"]),
        'Garage Qual':    4,
        'Garage Cond':    4,
        'Paved Drive':    2,
        'Wood Deck SF':   inputs["wood_deck_sf"],
        'Open Porch SF':  inputs["open_porch_sf"],
        'Enclosed Porch': 0,
        '3Ssn Porch':     0,
        'Screen Porch':   0,
        'Pool Area':      0,
        'Misc Val':       0,
        'Mo Sold':        6,
        'Yr Sold':        yr_sold,
        'Sale Type':      8,
        'Sale Condition': 4,
        'House Age':      house_age,
        'Remod Age':      remod_age,
        'Garage Age':     float(garage_age),
        'Total SF':       total_sf,
        'Overall Score':  overall_score,
        'Total Bath':     total_bath,
    }
    return pd.DataFrame([row])


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🤖 AI-Powered Prediction</div>
    <div class="hero-title">🏡 House Price Predictor</div>
    <div class="hero-subtitle">CatBoost ML Model • Ames, Iowa Housing Dataset • Real-Time Estimation</div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR INFO ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem 0;">
        <div style="font-size:2.5rem">🏠</div>
        <div style="color:#a8c6fa; font-weight:700; font-size:1.1rem; margin-top:0.5rem;">How It Works</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#cbd5e0; font-size:0.88rem; line-height:1.7; padding:0 0.5rem;">
    <b style="color:#63b3ed;">1. Fill in Details</b><br>
    Enter the key characteristics of the property across the 4 sections.<br><br>
    <b style="color:#63b3ed;">2. Click Predict</b><br>
    Our CatBoost model trained on 1,400+ Ames homes processes your inputs.<br><br>
    <b style="color:#63b3ed;">3. Get Estimate</b><br>
    Receive an instant price prediction with a confidence range.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#718096; font-size:0.78rem; padding:0 0.5rem;">
    <b style="color:#a8c6fa;">Model Details</b><br>
    • Algorithm: CatBoost Regressor<br>
    • Features: 79 engineered variables<br>
    • Dataset: Ames Housing (Iowa)<br>
    • Training: 1,460 homes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#718096; font-size:0.75rem; padding:0 0.5rem; font-style:italic;">
    ⚠️ Predictions are estimates for the Ames, Iowa market. 
    Always consult a real estate professional for final valuations.
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN FORM ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:

    # ── SECTION 1: Property Basics ──
    st.markdown("""<div class="section-card">
    <div class="section-title">🏗️ Property Basics</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            overall_qual = st.slider("Overall Quality", 1, 10, 7,
                help="Overall material and finish quality (1=Very Poor, 10=Excellent)")
        with r1c2:
            overall_cond = st.slider("Overall Condition", 1, 10, 5,
                help="Overall condition rating (1=Very Poor, 10=Excellent)")
        with r1c3:
            neighborhood = st.selectbox("Neighborhood", list(NEIGHBORHOOD_MAP.keys()), index=15,
                help="Physical location within Ames city limits")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            bldg_type = st.selectbox("Building Type", list(BLDG_TYPE_MAP.keys()), index=0)
        with r2c2:
            house_style = st.selectbox("House Style", list(HOUSE_STYLE_MAP.keys()), index=3)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            year_built = st.number_input("Year Built", 1872, 2010, 2000)
        with r3c2:
            year_remod = st.number_input("Year Remodeled", 1950, 2010, 2005,
                help="Year of last remodel/addition (same as Year Built if none)")
        with r3c3:
            ms_zoning = st.selectbox("Zoning", list(ZONING_MAP.keys()), index=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: Size & Area ──
    st.markdown("""<div class="section-card">
    <div class="section-title">📐 Size & Area (sq ft)</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        s1c1, s1c2, s1c3 = st.columns(3)
        with s1c1:
            lot_area = st.number_input("Lot Area (sq ft)", 1000, 200000, 9000)
        with s1c2:
            lot_frontage = st.number_input("Lot Frontage (ft)", 0, 300, 70)
        with s1c3:
            mas_vnr_area = st.number_input("Masonry Veneer Area", 0, 1600, 100,
                help="Masonry veneer area in sq ft")

        s2c1, s2c2, s2c3 = st.columns(3)
        with s2c1:
            first_flr_sf = st.number_input("1st Floor (sq ft)", 300, 4000, 1000)
        with s2c2:
            second_flr_sf = st.number_input("2nd Floor (sq ft)", 0, 2500, 800)
        with s2c3:
            total_bsmt_sf = st.number_input("Total Basement (sq ft)", 0, 3000, 1000)

        s3c1, s3c2, s3c3 = st.columns(3)
        with s3c1:
            bsmtfin_sf1 = st.number_input("Finished Bsmt SF", 0, 2000, 400)
        with s3c2:
            wood_deck_sf = st.number_input("Wood Deck (sq ft)", 0, 900, 100)
        with s3c3:
            open_porch_sf = st.number_input("Open Porch (sq ft)", 0, 600, 50)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Rooms & Amenities ──
    st.markdown("""<div class="section-card">
    <div class="section-title">🛏️ Rooms & Amenities</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        rm1, rm2, rm3, rm4 = st.columns(4)
        with rm1:
            bedroom_abvgr = st.number_input("Bedrooms", 0, 8, 3)
        with rm2:
            full_bath = st.number_input("Full Baths", 0, 4, 2)
        with rm3:
            half_bath = st.number_input("Half Baths", 0, 2, 1)
        with rm4:
            tot_rms_abvgrd = st.number_input("Total Rooms", 2, 14, 8,
                help="Total rooms above grade (excl. bathrooms)")

        rm5, rm6, rm7, rm8 = st.columns(4)
        with rm5:
            bsmt_full_bath = st.number_input("Bsmt Full Bath", 0, 3, 1)
        with rm6:
            bsmt_half_bath = st.number_input("Bsmt Half Bath", 0, 2, 0)
        with rm7:
            fireplaces = st.number_input("Fireplaces", 0, 4, 1)
        with rm8:
            kitchen_qual = st.selectbox("Kitchen Quality",
                list(KITCHEN_QUAL_MAP.keys()), index=3)


with col_right:

    # ── SECTION 4: Construction & Garage ──
    st.markdown("""<div class="section-card">
    <div class="section-title">🔨 Construction Quality</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        foundation = st.selectbox("Foundation", list(FOUNDATION_MAP.keys()), index=2)

        cq1, cq2 = st.columns(2)
        with cq1:
            exter_qual = st.selectbox("Exterior Quality",
                ["Poor","Fair","Below Average","Average","Good","Excellent"], index=4)
        with cq2:
            bsmt_qual = st.selectbox("Basement Quality",
                ["Poor","Fair","Below Average","Average","Good","Excellent"], index=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 5: Garage ──
    st.markdown("""<div class="section-card">
    <div class="section-title">🚗 Garage</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        garage_type = st.selectbox("Garage Type", list(GARAGE_TYPE_MAP.keys()), index=2)

        gc1, gc2 = st.columns(2)
        with gc1:
            garage_cars = st.number_input("Garage Capacity (cars)", 0, 5, 2)
        with gc2:
            garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 450)

        gf1, gf2 = st.columns(2)
        with gf1:
            garage_finish = st.selectbox("Garage Finish", list(GARAGE_FINISH_MAP.keys()), index=2)
        with gf2:
            garage_yr_blt = st.number_input("Garage Year Built", 1900, 2010, 2000)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮  Predict House Price", use_container_width=True)

    # ── RESULT ──
    if predict_clicked:
        # Build inputs dict
        inputs = {
            "ms_zoning":       ZONING_MAP[ms_zoning],
            "lot_frontage":    float(lot_frontage),
            "lot_area":        int(lot_area),
            "neighborhood":    NEIGHBORHOOD_MAP[neighborhood],
            "bldg_type":       BLDG_TYPE_MAP[bldg_type],
            "house_style":     HOUSE_STYLE_MAP[house_style],
            "overall_qual":    int(overall_qual),
            "overall_cond":    int(overall_cond),
            "year_built":      int(year_built),
            "year_remod":      int(year_remod),
            "mas_vnr_area":    float(mas_vnr_area),
            "exter_qual":      QUAL_MAP.get(exter_qual, 4),
            "foundation":      FOUNDATION_MAP[foundation],
            "bsmt_qual":       QUAL_MAP.get(bsmt_qual, 4),
            "bsmtfin_sf1":     float(bsmtfin_sf1),
            "total_bsmt_sf":   float(total_bsmt_sf),
            "first_flr_sf":    int(first_flr_sf),
            "second_flr_sf":   int(second_flr_sf),
            "bsmt_full_bath":  float(bsmt_full_bath),
            "bsmt_half_bath":  float(bsmt_half_bath),
            "full_bath":       int(full_bath),
            "half_bath":       int(half_bath),
            "bedroom_abvgr":   int(bedroom_abvgr),
            "kitchen_qual":    KITCHEN_QUAL_MAP[kitchen_qual],
            "tot_rms_abvgrd":  int(tot_rms_abvgrd),
            "fireplaces":      int(fireplaces),
            "garage_type":     GARAGE_TYPE_MAP[garage_type],
            "garage_yr_blt":   int(garage_yr_blt),
            "garage_finish":   GARAGE_FINISH_MAP[garage_finish],
            "garage_cars":     int(garage_cars),
            "garage_area":     float(garage_area),
            "wood_deck_sf":    int(wood_deck_sf),
            "open_porch_sf":   int(open_porch_sf),
        }

        with st.spinner("Running CatBoost prediction..."):
            df_input = build_feature_vector(inputs)
            prediction = model.predict(df_input)[0]
            price = float(prediction)
            low  = price * 0.90
            high = price * 1.10

        total_sf_val = total_bsmt_sf + first_flr_sf + second_flr_sf
        price_per_sqft = price / total_sf_val if total_sf_val > 0 else 0
        total_bath_val = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-range">90% Confidence: ${low:,.0f} – ${high:,.0f}</div>
            <div class="stat-row">
                <div class="stat-badge">
                    <div class="stat-value">${price_per_sqft:,.0f}</div>
                    <div class="stat-key">Per Sq Ft</div>
                </div>
                <div class="stat-badge">
                    <div class="stat-value">{total_sf_val:,}</div>
                    <div class="stat-key">Total SF</div>
                </div>
                <div class="stat-badge">
                    <div class="stat-value">{total_bath_val:.1f}</div>
                    <div class="stat-key">Total Baths</div>
                </div>
                <div class="stat-badge">
                    <div class="stat-value">{overall_qual}/10</div>
                    <div class="stat-key">Quality</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quality indicator
        if price < 100000:
            tier, color = "Budget Home", "#fc8181"
        elif price < 175000:
            tier, color = "Mid-Range Home", "#f6ad55"
        elif price < 275000:
            tier, color = "Upper Mid-Range", "#68d391"
        elif price < 400000:
            tier, color = "Luxury Home", "#63b3ed"
        else:
            tier, color = "Premium Luxury", "#b794f4"

        st.markdown(f"""
        <div style="text-align:center; margin-top:0.8rem;">
            <span style="background:{color}22; color:{color}; border:1.5px solid {color}55;
                border-radius:20px; padding:0.3rem 1rem; font-size:0.85rem; font-weight:700;">
                🏷️ {tier}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box">
        💡 <b>Pro Tip:</b> The top factors driving this prediction are 
        <b>Total SF</b>, <b>Overall Quality</b>, and <b>Neighborhood</b>. 
        Improving quality rating or adding finished space can significantly boost estimated value.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder card
        st.markdown("""
        <div style="background:white; border-radius:20px; padding:2.5rem 2rem;
             text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.06);
             border: 2px dashed #e2e8f0; margin:1rem 0;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏠</div>
            <div style="font-size:1.1rem; font-weight:700; color:#2d3748; margin-bottom:0.5rem;">
                Ready to Estimate
            </div>
            <div style="font-size:0.9rem; color:#718096; line-height:1.6;">
                Fill in the property details on the left,<br>
                then click <b>Predict House Price</b><br>
                to get an instant AI-powered valuation.
            </div>
            <div style="margin-top:1.5rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
                <span style="background:#ebf8ff; color:#2b6cb0; border-radius:8px; 
                    padding:0.3rem 0.8rem; font-size:0.78rem; font-weight:600;">⚡ Instant Results</span>
                <span style="background:#f0fff4; color:#276749; border-radius:8px; 
                    padding:0.3rem 0.8rem; font-size:0.78rem; font-weight:600;">🎯 High Accuracy</span>
                <span style="background:#faf5ff; color:#553c9a; border-radius:8px; 
                    padding:0.3rem 0.8rem; font-size:0.78rem; font-weight:600;">🤖 CatBoost AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.8rem; padding:1rem 0; 
     border-top:1px solid #e2e8f0; margin-top:1rem;">
    Built with <b>Streamlit</b> + <b>CatBoost</b> • Ames Housing Dataset • 
    For educational purposes only
</div>
""", unsafe_allow_html=True)
