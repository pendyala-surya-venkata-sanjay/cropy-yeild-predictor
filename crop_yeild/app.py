import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="logo.png",
    layout="wide"
)

# -----------------------------------------------------
# PERFECT CENTERED LOGO + TITLE (Final Clean Version)
# -----------------------------------------------------

# Center page layout
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# One centered column for logo + title
c = st.container()
with c:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

    st.image("images/logo.png", width=200)

    st.markdown(
        "<h1 style='text-align:center; color:black; margin-top:10px;'>"
        "Crop Yield Predictor"
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------
# SIMPLE CLEAN UI STYLE
# -----------------------------------------------------
st.markdown("""
<style>
body { background-color: white; color: black; }

.reco-card {
    background: #f6fff6;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #4CAF50;
    margin: 6px 0;
}
.stButton>button {
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# WEATHER API + DISTRICT TO CITY FIXES
# -----------------------------------------------------
WEATHER_API_KEY = "3ebf251a6f8a7098285fb2ec247c8410"

district_to_city = {
    "Vizag": "Visakhapatnam", "Vizianagaram": "Vizianagaram",
    "Banglore": "Bengaluru", "Bangalore": "Bengaluru",
    "Hyd": "Hyderabad", "Gulbarga": "Kalaburagi",
    "Calicut": "Kozhikode", "Cochin": "Kochi",
    "Trivandrum": "Thiruvananthapuram", "Madras": "Chennai",
    "Pondy": "Puducherry", "Bombay": "Mumbai",
    "Delhi NCR": "New Delhi", "Warangal": "Warangal",
    "Karimnagar": "Karimnagar", "Guntur": "Guntur",
    "Nellore": "Nellore", "Kurnool": "Kurnool", "Chittoor": "Chittoor"
}

def clean_city_name(district):
    return district_to_city.get(district, district)

# -----------------------------------------------------
# LOAD DATASET & MODEL
# -----------------------------------------------------
@st.cache_data
def load_dataset():
    pd.read_csv("data/crop_yield_dataset1.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_dataset()

@st.cache_resource
def load_model():
    with open("model/best_crop_yield_model.pkl","rb") as f:
        return pickle.load(f)

bundle = load_model()
model_pipeline = bundle["model"]
model_features = bundle["features"]
crop_means = bundle["crop_means"]

# -----------------------------------------------------
# DISTRICT RAINFALL FALLBACK
# -----------------------------------------------------
district_rain_avg = df.groupby("District")["Rainfall_mm"].mean().to_dict()

# -----------------------------------------------------
# WEATHER API FETCH (ADVANCED + FALLBACK)
# -----------------------------------------------------
def get_weather(city):
    city = clean_city_name(city)
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        cur = requests.get(url, timeout=8).json()

        if cur is None or cur.get("cod") in ["404", "401", 404, 401]:
            return None

        temp = cur["main"]["temp"]
        humidity = cur["main"]["humidity"]
        rain_now = float(cur.get("rain", {}).get("1h", 0.0))

        # OneCall fallback
        lat = cur.get("coord", {}).get("lat")
        lon = cur.get("coord", {}).get("lon")

        rain_total = rain_now

        if lat and lon:
            try:
                oc = requests.get(
                    f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&appid={WEATHER_API_KEY}&units=metric",
                    timeout=8
                ).json()

                rain24 = 0
                for h in oc.get("hourly", [])[:24]:
                    rain24 += h.get("rain", {}).get("1h", 0)

                if rain24 > 0:
                    rain_total = rain24
            except:
                pass

        # If description says rain but value is missing
        if rain_total == 0 and "rain" in cur["weather"][0]["main"].lower():
            rain_total = 2.0

        # Fallback to dataset rainfall average
        if rain_total == 0:
            avg = district_rain_avg.get(city, 0)
            if avg > 0:
                rain_total = round(avg / 30, 2)

        return {
            "temperature": temp,
            "humidity": humidity,
            "rainfall": round(float(rain_total), 2)
        }

    except:
        return None

# -----------------------------------------------------
# CATEGORY OPTIONS
# -----------------------------------------------------
states = sorted(df["State"].dropna().unique().tolist())
crops = sorted(df["Crop"].dropna().unique().tolist())

state_to_districts = (
    df.groupby("State")["District"].unique().apply(lambda x: sorted(list(x))).to_dict()
)

crop_to_seeds = (
    df.groupby("Crop")["Seed_Variety"].unique().apply(lambda x: sorted(list(x))).to_dict()
)

# -----------------------------------------------------
# INIT SESSION_STATE
# -----------------------------------------------------
if "auto_fill_weather" not in st.session_state:
    st.session_state["auto_fill_weather"] = False

# Apply auto-fill before widgets
if st.session_state["auto_fill_weather"]:

    # For both farmer & scientist
    for key in st.session_state:
        if key.endswith("_auto"):
            base = key.replace("_auto", "")
            st.session_state[base] = st.session_state[key]

    st.session_state["auto_fill_weather"] = False


# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def convert_farmer_inputs(inp):
    LMH = {"Low": 1, "Moderate": 2, "High": 3}
    soil = {"Acidic": 6, "Neutral": 7, "Alkaline": 8}

    return {
        "State": inp["State"],
        "District": inp["District"],
        "Crop": inp["Crop"],
        "Seed_Variety": inp["Seed_Variety"],
        "Rainfall_mm": LMH.get(inp["Rainfall"], 2),
        "Temperature_C": LMH.get(inp["Temperature"], 2),
        "Soil_pH": soil.get(inp["Soil_pH"], 7),
        "Soil_Moisture": LMH.get(inp["Soil_Moisture"], 2),
        "Humidity": LMH.get(inp["Humidity"], 2),
        "Irrigation_Count": LMH.get(inp["Irrigation"], 2),
        "Previous_Yield_kg_ha": float(inp["Previous_Yield_kg_ha"]),
        "Area_Hectares": float(inp["Area_Hectares"]),
    }


def predict_actual(row):
    df_in = pd.DataFrame([row], columns=model_features)
    normalized = model_pipeline.predict(df_in)[0]
    return normalized * crop_means[row["Crop"]]


def recommend_soil(pH, moisture):
    text = []
    if pH < 6: text.append("Apply Lime — Soil acidic.")
    elif pH > 7.5: text.append("Apply Gypsum — Soil alkaline.")
    else: text.append("Soil pH optimal.")

    if moisture <= 1: text.append("Increase irrigation — low moisture.")
    elif moisture == 2: text.append("Moisture ideal.")
    else: text.append("Too much moisture — improve drainage.")
    return text


def suggest_crops(rain, temp, pH, moist):
    out = []
    if rain >= 2 and moist >= 2:
        out += ["Rice", "Sugarcane", "Maize"]
    if rain == 1 and temp >= 2:
        out += ["Cotton", "Millets", "Groundnut"]
    if 6 <= pH <= 7.5:
        out += ["Wheat", "Pulses"]
    return list(set(out))[:5]


# -----------------------------------------------------
# MODE SELECT
# -----------------------------------------------------
mode = st.radio("Choose Mode", ["Farmer Mode", "Scientist Mode"], horizontal=True)

# -----------------------------------------------------
# LOCATION INPUTS
# -----------------------------------------------------
st.subheader("Location & Crop Details")
st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    State = st.selectbox("State", [""] + states)

with c2:
    District = st.selectbox("District", [""] + state_to_districts.get(State, []))

with c3:
    Crop = st.selectbox("Crop", [""] + crops)

Seed_Variety = st.selectbox("Seed Variety", [""] + crop_to_seeds.get(Crop, []))


# -----------------------------------------------------
# FARMER MODE
# -----------------------------------------------------
if mode == "Farmer Mode":
    st.subheader("Farmer Inputs")
    st.markdown("---")

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        Rain = st.selectbox("Rainfall", ["", "Low", "Moderate", "High"], key="Rain")
        Temp = st.selectbox("Temperature", ["", "Low", "Moderate", "High"], key="Temp")
        Moist = st.selectbox("Soil Moisture", ["", "Low", "Moderate", "High"], key="Moist")

    with fc2:
        Hum = st.selectbox("Humidity", ["", "Low", "Moderate", "High"], key="Hum")
        Irr = st.selectbox("Irrigation", ["", "Low", "Moderate", "High"], key="Irr")
        Soil = st.selectbox("Soil pH", ["", "Acidic", "Neutral", "Alkaline"], key="Soil_pH_sel")

    with fc3:
        Prev = st.number_input("Previous Yield (kg/ha)", min_value=0.0, key="Prev")
        Area = st.number_input("Area (Hectares)", min_value=0.0, key="Area")

    # Auto weather (farmer)
    if st.button("Auto Fill Weather"):
        if not District:
            st.error("Select district first.")
        else:
            w = get_weather(District)
            if w:
                st.session_state["Rain_auto"] = "High" if w["rainfall"] >= 10 else "Moderate" if w["rainfall"] >= 3 else "Low"
                st.session_state["Temp_auto"] = "High" if w["temperature"] > 30 else "Moderate" if w["temperature"] > 20 else "Low"
                st.session_state["Hum_auto"]  = "High" if w["humidity"] > 70 else "Moderate" if w["humidity"] > 40 else "Low"

                st.session_state["auto_fill_weather"] = True
                st.success(f"Weather loaded for {District}")
                st.rerun()
            else:
                st.error("Weather fetch failed.")

    raw = {
        "State": State, "District": District, "Crop": Crop,
        "Seed_Variety": Seed_Variety, "Rainfall": Rain, "Temperature": Temp,
        "Soil_Moisture": Moist, "Humidity": Hum, "Irrigation": Irr,
        "Soil_pH": Soil, "Previous_Yield_kg_ha": Prev, "Area_Hectares": Area
    }


# -----------------------------------------------------
# SCIENTIST MODE
# -----------------------------------------------------
else:
    st.subheader("Scientist Inputs")
    st.markdown("---")

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        RainSci = st.number_input("Rainfall (mm)", min_value=0.0, key="RainSci")
        TempSci = st.number_input("Temperature (°C)", min_value=0.0, key="TempSci")
        SoilSci = st.number_input("Soil pH", min_value=0.0, key="SoilSci")

    with sc2:
        MoistSci = st.number_input("Soil Moisture", min_value=0.0, key="MoistSci")
        HumSci = st.number_input("Humidity (%)", min_value=0.0, key="HumSci")
        IrrSci = st.number_input("Irrigation Count", min_value=0, key="IrrSci")

    with sc3:
        PrevSci = st.number_input("Previous Yield (kg/ha)", min_value=0.0, key="PrevSci")
        AreaSci = st.number_input("Area (Hectares)", min_value=0.0, key="AreaSci")

    # Auto weather (scientist)
    if st.button("Auto Fill Weather (Scientist)"):

        if not District:
            st.error("Select district first.")
        else:
            w = get_weather(District)
            if w:
                st.session_state["RainSci_auto"] = w["rainfall"]
                st.session_state["TempSci_auto"] = w["temperature"]
                st.session_state["HumSci_auto"] = w["humidity"]

                st.session_state["auto_fill_weather"] = True
                st.success(f"Weather loaded for {District}")
                st.rerun()
            else:
                st.error("Weather fetch failed.")

    raw = {
        "State": State, "District": District, "Crop": Crop,
        "Seed_Variety": Seed_Variety, "Rainfall_mm": RainSci,
        "Temperature_C": TempSci, "Soil_pH": SoilSci,
        "Soil_Moisture": MoistSci, "Humidity": HumSci,
        "Irrigation_Count": IrrSci, "Previous_Yield_kg_ha": PrevSci,
        "Area_Hectares": AreaSci
    }

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
st.subheader("Prediction")
st.markdown("---")

if st.button("Predict Yield"):

    missing = []
    if not raw["State"]: missing.append("State")
    if not raw["District"]: missing.append("District")
    if not raw["Crop"]: missing.append("Crop")
    if not raw["Seed_Variety"]: missing.append("Seed Variety")
    if raw["Previous_Yield_kg_ha"] <= 0: missing.append("Previous Yield > 0 required")
    if raw["Area_Hectares"] <= 0: missing.append("Area > 0 required")

    if missing:
        st.error("Fix the following:\n- " + "\n- ".join(missing))
        st.stop()

    try:
        prepared = convert_farmer_inputs(raw) if mode == "Farmer Mode" else raw
        pred = predict_actual(prepared)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Realism blend
    pred = 0.7 * pred + 0.3 * prepared["Previous_Yield_kg_ha"]
    total_yield = pred * prepared["Area_Hectares"]

    st.success(f"Yield per Hectare: {pred:.2f} kg/ha")
    st.success(f"Total Yield (Area × Yield): {total_yield:.2f} kg")
    st.balloons()

    # Soil Recommendations
    st.subheader("Soil Recommendations")
    for rec in recommend_soil(prepared["Soil_pH"], prepared["Soil_Moisture"]):
        st.markdown(f'<div class="reco-card">{rec}</div>', unsafe_allow_html=True)

    # Crop Suggestions
    st.subheader("Crop Suggestions")

    try:
        rain_s = int(prepared.get("Rainfall_mm", prepared.get("Rainfall", 1)))
        temp_s = int(prepared.get("Temperature_C", prepared.get("Temperature", 2)))
        moist_s = int(prepared.get("Soil_Moisture", 2))
    except:
        rain_s, temp_s, moist_s = 1, 2, 2

    for c in suggest_crops(rain_s, temp_s, prepared["Soil_pH"], moist_s):
        st.markdown(f"<div class='reco-card'>🌱 {c}</div>", unsafe_allow_html=True)

# Footer
st.caption("Crop Yield Predictor")

