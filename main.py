import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(page_title="Freightizer AI", page_icon="🚚", layout="wide", initial_sidebar_state="expanded")

# --- 2. CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
    <style>
    /* Custom styling for the final price card */
    .price-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .price-card h2 {
        color: #1f77b4;
        margin-bottom: 0px;
    }
    .price-card p {
        color: #555555;
        font-size: 14px;
    }
    /* Dark mode support for the card */
    @media (prefers-color-scheme: dark) {
        .price-card {
            background-color: #1e1e2e;
            box-shadow: 0 4px 6px rgba(255, 255, 255, 0.05);
        }
        .price-card p {
            color: #aaaaaa;
        }
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #888888;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD THE SAVED ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('Models/freight_ensemble_model.pkl')
    encoder = joblib.load('Models/freight_target_encoder.pkl')
    return model, encoder

model, encoder = load_assets()

# --- 4. SIDEBAR (Secondary Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2766/2766194.png", width=60)
    st.title("Settings")
    st.markdown("Adjust temporal parameters to see how seasonality affects pricing.")
    
    st.divider()
    
    month = st.slider("📅 Shipment Month", 1, 12, 6, help="1 = January, 12 = December. Freight rates often spike in Q4 (Oct-Dec) due to holiday shipping.")
    day_of_week = st.slider("📆 Day of Week", 0, 6, 2, help="0 = Monday, 6 = Sunday. Weekend dispatch can sometimes incur premium fees.")
    
    st.divider()
    st.caption("Powered by XGBoost, LightGBM, and CatBoost Ensemble.")

# --- 5. MAIN PAGE UI ---
st.title("🚚 Freightizer AI Pricing Engine")
st.markdown("Enter your logistics parameters below to generate a highly optimized, real-time B2B shipping quote.")
st.write("") 

with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📍 Route Details")
        origin = st.selectbox("Origin Warehouse", 
                              ["Warehouse_NYC", "Warehouse_LA", "Warehouse_CHI", "Warehouse_ATL", "Warehouse_BOS", "Warehouse_SEA", "Warehouse_SF", "Warehouse_DEN", "Warehouse_MIA"], 
                              index=None, placeholder="Select an origin...")
        
        destination = st.selectbox("Destination City", 
                                   ["San Francisco", "Dallas", "Miami", "Boston", "Denver", "Chicago", "Phoenix", "Portland", "Atlanta"], 
                                   index=None, placeholder="Select a destination...")
        
        carrier = st.selectbox("Carrier Partner", 
                               ["FedEx", "UPS", "DHL", "LaserShip", "Amazon Logistics", "OnTrac", "USPS"], 
                               index=None, placeholder="Select preferred carrier...",
                               help="Different carriers have vastly different base rates and regional dominance.")

    with col2:
        st.subheader("📦 Cargo & Macro Data")
        weight = st.number_input("Total Weight (kg)", min_value=1.0, max_value=6000.0, value=None, placeholder="e.g., 250", help="Maximum allowable weight is 6,000 kg per pallet.")
        distance = st.number_input("Travel Distance (miles)", min_value=10.0, max_value=3000.0, value=None, placeholder="e.g., 1200", help="Total transit distance. Longer distances increase the Kg-Mile multiplier.")
        fuel_price = st.number_input("Current Diesel Price ($/gal)", min_value=2.00, max_value=7.00, value=None, placeholder="e.g., 4.10", help="The U.S. National Average Diesel price. Crucial for calculating fuel surcharges.")

st.write("") 
st.write("") 

# --- 6. THE PREDICTION ENGINE ---
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    if st.button("Generate AI Quote 🚀", use_container_width=True, type="primary"):
        
        if None in [weight, distance, fuel_price, origin, destination, carrier]:
            st.warning("⚠️ Action Required: Please fill out all Route and Cargo fields before generating a quote.", icon="🛑")
            
        else:
            with st.status("Analyzing routing nodes and macroeconomic data...", expanded=True) as status:
                st.write("Applying target encoding to geographic regions...")
                time.sleep(0.5)
                st.write("Calculating Kg-Mile interaction multipliers...")
                
                input_data = pd.DataFrame({
                    'Origin_Warehouse': [origin],
                    'Destination': [destination],
                    'Carrier': [carrier],
                    'Weight_kg': [weight],
                    'Distance_miles': [distance],
                    'Fuel_Price': [fuel_price],
                    'Shipment_Month': [month],
                    'Shipment_DayOfWeek': [day_of_week],
                    'Kg_Mile_Interaction': [weight * distance]
                })
                
                encoded_input = encoder.transform(input_data)
                log_prediction = model.predict(encoded_input)[0]
                final_price = np.expm1(log_prediction)
                
                time.sleep(0.5)
                status.update(label="Quote generated successfully!", state="complete", expanded=False)
            
            st.markdown(f"""
                <div class="price-card">
                    <p>OPTIMIZED FREIGHT RATE</p>
                    <h2 style="font-size: 3rem;">${final_price:,.2f}</h2>
                    <p>Includes ${fuel_price}/gal fuel surcharge for a {distance:,.0f} mile transit via {carrier}.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.balloons() 

# --- 7. FOOTER / EXPLANATION ---
st.divider()
with st.expander("ℹ️ How does this AI work?"):
    st.write("""
        This pricing engine is powered by a **Machine Learning Ensemble** consisting of XGBoost, LightGBM, and CatBoost. 
        It mathematically models historical pricing inefficiencies, carrier-specific premium rates, and the impact of 
        macroeconomic variables like the U.S. National Average Diesel Price.
        
        *Model Performance: 6.78% MAPE on blind test data.*
    """)

# --- 8. PERSONAL BRANDING FOOTER ---
st.markdown('<div class="footer">Created by Subham Mohanty</div>', unsafe_allow_html=True)