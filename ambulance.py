import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="Ambulance Dispatch Optimization Dashboard",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Caching and Generation ---
@st.cache_data
def load_data():
    """Generates realistic mock data constrained to the Tijuana, Mexico area."""
    lat_min, lat_max = 32.40, 32.55
    lon_min, lon_max = -117.12, -116.60
    
    num_calls = 500
    np.random.seed(42)
    latitudes = np.random.uniform(lat_min, lat_max, num_calls)
    longitudes = np.random.uniform(lon_min, lon_max, num_calls)
    
    api_time = np.random.uniform(5, 30, num_calls)
    real_time = api_time * np.random.uniform(0.6, 0.95, num_calls)
    corrected_time = real_time * np.random.uniform(0.95, 1.05, num_calls)
    
    calls_df = pd.DataFrame({
        'lat': latitudes,
        'lon': longitudes,
        'api_time_minutes': api_time,
        'real_time_minutes': real_time,
        'corrected_time_minutes': corrected_time
    })
    
    current_bases = pd.DataFrame({
        'name': ['Current Base - Centro', 'Current Base - La Mesa', 'Current Base - Otay', 'Current Base - El Florido'],
        'lat': [32.533, 32.515, 32.528, 32.463],
        'lon': [-117.03, -116.98, -116.94, -116.82],
        'type': ['Current'] * 4
    })
    
    num_optimized = 12
    optimized_bases = pd.DataFrame({
        'name': [f'Optimized Station {i+1}' for i in range(num_optimized)],
        'lat': np.random.uniform(lat_min, lat_max, num_optimized),
        'lon': np.random.uniform(lon_min, lon_max, num_optimized),
        'type': ['Optimized'] * num_optimized
    })

    return calls_df, current_bases, optimized_bases

# Load the data
calls_df, current_bases, optimized_bases = load_data()


# --- Sidebar Navigation ---
st.sidebar.title("🚑 Navigation")
st.sidebar.markdown("""
This dashboard presents the key findings of the PhD thesis:
**"Sistema de despacho para ambulancias de la ciudad de Tijuana"**
by M.C. Noelia Araceli Torres Cortés.
""")

page = st.sidebar.radio("Go to:", 
    ["Thesis Overview", "Data & Time Correction", "Demand Clustering", "Location Optimization"]
)

st.sidebar.info("Data is simulated for demonstration purposes, reflecting the concepts and geography of the original research.")


# --- Page Rendering ---

if page == "Thesis Overview":
    st.title("Sistema de Despacho para Ambulancias de la Ciudad de Tijuana")
    st.subheader("PhD Thesis Dashboard by M.C. Noelia Araceli Torres Cortés")
    st.markdown("...") # Content is fine, truncated for brevity

elif page == "Data & Time Correction":
    st.title("Data Exploration & Travel Time Correction")
    st.markdown("...") # Content is fine, truncated for brevity

elif page == "Demand Clustering":
    st.title("Identifying Demand Hotspots via Clustering")
    st.markdown("To determine where to place ambulances, the historical emergency calls were grouped using K-Means clustering. The center of each cluster represents a 'demand point' or hotspot.")

    k = st.slider("Select Number of Demand Clusters (k):", min_value=5, max_value=25, value=15, step=1)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    calls_df['cluster'] = kmeans.fit_predict(calls_df[['lat', 'lon']])
    centroids = kmeans.cluster_centers_
    centroids_df = pd.DataFrame(centroids, columns=['lat', 'lon'])
    
    st.subheader(f"Map of {k} Emergency Call Clusters")
    
    # --- FIX #1: Use px.scatter_map instead of scatter_mapbox ---
    fig = px.scatter_map(
        calls_df,
        lat="lat",
        lon="lon",
        color="cluster",
        # mapbox_style is not needed; it uses OpenStreetMap by default
        zoom=10,
        height=600,
        title="Emergency Calls Color-Coded by Cluster"
    )
    
    fig.add_scattermapbox(
        lat=centroids_df['lat'],
        lon=centroids_df['lon'],
        mode='markers',
        marker=dict(size=15, symbol='star', color='red'),
        name='Demand Hotspot'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("The red stars ★ represent the calculated demand hotspots, which are the inputs for the location optimization model.")

elif page == "Location Optimization":
    st.title("Ambulance Location Optimization")
    st.markdown("Using the demand hotspots and corrected travel times, the Robust Double Standard Model (RDSM) was used to find the optimal locations for ambulances to maximize coverage across the city.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimization Results")
        st.write("...") # Content is fine, truncated for brevity
        st.metric(label="...", value="83.9%")
        st.metric(label="...", value="100%", delta="16.1%")
        st.info("The map on the right shows the optimized vs. current base locations.")

    with col2:
        st.subheader("Optimized vs. Current Ambulance Locations")
        all_bases = pd.concat([current_bases, optimized_bases], ignore_index=True)

        # --- FIX #2: Use px.scatter_map and remove the unsupported 'symbol' argument ---
        fig = px.scatter_map(
            all_bases,
            lat="lat",
            lon="lon",
            color="type",
            size_max=15, # Use size to differentiate if needed, as symbol is not supported
            zoom=10,
            height=600,
            title="Comparison of Ambulance Base Locations",
            hover_name="name",
            color_discrete_map={
                "Current": "orange",
                "Optimized": "green"
            },
        )
        
        # Manually set the symbols if desired using go.Scattermapbox
        # For simplicity here, we rely on color and hover data.
        fig.update_layout(legend_title_text='Base Type')
        
        st.plotly_chart(fig, use_container_width=True)
