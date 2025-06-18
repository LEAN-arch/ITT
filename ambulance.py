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
    """Generates realistic mock data for demonstration."""
    # Central point for Tijuana
    lat_center, lon_center = 32.5149, -117.0382
    
    # Generate 500 mock emergency calls around the central point
    num_calls = 500
    np.random.seed(42)
    latitudes = lat_center + np.random.randn(num_calls) * 0.05
    longitudes = lon_center + np.random.randn(num_calls) * 0.05
    
    # Simulate travel times
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
    
    # Mock Ambulance Bases
    current_bases = pd.DataFrame({
        'name': ['Current Base 1', 'Current Base 2', 'Current Base 3', 'Current Base 4'],
        'lat': [32.533, 32.501, 32.48, 32.52],
        'lon': [-117.01, -117.04, -116.95, -116.98],
        'type': ['Current'] * 4  # <-- FIX #1: Length must be 4
    })
    
    optimized_bases = pd.DataFrame({
        'name': [f'Optimized Station {i+1}' for i in range(12)],
        'lat': lat_center + np.random.randn(12) * 0.06,
        'lon': lon_center + np.random.randn(12) * 0.06,
        'type': ['Optimized'] * 12 # <-- FIX #2: Length must be 12
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

    st.markdown("""
    This dashboard provides an interactive summary of the doctoral research aimed at optimizing the Emergency Medical Services (EMS) for the Red Cross in Tijuana, Mexico. The project addresses the critical challenge of reducing ambulance response times in a city with limited resources and complex urban conditions.
    """)
    
    st.header("Core Contribution & Novelty")
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Travel Time Correction Model**")
        st.write("""
        The primary innovation is a machine learning model that corrects travel time estimations from standard APIs (like OSRM). It learns the discrepancy between API predictions and actual ambulance travel times, accounting for factors like siren usage and traffic law exemptions. This resulted in a **20% improvement in location coverage**.
        """)

    with col2:
        st.info("🌐 **Real-World Application**")
        st.write("""
        Unlike studies in well-structured cities, this research tackles the 'messy' reality of a developing region. By creating a practical, data-driven solution for the Tijuana Red Cross, it bridges the gap between academic theory and on-the-ground impact. The final model uses OSRM, a free, open-source tool, making it sustainable for the organization.
        """)
        
    st.header("Methodology Pipeline")
    st.image("https://i.imgur.com/L1iF7hQ.png", caption="The research followed a comprehensive pipeline from data analysis to optimization and web tool design.")


elif page == "Data & Time Correction":
    st.title("Data Exploration & Travel Time Correction")
    st.markdown("A key finding was the significant discrepancy between API-estimated travel times and the actual travel times recorded by ambulance GPS. This section visualizes this gap and the improvement made by the correction model.")

    st.subheader("Map of Simulated Emergency Calls in Tijuana")
    st.map(calls_df[['lat', 'lon']], zoom=11)
    
    st.subheader("Correcting Travel Time Estimations")
    
    # Calculate errors
    error_before = calls_df['api_time_minutes'] - calls_df['real_time_minutes']
    error_after = calls_df['corrected_time_minutes'] - calls_df['real_time_minutes']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Correction** (API vs. Real Time)")
        fig1 = px.histogram(error_before, nbins=50, title="Error Distribution (API - Real)")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.write("The standard API consistently overestimates travel time (positive error), as it doesn't account for an ambulance's ability to bypass traffic.")

    with col2:
        st.markdown("**After Correction** (ML Model vs. Real Time)")
        fig2 = px.histogram(error_after, nbins=50, title="Error Distribution (Corrected - Real)")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.write("The machine learning correction model produces estimates much closer to the real travel time, with the error centered around zero.")

elif page == "Demand Clustering":
    st.title("Identifying Demand Hotspots via Clustering")
    st.markdown("To determine where to place ambulances, the historical emergency calls were grouped using K-Means clustering. The center of each cluster represents a 'demand point' or hotspot.")

    k = st.slider("Select Number of Demand Clusters (k):", min_value=5, max_value=25, value=15, step=1)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    calls_df['cluster'] = kmeans.fit_predict(calls_df[['lat', 'lon']])
    centroids = kmeans.cluster_centers_
    centroids_df = pd.DataFrame(centroids, columns=['lat', 'lon'])
    
    st.subheader(f"Map of {k} Emergency Call Clusters")
    
    fig = px.scatter_mapbox(
        calls_df,
        lat="lat",
        lon="lon",
        color="cluster",
        mapbox_style="carto-positron",
        zoom=10,
        height=600,
        title="Emergency Calls Color-Coded by Cluster"
    )
    
    # Use a separate trace for centroids so we can customize them
    centroid_trace = px.scatter_mapbox(
        centroids_df,
        lat="lat",
        lon="lon"
    ).data[0]

    # Customize the centroid markers
    centroid_trace.marker = {'size': 15, 'symbol': 'star', 'color': 'red'}
    centroid_trace.name = 'Demand Hotspot'
    
    fig.add_trace(centroid_trace)
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("The red stars ★ represent the calculated demand hotspots, which are the inputs for the location optimization model.")

elif page == "Location Optimization":
    st.title("Ambulance Location Optimization")
    st.markdown("Using the demand hotspots and corrected travel times, the Robust Double Standard Model (RDSM) was used to find the optimal locations for ambulances to maximize coverage across the city.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimization Results")
        st.write("The model significantly improved service coverage, especially after applying the travel time correction.")
        
        st.metric(
            label="Double Coverage (Before Correction)", 
            value="83.9%", 
            help="Percentage of demand serviceable by at least two ambulances within the time threshold using standard API times."
        )
        st.metric(
            label="Double Coverage (After Correction)", 
            value="100%", 
            delta="16.1%",
            help="Coverage achieved using the ML-corrected travel times. The improvement is substantial."
        )
        st.info("The map on the right shows the optimized vs. current base locations.")

    with col2:
        st.subheader("Optimized vs. Current Ambulance Locations")
        # Combine bases for plotting
        all_bases = pd.concat([current_bases, optimized_bases], ignore_index=True)
        
        fig = px.scatter_mapbox(
            all_bases,
            lat="lat",
            lon="lon",
            color="type",
            symbol="type",
            mapbox_style="carto-positron",
            zoom=10,
            height=600,
            title="Comparison of Ambulance Base Locations",
            hover_name="name",
            color_discrete_map={"Current": "orange", "Optimized": "green"},
            hover_data={"type": True, "lat": False, "lon": False}
        )
        fig.update_layout(legend_title_text='Base Type')
        st.plotly_chart(fig, use_container_width=True)
