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
    """
    Generates realistic mock data constrained to the Tijuana, Mexico area.
    """
    # Bounding Box for Tijuana, Mexico (approximates the municipality)
    lat_min, lat_max = 32.40, 32.55
    lon_min, lon_max = -117.12, -116.60
    
    # Generate 500 mock emergency calls within the bounding box
    num_calls = 500
    np.random.seed(42)
    latitudes = np.random.uniform(lat_min, lat_max, num_calls)
    longitudes = np.random.uniform(lon_min, lon_max, num_calls)
    
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
    
    # Mock Ambulance Bases - Placed within Tijuana
    current_bases = pd.DataFrame({
        'name': ['Current Base - Centro', 'Current Base - La Mesa', 'Current Base - Otay', 'Current Base - El Florido'],
        'lat': [32.533, 32.515, 32.528, 32.463],
        'lon': [-117.03, -116.98, -116.94, -116.82],
        'type': ['Current'] * 4
    })
    
    # Optimized bases are more spread out based on demand
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
    st.markdown("""
    The research followed a comprehensive, multi-stage methodology to move from raw data to an actionable solution. Each step built upon the last, ensuring the final recommendations were robust and data-driven:
    
    1.  **Data Analysis & Filtering:** The process began by collecting and cleaning historical emergency call records (FRAP) and ambulance GPS logs. This crucial first step involved handling missing data, filtering inconsistencies, and creating a unified, reliable dataset.
    
    2.  **Travel Time Correction:** A machine learning model (Random Forest) was developed to predict the error between standard API travel times (OSRM) and actual ambulance travel times. This correction is the core novelty, making all subsequent calculations more realistic.
    
    3.  **Demand Clustering:** Historical emergency call locations were grouped using the K-Means clustering algorithm. The center of each cluster was identified as a "demand point," representing a statistical hotspot for incidents.
    
    4.  **Location Optimization:** Using the demand points and corrected travel times as inputs, an optimization model (Robust Double Standard Model - RDSM) was executed. This model determined the optimal strategic locations for ambulances to be stationed throughout the day to maximize the probability of covering any incident with at least two units within a critical time window.
    
    5.  **Web-Based Tool Design:** Finally, the entire pipeline was integrated into the design of a web-based decision-support tool, allowing dispatchers to interact with the findings and run simulations, as demonstrated by this dashboard.
    """)

elif page == "Data & Time Correction":
    st.title("Data Exploration & Travel Time Correction")
    st.markdown("""
    A foundational challenge in optimizing ambulance dispatch is accurately predicting how long it will take for an ambulance to reach an incident. Standard routing APIs, like Google Maps or OSRM, are designed for civilian vehicles and do not account for the unique operational advantages of an emergency vehicle. This section visualizes this critical discrepancy and the effectiveness of the thesis's proposed correction model.
    """)

    st.subheader("Map of Simulated Emergency Calls in Tijuana")
    st.map(calls_df[['lat', 'lon']], zoom=11, use_container_width=True)
    
    st.subheader("Correcting Travel Time Estimations")
    
    # Calculate errors
    error_before = calls_df['api_time_minutes'] - calls_df['real_time_minutes']
    error_after = calls_df['corrected_time_minutes'] - calls_df['real_time_minutes']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Correction** (API vs. Real Time)")
        fig1 = px.histogram(error_before, nbins=50, title="Error Distribution (API - Real)")
        fig1.update_layout(showlegend=False, yaxis_title="Frequency", xaxis_title="Time Error (minutes)")
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""
        This chart shows the error distribution when comparing the travel time estimated by a standard API to the actual time it took an ambulance to travel, based on GPS data. The vast majority of the bars are on the positive side of zero, indicating that the **API consistently overestimates the travel time**. This "optimism gap" occurs because the API calculates routes for regular traffic, while an ambulance with sirens can often bypass congestion, take more direct routes, and exceed normal speed limits. Relying on these uncorrected, overly pessimistic estimates leads to suboptimal placement of ambulances, as the system believes it takes longer to cover distances than it actually does.
        """)

    with col2:
        st.markdown("**After Correction** (ML Model vs. Real Time)")
        fig2 = px.histogram(error_after, nbins=50, title="Error Distribution (Corrected - Real)")
        fig2.update_layout(showlegend=False, yaxis_title="Frequency", xaxis_title="Time Error (minutes)")
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""
        This chart displays the error distribution after applying the machine learning correction model developed in the thesis. The model was trained on historical data to predict the *discrepancy* between the API and reality. As shown, the error distribution is now tightly centered around zero. This demonstrates that the model successfully learns the unique travel characteristics of Tijuana's ambulances, producing far more accurate and reliable travel time predictions. This accuracy is the cornerstone of the subsequent location optimization, enabling the system to place ambulances more effectively and ultimately improve city-wide coverage by over 20%.
        """)

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
    
    # Use the modern px.scatter_map function
    fig = px.scatter_map(
        calls_df,
        lat="lat",
        lon="lon",
        color="cluster",
        zoom=10,
        height=600,
        title="Emergency Calls Color-Coded by Cluster"
    )
    
    # Add a separate trace for the centroids to style them differently
    fig.add_scattermapbox(
        lat=centroids_df['lat'],
        lon=centroids_df['lon'],
        mode='markers',
        marker=dict(size=15, symbol='star', color='red'),
        name='Demand Hotspot',
        hoverinfo='text',
        text=[f'Hotspot {i+1}' for i in range(len(centroids_df))]
    )
    
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
        all_bases = pd.concat([current_bases, optimized_bases], ignore_index=True)

        # Use the modern px.scatter_map function
        fig = px.scatter_map(
            all_bases,
            lat="lat",
            lon="lon",
            color="type",
            size_max=15,
            zoom=10,
            height=600,
            title="Comparison of Ambulance Base Locations",
            hover_name="name",
            color_discrete_map={
                "Current": "orange",
                "Optimized": "green"
            }
        )
        
        fig.update_layout(legend_title_text='Base Type')
        
        st.plotly_chart(fig, use_container_width=True)
