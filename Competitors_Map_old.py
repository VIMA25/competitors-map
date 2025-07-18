import pandas as pd
import folium
from folium import plugins
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime
import requests
from typing import Optional, Dict, Any
import streamlit.components.v1
import logging
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAPBOX_TOKEN = "pk.eyJ1IjoianVhbnY5NSIsImEiOiJjbWN5Z3I5dHUwbjg1MmpzYW1nY3JjM3hoIn0.HHuDkkuSh3iIXCurZ5DGPg"
DEFAULT_CENTER = [27.7663, -82.6404]  # Florida center
DEFAULT_ZOOM = 7

# Enhanced styling constants
COLORS = {
    'primary': '#4facfe',
    'secondary': '#00f2fe',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

TYPE_COLORS = {
    'Lab': '#e74c3c',
    'Satellite': '#3498db',
    'Clinical Laboratory': '#f39c12',
    'IVF Center': '#9b59b6',
    'Fertility Clinic': '#2ecc71'
}

def find_csv_file():
    """Find the CSV file with flexible naming"""
    # Possible file names to look for
    possible_names = [
        'Competitors SART-CDC-AHCA.csv',
        'Competitors_SART-CDC-AHCA.csv',
        'competitors sart-cdc-ahca.csv',
        'Competitors SART-C*HCA.csv'  # wildcard pattern
    ]
    
    # Get current working directory
    current_dir = os.getcwd()
    st.sidebar.write(f"Current directory: {current_dir}")
    
    # List all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    st.sidebar.write(f"Available CSV files: {csv_files}")
    
    # First try exact matches
    for name in possible_names[:-1]:  # exclude wildcard
        if os.path.exists(name):
            return name
    
    # Then try wildcard pattern
    wildcard_matches = glob.glob('Competitors*SART*HCA*.csv')
    if wildcard_matches:
        return wildcard_matches[0]
    
    # If no exact match, look for any CSV with "Competitors" in the name
    competitor_files = [f for f in csv_files if 'competitors' in f.lower() or 'competitor' in f.lower()]
    if competitor_files:
        return competitor_files[0]
    
    # If still no match, show available files and let user choose
    if csv_files:
        st.error(f"Could not find the expected CSV file. Available CSV files: {csv_files}")
        st.info("Please rename your CSV file to 'Competitors SART-CDC-AHCA.csv' or update the code with the correct filename.")
        return csv_files[0]  # Return the first CSV file found
    
    return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data() -> pd.DataFrame:
    """Load and process the fertility clinic data with enhanced error handling"""
    try:
        # Find the CSV file
        csv_file = find_csv_file()
        
        if csv_file is None:
            st.error("No CSV file found in the current directory.")
            return pd.DataFrame()
        
        st.info(f"Loading data from: {csv_file}")
        
        # Try to read the CSV file
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully loaded {len(df)} records from {csv_file}")
        
        # Show column names for debugging
        st.sidebar.write("Available columns:", list(df.columns))
        
        # Try to auto-detect column names
        df = auto_detect_columns(df)
        
        # Data cleaning and validation
        df = clean_and_validate_data(df)

        return df
    except FileNotFoundError:
        st.error(f"CSV file not found. Please ensure the file is in the correct location: {os.getcwd()}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect and standardize column names"""
    column_mapping = {}
    
    # Common variations for each required column
    name_variations = ['name', 'clinic_name', 'facility_name', 'organization', 'business_name']
    address_variations = ['address', 'street', 'street_address', 'location', 'addr']
    city_variations = ['city', 'town', 'municipality']
    state_variations = ['state', 'province', 'region']
    lat_variations = ['latitude', 'lat', 'y', 'coord_lat']
    lon_variations = ['longitude', 'lon', 'lng', 'long', 'x', 'coord_lon']
    type_variations = ['type', 'category', 'classification', 'facility_type']
    
    # Auto-detect columns
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if any(var in col_lower for var in name_variations):
            column_mapping['Name'] = col
        elif any(var in col_lower for var in address_variations):
            column_mapping['Address'] = col
        elif any(var in col_lower for var in city_variations):
            column_mapping['City'] = col
        elif any(var in col_lower for var in state_variations):
            column_mapping['State'] = col
        elif any(var in col_lower for var in lat_variations):
            column_mapping['Latitude'] = col
        elif any(var in col_lower for var in lon_variations):
            column_mapping['Longitude'] = col
        elif any(var in col_lower for var in type_variations):
            column_mapping['Type'] = col
    
    # Show detected mappings
    st.sidebar.write("Column mappings detected:", column_mapping)
    
    # Rename columns
    df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Add missing columns with defaults
    required_columns = ['Name', 'Address', 'City', 'State', 'Latitude', 'Longitude', 'Type']
    for col in required_columns:
        if col not in df_renamed.columns:
            if col == 'Type':
                df_renamed[col] = 'Clinic'
            else:
                df_renamed[col] = 'Unknown'
    
    return df_renamed

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the loaded data"""
    try:
        # Show initial data info
        st.sidebar.write(f"Initial data shape: {df.shape}")
        
        # Remove rows with missing essential coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])
        st.sidebar.write(f"After removing missing coordinates: {df.shape}")
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['Latitude', 'Longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid coordinates
        df = df[(df['Latitude'].between(-90, 90)) & (df['Longitude'].between(-180, 180))]
        st.sidebar.write(f"After coordinate validation: {df.shape}")
        
        # Clean string columns
        string_columns = ['Name', 'Address', 'City', 'State', 'Type']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        logger.info(f"Data cleaned. {len(df)} records remaining")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        return df

def create_enhanced_map(df: pd.DataFrame, selected_types: list = None) -> folium.Map:
    """Create an enhanced interactive map with fertility clinics"""
    try:
        # Filter data based on selected types
        if selected_types:
            df_filtered = df[df['Type'].isin(selected_types)]
        else:
            df_filtered = df
        
        if df_filtered.empty:
            st.warning("No data to display on map")
            return None
        
        # Calculate center point
        center_lat = df_filtered['Latitude'].mean()
        center_lon = df_filtered['Longitude'].mean()
        
        # Create the map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=DEFAULT_ZOOM,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Create marker clusters
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Add markers for each clinic
        for idx, row in df_filtered.iterrows():
            # Get color based on type
            color = TYPE_COLORS.get(row['Type'], '#333333')
            
            # Create popup content
            popup_content = f"""
            <div style="width: 250px;">
                <h4 style="color: {color}; margin-bottom: 10px;">{row['Name']}</h4>
                <p><strong>Type:</strong> {row['Type']}</p>
                <p><strong>Address:</strong> {row['Address']}</p>
                <p><strong>City:</strong> {row['City']}, {row['State']}</p>
                <p><strong>Coordinates:</strong> {row['Latitude']:.4f}, {row['Longitude']:.4f}</p>
            </div>
            """
            
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=row['Name'],
                icon=folium.Icon(color='blue' if row['Type'] == 'IVF Center' else 'green', 
                               icon='info-sign')
            ).add_to(marker_cluster)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        logger.error(f"Error creating map: {str(e)}")
        return None

def create_charts(df: pd.DataFrame) -> None:
    """Create interactive charts for data analysis"""
    try:
        st.subheader("📊 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Type distribution
            type_counts = df['Type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution by Type",
                color_discrete_map=TYPE_COLORS
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # State distribution
            state_counts = df['State'].value_counts().head(10)
            fig_bar = px.bar(
                x=state_counts.index,
                y=state_counts.values,
                title="Top 10 States by Clinic Count",
                labels={'x': 'State', 'y': 'Count'},
                color=state_counts.values,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # City analysis
        st.subheader("🏙️ City Analysis")
        city_counts = df['City'].value_counts().head(15)
        fig_city = px.bar(
            x=city_counts.values,
            y=city_counts.index,
            orientation='h',
            title="Top 15 Cities by Clinic Count",
            labels={'x': 'Count', 'y': 'City'},
            color=city_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_city.update_layout(height=600)
        st.plotly_chart(fig_city, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        logger.error(f"Error creating charts: {str(e)}")

def create_sidebar_filters(df: pd.DataFrame) -> dict:
    """Create sidebar filters and return filter values"""
    st.sidebar.header("🔧 Filters")
    
    filters = {}
    
    # Type filter
    available_types = sorted(df['Type'].unique())
    filters['types'] = st.sidebar.multiselect(
        "Select Clinic Types",
        available_types,
        default=available_types
    )
    
    # State filter
    available_states = sorted(df['State'].unique())
    filters['states'] = st.sidebar.multiselect(
        "Select States",
        available_states,
        default=available_states
    )
    
    # City filter
    available_cities = sorted(df['City'].unique())
    filters['cities'] = st.sidebar.multiselect(
        "Select Cities",
        available_cities,
        default=available_cities
    )
    
    return filters

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    if filters['types']:
        filtered_df = filtered_df[filtered_df['Type'].isin(filters['types'])]
    
    if filters['states']:
        filtered_df = filtered_df[filtered_df['State'].isin(filters['states'])]
    
    if filters['cities']:
        filtered_df = filtered_df[filtered_df['City'].isin(filters['cities'])]
    
    return filtered_df

def create_streamlit_app():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Fertility Clinic Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4facfe;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4facfe;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Fertility Clinic Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.stop()
    
    # Create sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    df_filtered = apply_filters(df, filters)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clinics", len(df_filtered))
    
    with col2:
        st.metric("States", df_filtered['State'].nunique())
    
    with col3:
        st.metric("Cities", df_filtered['City'].nunique())
    
    with col4:
        st.metric("Types", df_filtered['Type'].nunique())
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📊 Analytics", "📋 Data Table", "📥 Export"])
    
    with tab1:
        st.subheader("🗺️ Interactive Map")
        
        if not df_filtered.empty:
            # Create and display map
            map_obj = create_enhanced_map(df_filtered, filters['types'])
            if map_obj:
                # Display folium map using st.components.v1.html
                map_html = map_obj._repr_html_()
                st.components.v1.html(map_html, height=600)
            else:
                st.error("Could not create map")
        else:
            st.warning("No data to display with current filters")
    
    with tab2:
        if not df_filtered.empty:
            create_charts(df_filtered)
        else:
            st.warning("No data to display with current filters")
    
    with tab3:
        st.subheader("📋 Data Table")
        
        if not df_filtered.empty:
            # Display data table with search
            st.dataframe(df_filtered, use_container_width=True, height=400)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Records by Type:**")
                st.write(df_filtered['Type'].value_counts())
            
            with col2:
                st.write("**Records by State:**")
                st.write(df_filtered['State'].value_counts().head(10))
        else:
            st.warning("No data to display with current filters")
    
    with tab4:
        st.subheader("📥 Export Data")
        
        if not df_filtered.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = df_filtered.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f'fertility_clinics_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
            
            with col2:
                # JSON export
                json_data = df_filtered.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_data,
                    file_name=f'fertility_clinics_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json'
                )
        else:
            st.warning("No data to export with current filters")

def main():
    """Main application entry point"""
    try:
        create_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
