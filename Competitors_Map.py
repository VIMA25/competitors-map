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
from typing import Optional, Dict, Any, List
import streamlit.components.v1 as components
import logging
import os
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Constants
DEFAULT_CENTER = [27.7663, -82.6404]  # Florida center
DEFAULT_ZOOM = 7
CACHE_TTL = 300  # 5 minutes

# ProCreate Fertility Brand Colors
BRAND_COLORS = {
    'primary': '#1B4B77',
    'secondary': '#4A90A4',
    'accent': '#87CEEB',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'light': '#F8F9FA',
    'dark': '#2C3E50',
    'text': '#333333',
    'background': '#FFFFFF'
}

# Enhanced color scheme for different facility types
TYPE_COLORS = {
    'Lab': BRAND_COLORS['danger'],
    'Satellite': BRAND_COLORS['secondary'],
    'Clinical Laboratory': BRAND_COLORS['warning'],
    'IVF Center': BRAND_COLORS['primary'],
    'Fertility Clinic': BRAND_COLORS['success'],
    'Clinic': BRAND_COLORS['dark'],
    'Unknown': BRAND_COLORS['accent']
}

def get_mapbox_token() -> Optional[str]:
    """Get Mapbox token from environment variables or Streamlit secrets"""
    try:
        # Method 1: From Streamlit secrets
        if hasattr(st, 'secrets') and 'MAPBOX_TOKEN' in st.secrets:
            logger.info("Mapbox token loaded from Streamlit secrets")
            return st.secrets['MAPBOX_TOKEN']
    except Exception as e:
        logger.debug(f"Could not load from Streamlit secrets: {e}")
    
    try:
        # Method 2: From environment variable
        token = os.getenv("MAPBOX_TOKEN")
        if token:
            logger.info("Mapbox token loaded from environment variable")
            return token
    except Exception as e:
        logger.debug(f"Could not load from environment variable: {e}")
    
    logger.warning("No Mapbox token found - will use free map tiles")
    return None

def find_csv_file() -> Optional[str]:
    """Find the CSV file with flexible naming and better error handling"""
    possible_patterns = [
        '*competitors*sart*cdc*ahca*.csv',
        '*competitors*.csv',
        '*fertility*.csv',
        '*clinic*.csv'
    ]
    
    current_dir = Path.cwd()
    
    # Get all CSV files
    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        st.error("⚠️ No CSV files found in current directory")
        st.info("Please upload a CSV file containing fertility clinic data")
        return None
    
    # Convert to string names for display
    csv_names = [f.name for f in csv_files]
    st.sidebar.info(f"📄 Available CSV files: {csv_names}")
    
    # Try pattern matching
    for pattern in possible_patterns:
        matches = list(current_dir.glob(pattern))
        if matches:
            selected_file = matches[0].name
            st.sidebar.success(f"✅ Found matching file: {selected_file}")
            return selected_file
    
    # If no pattern matches, let user select
    if len(csv_files) == 1:
        selected_file = csv_files[0].name
        st.sidebar.info(f"ℹ️ Using only available file: {selected_file}")
        return selected_file
    else:
        st.sidebar.warning("⚠️ Multiple CSV files found. Please select one:")
        selected_file = st.sidebar.selectbox(
            "Select CSV file:",
            csv_names,
            index=0
        )
        return selected_file

@st.cache_data(ttl=CACHE_TTL)
def load_data() -> pd.DataFrame:
    """Load and process the fertility clinic data with enhanced error handling"""
    try:
        csv_file = find_csv_file()
        
        if csv_file is None:
            return pd.DataFrame()
        
        st.info(f"📊 Loading data from: {csv_file}")
        
        # Read CSV with multiple encoding attempts
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            st.error("❌ Could not read CSV file with any supported encoding")
            return pd.DataFrame()
        
        logger.info(f"Successfully loaded {len(df)} records from {csv_file}")
        st.sidebar.write("📋 Raw columns detected:", list(df.columns))
        
        # Process the data
        df = process_csv_structure(df)
        df = clean_and_validate_data(df)
        
        if df.empty:
            st.warning("⚠️ No valid data found after processing")
        else:
            st.sidebar.success(f"✅ Successfully processed {len(df)} records")
        
        return df
        
    except FileNotFoundError:
        st.error(f"❌ CSV file not found: {csv_file}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def process_csv_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Process CSV structure with improved column mapping and validation"""
    try:
        if df.empty:
            return df
            
        cols = df.columns.tolist()
        processed_df = df.copy()
        
        # Define column mapping patterns
        column_patterns = {
            'Total_Clinics_Competitors': ['clinic', 'competitor', 'total', 'count'],
            'Location': ['location', 'address', 'place'],
            'ART_Cycles': ['art', 'cycle', 'treatment'],
            'Deliveries': ['deliver', 'birth', 'outcome'],
            'Success_Rate': ['success', 'rate', 'percentage', '%'],
            'Latitude': ['lat', 'latitude'],
            'Longitude': ['lon', 'lng', 'longitude'],
            'City': ['city', 'municipality'],
            'State': ['state', 'province'],
            'Name': ['name', 'facility', 'clinic'],
            'Type': ['type', 'category', 'kind']
        }
        
        # Initialize required columns with defaults
        required_columns = {
            'Total_Clinics_Competitors': 1,
            'Location': 'Unknown Location',
            'ART_Cycles': 0,
            'Deliveries': 0,
            'Success_Rate': 0.0,
            'Latitude': DEFAULT_CENTER[0],
            'Longitude': DEFAULT_CENTER[1],
            'City': 'Unknown',
            'State': 'FL',
            'Name': 'Unknown Clinic',
            'Type': 'Fertility Clinic'
        }
        
        # Smart column mapping
        for target_col, patterns in column_patterns.items():
            if target_col in processed_df.columns:
                continue
                
            for col in cols:
                col_lower = str(col).lower().strip()
                if any(pattern in col_lower for pattern in patterns):
                    processed_df = processed_df.rename(columns={col: target_col})
                    logger.info(f"Mapped '{col}' to '{target_col}'")
                    break
        
        # Fallback: positional mapping if we have enough columns
        if len(cols) >= 11 and 'Total_Clinics_Competitors' not in processed_df.columns:
            positional_mapping = {
                0: 'Total_Clinics_Competitors',
                1: 'Location',
                7: 'ART_Cycles',
                9: 'Deliveries',
                10: 'Success_Rate'
            }
            
            for pos, target_col in positional_mapping.items():
                if pos < len(cols) and target_col not in processed_df.columns:
                    processed_df = processed_df.rename(columns={cols[pos]: target_col})
        
        # Ensure all required columns exist
        for col, default_value in required_columns.items():
            if col not in processed_df.columns:
                processed_df[col] = default_value
                logger.info(f"Added missing column '{col}' with default value")
        
        # Display column mapping results
        mapped_cols = {col: col for col in processed_df.columns if col in required_columns}
        st.sidebar.write("🔄 Column mapping results:", mapped_cols)
        
        return processed_df
        
    except Exception as e:
        st.error(f"❌ Error processing CSV structure: {str(e)}")
        logger.error(f"Error processing CSV structure: {str(e)}")
        return df

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the loaded data with improved error handling"""
    try:
        if df.empty:
            return df
            
        st.sidebar.write(f"📊 Initial data shape: {df.shape}")
        initial_count = len(df)
        
        # Convert numeric columns with better error handling
        numeric_columns = ['Total_Clinics_Competitors', 'ART_Cycles', 'Deliveries', 'Success_Rate', 'Latitude', 'Longitude']
        for col in numeric_columns:
            if col in df.columns:
                # Handle percentage strings for Success_Rate
                if col == 'Success_Rate':
                    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Log conversion issues
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Column '{col}': {null_count} values couldn't be converted to numeric")
        
        # Fill NaN values with appropriate defaults
        fill_values = {
            'Total_Clinics_Competitors': 1,
            'ART_Cycles': 0,
            'Deliveries': 0,
            'Success_Rate': 0.0,
            'Latitude': DEFAULT_CENTER[0],
            'Longitude': DEFAULT_CENTER[1]
        }
        
        for col, fill_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        
        # Validate and clean coordinates
        initial_coord_count = len(df)
        df = df[
            (df['Latitude'].between(-90, 90)) & 
            (df['Longitude'].between(-180, 180))
        ]
        coord_removed = initial_coord_count - len(df)
        if coord_removed > 0:
            logger.warning(f"Removed {coord_removed} records with invalid coordinates")
        
        # Clean and validate string columns
        string_columns = ['Location', 'Name', 'City', 'State', 'Type']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], 'Unknown')
        
        # Validate and clean success rate
        df['Success_Rate'] = df['Success_Rate'].clip(0, 100)
        
        # Remove rows with all default/unknown values
        df = df[~(
            (df['Name'] == 'Unknown Clinic') & 
            (df['Location'] == 'Unknown Location') & 
            (df['ART_Cycles'] == 0) & 
            (df['Deliveries'] == 0)
        )]
        
        final_count = len(df)
        records_removed = initial_count - final_count
        
        st.sidebar.write(f"✅ After cleaning: {df.shape}")
        if records_removed > 0:
            st.sidebar.warning(f"⚠️ Removed {records_removed} invalid records")
        
        logger.info(f"Data cleaned. {final_count} records remaining from {initial_count} initial records")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        st.error(f"❌ Error cleaning data: {str(e)}")
        return df

def calculate_key_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key statistics for the dashboard with error handling"""
    try:
        if df.empty:
            return {
                'total_clinics': 0,
                'locations': 0,
                'art_cycles': 0,
                'deliveries': 0,
                'avg_success_rate': 0.0,
                'top_performer': 'N/A',
                'total_success_rate': 0.0
            }
        
        # Calculate statistics
        total_cycles = df['ART_Cycles'].sum()
        total_deliveries = df['Deliveries'].sum()
        
        # Find top performer
        top_performer = df.loc[df['Success_Rate'].idxmax(), 'Name'] if not df.empty else 'N/A'
        
        # Calculate overall success rate
        overall_success_rate = (total_deliveries / total_cycles * 100) if total_cycles > 0 else 0.0
        
        stats = {
            'total_clinics': int(df['Total_Clinics_Competitors'].sum()),
            'locations': int(df['Location'].nunique()),
            'art_cycles': int(total_cycles),
            'deliveries': int(total_deliveries),
            'avg_success_rate': float(df['Success_Rate'].mean()),
            'top_performer': str(top_performer),
            'total_success_rate': float(overall_success_rate)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return {
            'total_clinics': 0,
            'locations': 0,
            'art_cycles': 0,
            'deliveries': 0,
            'avg_success_rate': 0.0,
            'top_performer': 'N/A',
            'total_success_rate': 0.0
        }

def create_enhanced_map(df: pd.DataFrame, selected_types: List[str] = None) -> Optional[folium.Map]:
    """Create an enhanced interactive map with fertility clinics"""
    try:
        if df.empty:
            st.warning("⚠️ No data to display on map")
            return None
        
        # Filter data based on selected types
        if selected_types:
            df_filtered = df[df['Type'].isin(selected_types)]
        else:
            df_filtered = df
        
        if df_filtered.empty:
            st.warning("⚠️ No data matches the selected filters")
            return None
        
        # Calculate center point
        center_lat = df_filtered['Latitude'].mean()
        center_lon = df_filtered['Longitude'].mean()
        
        # Validate center point
        if pd.isna(center_lat) or pd.isna(center_lon):
            center_lat, center_lon = DEFAULT_CENTER
        
        # Create the map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=DEFAULT_ZOOM,
            tiles=None
        )
        
        # Add tile layers
        mapbox_token = get_mapbox_token()
        if mapbox_token:
            try:
                folium.TileLayer(
                    tiles=f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}',
                    attr='© <a href="https://www.mapbox.com/about/maps/">Mapbox</a> © <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
                    name='Streets (Mapbox)',
                    overlay=False,
                    control=True
                ).add_to(m)
            except Exception as e:
                logger.warning(f"Mapbox tiles failed, using OpenStreetMap: {e}")
        
        # Always add OpenStreetMap as fallback
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        
        # Create marker clusters for better performance
        marker_cluster = plugins.MarkerCluster(
            name='Fertility Clinics',
            overlay=True,
            control=True,
            options={
                'disableClusteringAtZoom': 12,
                'maxClusterRadius': 50
            }
        ).add_to(m)
        
        # Add markers with enhanced popups
        for idx, row in df_filtered.iterrows():
            try:
                lat, lon = float(row['Latitude']), float(row['Longitude'])
                
                # Skip invalid coordinates
                if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                    continue
                
                clinic_type = str(row['Type'])
                color = TYPE_COLORS.get(clinic_type, BRAND_COLORS['primary'])
                
                # Create enhanced popup content
                popup_content = f"""
                <div style="width: 300px; font-family: Arial, sans-serif; color: {BRAND_COLORS['text']}; line-height: 1.4;">
                    <h4 style="color: {BRAND_COLORS['primary']}; margin: 0 0 10px 0; font-size: 16px;">{row['Name']}</h4>
                    <hr style="margin: 10px 0; border: 1px solid {BRAND_COLORS['accent']};">
                    <p style="margin: 5px 0;"><strong>Type:</strong> <span style="color: {color}; font-weight: bold;">{clinic_type}</span></p>
                    <p style="margin: 5px 0;"><strong>Location:</strong> {row['Location']}</p>
                    <p style="margin: 5px 0;"><strong>City:</strong> {row['City']}, {row['State']}</p>
                    <hr style="margin: 10px 0; border: 1px solid {BRAND_COLORS['accent']};">
                    <p style="margin: 5px 0;"><strong>A.R.T Cycles:</strong> {int(row['ART_Cycles']):,}</p>
                    <p style="margin: 5px 0;"><strong>Deliveries:</strong> {int(row['Deliveries']):,}</p>
                    <p style="margin: 5px 0;"><strong>Success Rate:</strong> <span style="color: {BRAND_COLORS['success']}; font-weight: bold;">{row['Success_Rate']:.1f}%</span></p>
                    <p style="margin: 5px 0;"><strong>Competitors:</strong> {int(row['Total_Clinics_Competitors'])}</p>
                </div>
                """
                
                # Choose icon based on type
                icon_mapping = {
                    'IVF Center': ('heart', 'red'),
                    'Lab': ('flask', 'blue'),
                    'Clinical Laboratory': ('microscope', 'blue'),
                    'Satellite': ('satellite', 'orange'),
                    'Fertility Clinic': ('plus', 'green'),
                    'Clinic': ('hospital-o', 'green')
                }
                
                icon_name, icon_color = icon_mapping.get(clinic_type, ('plus', 'green'))
                
                # Create marker
                marker = folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_content, max_width=350),
                    tooltip=f"{row['Name']} ({clinic_type})",
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                )
                
                marker.add_to(marker_cluster)
                
            except Exception as e:
                logger.warning(f"Error creating marker for row {idx}: {e}")
                continue
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add a legend
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <h4 style="margin: 0 0 10px 0; color: {BRAND_COLORS['primary']};">Clinic Types</h4>
        """
        
        for clinic_type, color in TYPE_COLORS.items():
            if clinic_type in df_filtered['Type'].values:
                legend_html += f'<p style="margin: 5px 0;"><span style="color: {color};">●</span> {clinic_type}</p>'
        
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
        
    except Exception as e:
        st.error(f"❌ Error creating map: {str(e)}")
        logger.error(f"Error creating map: {str(e)}")
        return None

def create_charts(df: pd.DataFrame) -> None:
    """Create interactive charts for data analysis with improved error handling"""
    try:
        if df.empty:
            st.warning("⚠️ No data available for charts")
            return
        
        st.subheader("📊 Data Analysis")
        
        # Row 1: Type distribution and Success Rate by Location
        col1, col2 = st.columns(2)
        
        with col1:
            # Type distribution pie chart
            type_counts = df['Type'].value_counts()
            if not type_counts.empty:
                colors = [TYPE_COLORS.get(t, BRAND_COLORS['primary']) for t in type_counts.index]
                
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Distribution by Clinic Type",
                    color_discrete_sequence=colors
                )
                fig_pie.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text']),
                    title_font=dict(size=16, color=BRAND_COLORS['primary']),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Success Rate by Location
            location_success = df.groupby('Location')['Success_Rate'].mean().sort_values(ascending=False).head(10)
            
            if not location_success.empty:
                fig_bar = px.bar(
                    x=location_success.values,
                    y=location_success.index,
                    orientation='h',
                    title="Top 10 Locations by Success Rate",
                    labels={'x': 'Average Success Rate (%)', 'y': 'Location'},
                    color=location_success.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_bar.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text']),
                    title_font=dict(size=16, color=BRAND_COLORS['primary']),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Row 2: Cycles vs Deliveries and Success Rate Gauge
        if df['ART_Cycles'].sum() > 0 and df['Deliveries'].sum() > 0:
            st.subheader("🔬 A.R.T Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: ART Cycles vs Deliveries
                fig_scatter = px.scatter(
                    df, 
                    x='ART_Cycles', 
                    y='Deliveries',
                    color='Type',
                    size='Success_Rate',
                    hover_name='Name',
                    hover_data=['Location', 'Success_Rate'],
                    title="A.R.T Cycles vs Deliveries",
                    color_discrete_map=TYPE_COLORS,
                    size_max=30
                )
                
                fig_scatter.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text']),
                    title_font=dict(size=16, color=BRAND_COLORS['primary'])
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Overall success rate gauge
                total_cycles = df['ART_Cycles'].sum()
                total_deliveries = df['Deliveries'].sum()
                overall_success_rate = (total_deliveries / total_cycles * 100) if total_cycles > 0 else 0
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_success_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Market Success Rate (%)"},
                    delta={'reference': df['Success_Rate'].mean()},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': BRAND_COLORS['primary']},
                        'steps': [
                            {'range': [0, 25], 'color': BRAND_COLORS['light']},
                            {'range': [25, 50], 'color': BRAND_COLORS['warning']},
                            {'range': [50, 75], 'color': BRAND_COLORS['secondary']},
                            {'range': [75, 100], 'color': BRAND_COLORS['success']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text'])
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Row 3: Additional insights
        st.subheader("📈 Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Competition intensity by location
            competition_by_location = df.groupby('Location')['Total_Clinics_Competitors'].sum().sort_values(ascending=False).head(10)
            
            if not competition_by_location.empty:
                fig_competition = px.bar(
                    x=competition_by_location.index,
                    y=competition_by_location.values,
                    title="Competition Intensity by Location",
                    labels={'x': 'Location', 'y': 'Total Competitors'},
                    color=competition_by_location.values,
                    color_continuous_scale='Reds'
                )
                fig_competition.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text']),
                    title_font=dict(size=16, color=BRAND_COLORS['primary']),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_competition, use_container_width=True)
        
        with col2:
            # Performance correlation
            if df['ART_Cycles'].sum() > 0 and df['Success_Rate'].sum() > 0:
                correlation = df[['ART_Cycles', 'Deliveries', 'Success_Rate']].corr()
                
                fig_heatmap = px.imshow(
                    correlation,
                    text_auto=True,
                    aspect="auto",
                    title="Performance Metrics Correlation",
                    color_continuous_scale='RdBu_r'
                )
                fig_heatmap.update_layout(
                    height=400,
                    font=dict(family="Arial, sans-serif", color=BRAND_COLORS['text']),
                    title_font=dict(size=16, color=BRAND_COLORS['primary'])
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating charts: {str(e)}")
        logger.error(f"Error creating charts: {str(e)}")

def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Create sidebar filters with improved UI and validation"""
    st.sidebar.header("🔧 Data Filters")
    
    # Mapbox token status
    mapbox_token = get_mapbox_token()
    if mapbox_token:
        st.sidebar.success("✅ Enhanced maps active")
    else:
        st.sidebar.info("ℹ️ Using standard map tiles")
    
    if df.empty:
        st.sidebar.warning("⚠️ No data available for filters")
        return {
            'types': [],
            'locations': [],
            'success_rate_range': (0.0, 100.0),
            'cycles_range': (0, 1000),
            'show_high_performers': False
        }
    
    # Data summary
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Clinic Types", df['Type'].nunique())
    st.sidebar.metric("Locations", df['Location'].nunique())
    
    st.sidebar.subheader("🎛️ Filter Options")
    
    # Type filter
    available_types = sorted(df['Type'].unique())
    filters = {}
    
    filters['types'] = st.sidebar.multiselect(
        "🏥 Select Clinic Types",
        available_types,
        default=available_types,
        help="Choose clinic types to display on the map and in analysis"
    )
    
    # Location filter
    available_locations = sorted(df['Location'].unique())
    max_locations = min(len(available_locations), 15)  # Limit default selection
    
    filters['locations'] = st.sidebar.multiselect(
        "📍 Select Locations",
        available_locations,
        default=available_locations[:max_locations],
        help="Choose specific locations to analyze"
    )
    
    # Success rate filter
    min_success = float(df['Success_Rate'].min())
    max_success = float(df['Success_Rate'].max())
    filters['success_rate_range'] = st.sidebar.slider(
        "📈 Success Rate Range (%)",
        min_value=min_success,
        max_value=max_success,
        value=(min_success, max_success),
        step=0.1,
        help="Filter clinics by success rate percentage"
    )
    
    # ART Cycles filter
    min_cycles = int(df['ART_Cycles'].min())
    max_cycles = int(df['ART_Cycles'].max())
    if max_cycles > min_cycles:
        filters['cycles_range'] = st.sidebar.slider(
            "🔬 A.R.T Cycles Range",
            min_value=min_cycles,
            max_value=max_cycles,
            value=(min_cycles, max_cycles),
            help="Filter by number of A.R.T cycles performed"
        )
    else:
        filters['cycles_range'] = (min_cycles, max_cycles)
    
    # High performers toggle
    filters['show_high_performers'] = st.sidebar.checkbox(
        "⭐ Show Only High Performers",
        value=False,
        help="Display only clinics with above-average success rates"
    )
    
    # Download options
    st.sidebar.subheader("📥 Export Options")
    if st.sidebar.button("Download Filtered Data"):
        filtered_df = apply_filters(df, filters)
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"fertility_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to the dataframe with improved validation"""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply type filter
    if filters.get('types'):
        filtered_df = filtered_df[filtered_df['Type'].isin(filters['types'])]
    
    # Apply location filter
    if filters.get('locations'):
        filtered_df = filtered_df[filtered_df['Location'].isin(filters['locations'])]
    
    # Apply success rate filter
    if filters.get('success_rate_range'):
        min_rate, max_rate = filters['success_rate_range']
        filtered_df = filtered_df[
            (filtered_df['Success_Rate'] >= min_rate) & 
            (filtered_df['Success_Rate'] <= max_rate)
        ]
    
    # Apply cycles filter
    if filters.get('cycles_range'):
        min_cycles, max_cycles = filters['cycles_range']
        filtered_df = filtered_df[
            (filtered_df['ART_Cycles'] >= min_cycles) & 
            (filtered_df['ART_Cycles'] <= max_cycles)
        ]
    
    # Apply high performers filter
    if filters.get('show_high_performers'):
        avg_success_rate = df['Success_Rate'].mean()
        filtered_df = filtered_df[filtered_df['Success_Rate'] >= avg_success_rate]
    
    return filtered_df

def create_data_table(df: pd.DataFrame) -> None:
    """Create an interactive data table with search and sort capabilities"""
    try:
        if df.empty:
            st.warning("⚠️ No data available for table view")
            return
        
        st.subheader("📋 Detailed Data View")
        
        # Prepare display dataframe
        display_df = df.copy()
        
        # Format columns for better display
        display_df['Success_Rate'] = display_df['Success_Rate'].apply(lambda x: f"{x:.1f}%")
        display_df['ART_Cycles'] = display_df['ART_Cycles'].apply(lambda x: f"{int(x):,}")
        display_df['Deliveries'] = display_df['Deliveries'].apply(lambda x: f"{int(x):,}")
        display_df['Total_Clinics_Competitors'] = display_df['Total_Clinics_Competitors'].apply(lambda x: f"{int(x):,}")
        
        # Select columns for display
        display_columns = [
            'Name', 'Type', 'Location', 'City', 'State', 
            'ART_Cycles', 'Deliveries', 'Success_Rate', 'Total_Clinics_Competitors'
        ]
        
        # Filter to existing columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        
        # Display the table
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Name': st.column_config.TextColumn('Clinic Name', width='large'),
                'Type': st.column_config.TextColumn('Type', width='medium'),
                'Location': st.column_config.TextColumn('Location', width='large'),
                'City': st.column_config.TextColumn('City', width='medium'),
                'State': st.column_config.TextColumn('State', width='small'),
                'ART_Cycles': st.column_config.TextColumn('A.R.T Cycles', width='medium'),
                'Deliveries': st.column_config.TextColumn('Deliveries', width='medium'),
                'Success_Rate': st.column_config.TextColumn('Success Rate', width='medium'),
                'Total_Clinics_Competitors': st.column_config.TextColumn('Competitors', width='medium')
            }
        )
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records Displayed", len(display_df))
        with col2:
            avg_success = df['Success_Rate'].mean()
            st.metric("Average Success Rate", f"{avg_success:.1f}%")
        with col3:
            total_cycles = df['ART_Cycles'].sum()
            st.metric("Total A.R.T Cycles", f"{int(total_cycles):,}")
        
    except Exception as e:
        st.error(f"❌ Error creating data table: {str(e)}")
        logger.error(f"Error creating data table: {str(e)}")

def create_streamlit_app():
    """Main Streamlit application with enhanced UI and error handling"""
    try:
        # Page configuration
        st.set_page_config(
            page_title="South Florida Fertility Market Analysis",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/fertility-dashboard',
                'Report a bug': "https://github.com/your-repo/fertility-dashboard/issues",
                'About': "# South Florida Fertility Market Analysis\nComprehensive competitive analysis dashboard for fertility clinics"
            }
        )
        
        # Custom CSS for improved styling
        st.markdown(f"""
        <style>
        .main-header {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {BRAND_COLORS['primary']};
            text-align: center;
            margin-bottom: 1rem;
            font-family: Arial, sans-serif;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .stAlert {{
            border-radius: 8px;
        }}
        
        .stSelectbox, .stMultiSelect {{
            border-radius: 8px;
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: {BRAND_COLORS['text']};
            font-size: 0.9rem;
            border-top: 1px solid {BRAND_COLORS['light']};
            margin-top: 2rem;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main-header {{
                font-size: 2rem;
            }}
            .metric-card {{
                padding: 1rem;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🏥 South Florida Fertility Market Analysis</h1>', unsafe_allow_html=True)
        
        # Load data
        with st.spinner("📊 Loading fertility market data..."):
            df = load_data()
        
        if df.empty:
            st.error("❌ No data available. Please check your CSV file and try again.")
            st.info("📝 **Expected CSV Format:**")
            st.info("- Columns should include clinic information, location data, ART cycles, deliveries, and success rates")
            st.info("- Ensure the CSV file is in the same directory as this application")
            return
        
        # Create sidebar filters
        filters = create_sidebar_filters(df)
        
        # Apply filters
        filtered_df = apply_filters(df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
            return
        
        # Calculate statistics
        stats = calculate_key_statistics(filtered_df)
        
        # Display key metrics
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Clinics", 
                value=f"{stats['total_clinics']:,}",
                help="Total number of fertility clinics in the dataset"
            )
        
        with col2:
            st.metric(
                label="Unique Locations", 
                value=f"{stats['locations']:,}",
                help="Number of unique locations with fertility services"
            )
        
        with col3:
            st.metric(
                label="Total A.R.T Cycles", 
                value=f"{stats['art_cycles']:,}",
                help="Total number of Assisted Reproductive Technology cycles"
            )
        
        with col4:
            st.metric(
                label="Total Deliveries", 
                value=f"{stats['deliveries']:,}",
                help="Total number of successful deliveries"
            )
        
        with col5:
            st.metric(
                label="Average Success Rate", 
                value=f"{stats['avg_success_rate']:.1f}%",
                help="Average success rate across all clinics"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Analytics", "📋 Data Table", "ℹ️ About"])
        
        with tab1:
            st.subheader("🗺️ Fertility Clinics Interactive Map")
            
            # Create and display map
            with st.spinner("🗺️ Generating interactive map..."):
                clinic_map = create_enhanced_map(filtered_df, filters.get('types'))
            
            if clinic_map:
                # Display map
                map_html = clinic_map._repr_html_()
                components.html(map_html, height=600)
                
                # Map statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clinics on Map", len(filtered_df))
                with col2:
                    st.metric("Average Success Rate", f"{filtered_df['Success_Rate'].mean():.1f}%")
                with col3:
                    top_performer = filtered_df.loc[filtered_df['Success_Rate'].idxmax(), 'Name']
                    st.metric("Top Performer", top_performer[:20] + "..." if len(top_performer) > 20 else top_performer)
        
        with tab2:
            # Create charts
            create_charts(filtered_df)
        
        with tab3:
            # Create data table
            create_data_table(filtered_df)
        
        with tab4:
            # About section
            st.markdown("""
            ## About This Dashboard
            
            This interactive dashboard provides comprehensive analysis of the South Florida fertility market, 
            offering insights into clinic distribution, performance metrics, and competitive landscape.
            
            ### Features
            - 🗺️ **Interactive Map**: Visualize clinic locations with detailed information
            - 📊 **Analytics**: Performance metrics and trend analysis
            - 📋 **Data Table**: Searchable and sortable clinic data
            - 🔧 **Filters**: Customize views by type, location, and performance
            
            ### Data Sources
            - SART (Society for Assisted Reproductive Technology)
            - CDC (Centers for Disease Control and Prevention)
            - AHCA (Agency for Health Care Administration)
            
            ### Technical Details
            - Built with Streamlit and Python
            - Interactive maps powered by Folium
            - Charts created with Plotly
            - Data processing with Pandas
            
            ### Support
            For technical support or questions, please contact the development team.
            """)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>© 2024 ProCreate Fertility Market Analysis Dashboard | 
            Built with ❤️ using Streamlit | 
            Data sources: SART, CDC, AHCA</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    create_streamlit_app()