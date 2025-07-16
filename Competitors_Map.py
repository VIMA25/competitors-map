import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import glob

# Configuration Constants
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

TYPE_COLORS = {
    'Lab': BRAND_COLORS['danger'],
    'Satellite': BRAND_COLORS['secondary'],
    'Clinical Laboratory': BRAND_COLORS['warning'],
    'IVF Center': BRAND_COLORS['primary'],
    'Fertility Clinic': BRAND_COLORS['success'],
    'Clinic': BRAND_COLORS['dark']
}

def find_csv_file():
    """Find the CSV file with flexible naming"""
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        st.error("❌ No CSV files found in current directory")
        return None
    
    # Try exact matches first
    possible_names = [
        'Competitors SART-CDC-AHCA.csv',
        'Competitors_SART-CDC-AHCA.csv',
        'competitors sart-cdc-ahca.csv',
    ]
    
    for name in possible_names:
        if name in csv_files:
            return name
    
    # Look for any CSV with "competitors" in the name
    competitor_files = [f for f in csv_files if 'competitors' in f.lower()]
    if competitor_files:
        return competitor_files[0]
    
    # Default to first CSV file
    return csv_files[0]

@st.cache_data
def load_data():
    """Load and process the fertility clinic data"""
    try:
        csv_file = find_csv_file()
        
        if csv_file is None:
            return pd.DataFrame()
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin1')
        
        st.sidebar.success(f"✅ Loaded: {csv_file}")
        st.sidebar.write(f"📊 Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Show first few column names for debugging
        st.sidebar.write("📋 First 5 columns:", list(df.columns[:5]))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

def process_data(df):
    """Process and clean the data"""
    if df.empty:
        return df
    
    # Create a copy to work with
    processed_df = df.copy()
    
    # Try to identify key columns by position or name
    cols = list(df.columns)
    
    # Show all columns for debugging
    st.sidebar.write("🔍 All columns:", cols)
    
    # Try to map important columns
    column_mapping = {}
    
    # Look for specific patterns in column names
    for i, col in enumerate(cols):
        col_lower = str(col).lower().strip()
        
        if 'location' in col_lower and 'Location' not in column_mapping:
            column_mapping[col] = 'Location'
        elif 'cycle' in col_lower and 'ART_Cycles' not in column_mapping:
            column_mapping[col] = 'ART_Cycles'
        elif 'deliver' in col_lower and 'Deliveries' not in column_mapping:
            column_mapping[col] = 'Deliveries'
        elif ('success' in col_lower or 'rate' in col_lower) and 'Success_Rate' not in column_mapping:
            column_mapping[col] = 'Success_Rate'
        elif 'name' in col_lower and 'Name' not in column_mapping:
            column_mapping[col] = 'Name'
        elif 'type' in col_lower and 'Type' not in column_mapping:
            column_mapping[col] = 'Type'
    
    # Apply the mapping
    processed_df = processed_df.rename(columns=column_mapping)
    
    # If we still don't have key columns, try by position
    if 'ART_Cycles' not in processed_df.columns and len(cols) > 7:
        # Assume column 8 (index 7) is ART_Cycles based on your description
        processed_df = processed_df.rename(columns={cols[7]: 'ART_Cycles'})
    
    if 'Deliveries' not in processed_df.columns and len(cols) > 9:
        # Assume column 10 (index 9) is Deliveries
        processed_df = processed_df.rename(columns={cols[9]: 'Deliveries'})
    
    if 'Success_Rate' not in processed_df.columns and len(cols) > 10:
        # Assume column 11 (index 10) is Success_Rate (K column)
        processed_df = processed_df.rename(columns={cols[10]: 'Success_Rate'})
    
    # Fill missing columns with defaults
    if 'Name' not in processed_df.columns:
        processed_df['Name'] = f'Clinic {processed_df.index + 1}'
    
    if 'Type' not in processed_df.columns:
        processed_df['Type'] = 'Fertility Clinic'
    
    if 'Location' not in processed_df.columns:
        processed_df['Location'] = 'Florida'
    
    # Convert numeric columns
    numeric_columns = ['ART_Cycles', 'Deliveries', 'Success_Rate']
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    # Clean string columns
    string_columns = ['Name', 'Type', 'Location']
    for col in string_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str).str.strip()
    
    st.sidebar.write(f"✅ Processed data shape: {processed_df.shape}")
    
    return processed_df

def calculate_statistics(df):
    """Calculate key statistics - FIXED VERSION"""
    if df.empty:
        return {
            'total_clinics': 0,
            'locations': 0,
            'art_cycles': 0,
            'deliveries': 0,
            'avg_success_rate': 0.0,
            'top_clinic_name': 'N/A',
            'top_clinic_cycles': 0,
            'top_clinic_success_rate': 0.0
        }
    
    # FIXED: Total clinics = number of rows (as requested)
    total_clinics = len(df)
    
    # FIXED: Calculate average success rate properly (K2:K15 equivalent)
    avg_success_rate = df['Success_Rate'].mean() if 'Success_Rate' in df.columns else 0.0
    
    # Find top clinic by ART cycles
    if 'ART_Cycles' in df.columns and df['ART_Cycles'].sum() > 0:
        top_clinic_idx = df['ART_Cycles'].idxmax()
        top_clinic = df.loc[top_clinic_idx]
        top_clinic_name = str(top_clinic['Name']) if 'Name' in df.columns else f'Clinic {top_clinic_idx + 1}'
        top_clinic_cycles = int(top_clinic['ART_Cycles']) if 'ART_Cycles' in df.columns else 0
        top_clinic_success_rate = float(top_clinic['Success_Rate']) if 'Success_Rate' in df.columns else 0.0
    else:
        top_clinic_name = 'N/A'
        top_clinic_cycles = 0
        top_clinic_success_rate = 0.0
    
    stats = {
        'total_clinics': total_clinics,  # FIXED: Number of rows
        'locations': int(df['Location'].nunique()) if 'Location' in df.columns else 1,  # FIXED: Just "Locations"
        'art_cycles': int(df['ART_Cycles'].sum()) if 'ART_Cycles' in df.columns else 0,
        'deliveries': int(df['Deliveries'].sum()) if 'Deliveries' in df.columns else 0,
        'avg_success_rate': float(avg_success_rate),  # FIXED: Proper average
        'top_clinic_name': top_clinic_name,
        'top_clinic_cycles': top_clinic_cycles,
        'top_clinic_success_rate': top_clinic_success_rate
    }
    
    return stats

def create_charts(df):
    """Create simple charts"""
    if df.empty:
        st.warning("⚠️ No data available for charts")
        return
    
    st.subheader("📊 Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Type' in df.columns:
            # Type distribution
            type_counts = df['Type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution by Type"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Type column not found")
    
    with col2:
        if 'ART_Cycles' in df.columns and 'Deliveries' in df.columns:
            # Cycles vs Deliveries
            fig_scatter = px.scatter(
                df, 
                x='ART_Cycles', 
                y='Deliveries',
                title="A.R.T Cycles vs Deliveries",
                hover_name='Name' if 'Name' in df.columns else None
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("ART_Cycles or Deliveries column not found")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="South Florida Fertility Market Analysis",
        page_icon="🏥",
        layout="wide"
    )
    
    # Simple CSS
    st.markdown(f"""
    <style>
    .metric-card {{
        background-color: {BRAND_COLORS['light']};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {BRAND_COLORS['primary']};
        margin: 0.5rem 0;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {BRAND_COLORS['primary']};
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {BRAND_COLORS['text']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🏥 South Florida Fertility Market")
    st.markdown("### Competitive Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Unable to load data. Please ensure the CSV file is in the current directory.")
        st.info("Expected file: 'Competitors SART-CDC-AHCA.csv' or similar")
        return
    
    # Process data
    processed_df = process_data(df)
    
    # Calculate statistics
    stats = calculate_statistics(processed_df)
    
    # Display KPIs
    st.subheader("📈 Key Performance Indicators")
    
    # First row of KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_clinics']}</div>
            <div class="metric-label">Total Clinics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['locations']}</div>
            <div class="metric-label">Locations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['art_cycles']:,}</div>
            <div class="metric-label">Total A.R.T Cycles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_success_rate']:.1f}%</div>
            <div class="metric-label">Average Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row of KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['deliveries']:,}</div>
            <div class="metric-label">Total Deliveries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['top_clinic_cycles']:,}</div>
            <div class="metric-label">Top Clinic Cycles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['top_clinic_success_rate']:.1f}%</div>
            <div class="metric-label">Top Clinic Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top clinic info
    if stats['top_clinic_name'] != 'N/A':
        st.info(f"🏆 **Top Performing Clinic:** {stats['top_clinic_name']}")
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Data Table", "ℹ️ Debug Info"])
    
    with tab1:
        create_charts(processed_df)
    
    with tab2:
        st.subheader("📋 Data Table")
        if not processed_df.empty:
            st.dataframe(processed_df, use_container_width=True)
            
            # Download button
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"fertility_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to display")
    
    with tab3:
        st.subheader("🔍 Debug Information")
        
        if not df.empty:
            st.write("**Original DataFrame Info:**")
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            st.write("First 5 rows:")
            st.dataframe(df.head())
            
            st.write("**Processed DataFrame Info:**")
            st.write(f"Shape: {processed_df.shape}")
            st.write("Columns:", list(processed_df.columns))
            
            st.write("**Statistics:**")
            st.json(stats)
        else:
            st.write("No data loaded")

if __name__ == "__main__":
    main()