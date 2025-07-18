import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import glob

# Optional: load environment variables if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import folium for advanced mapping
try:
    import folium
    from folium import plugins
    from streamlit_folium import st_folium
    import branca.colormap as cm
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    st.sidebar.warning("📍 Install folium for advanced mapping: `pip install folium streamlit-folium`")

# Configuration Constants
BRAND_COLORS = {
    'primary': '#0086A3',        # Pantone 7706 C (teal)
    'secondary': '#009CA6',      # Pantone 320 C (turquoise)
    'accent': '#63666A',         # Pantone Cool Gray 10 C
    'success': '#97999B',        # Pantone Cool Gray 7 C
    'warning': '#D1D3D4',        # Pantone Cool Gray 3 C
    'danger': '#DC3545',
    'light': '#F8F9FA',
    'dark': '#2C3E50',
    'text': '#333333',
    'background': '#FFFFFF'
}

FONT_NAME = "'Bell Centennial Address BT', sans-serif"

TYPE_COLORS = {
    'Lab': BRAND_COLORS['danger'],
    'Satellite': BRAND_COLORS['secondary'],
    'Clinical Laboratory': BRAND_COLORS['warning'],
    'IVF Center': BRAND_COLORS['primary'],
    'Fertility Clinic': BRAND_COLORS['success'],
    'Clinic': BRAND_COLORS['dark']
}

def find_data_file():
    """Find the Excel or CSV file with flexible naming"""
    # Look for both .xlsx and .xls files
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    csv_files = glob.glob("*.csv")
    
    # Debug: Show what files are found
    st.sidebar.write("**Debug - Files found:**")
    st.sidebar.write(f"Excel files (.xlsx/.xls): {excel_files}")
    st.sidebar.write(f"CSV files: {csv_files}")
    
    # Prioritize Excel files since they can contain multiple sheets
    possible_excel_names = [
        'Competitors SART-CDC-AHCA.xlsx',
        'Competitors SART-CDC-AHCA.xls',  # Added .xls support
        'Competitors_SART-CDC-AHCA.xlsx',
        'Competitors_SART-CDC-AHCA.xls',
        'competitors sart-cdc-ahca.xlsx',
        'competitors sart-cdc-ahca.xls',
    ]
    
    for name in possible_excel_names:
        if name in excel_files:
            st.sidebar.write(f"✅ Found exact Excel match: {name}")
            return name, 'excel'
    
    # If exact Excel match not found, use first Excel file
    if excel_files:
        st.sidebar.write(f"⚠️ Using first Excel file found: {excel_files[0]}")
        return excel_files[0], 'excel'
    
    # Fallback to CSV
    possible_csv_names = [
        'Competitors SART-CDC-AHCA.csv',
        'Competitors_SART-CDC-AHCA.csv',
        'competitors sart-cdc-ahca.csv',
    ]
    
    for name in possible_csv_names:
        if name in csv_files:
            st.sidebar.write(f"✅ Found exact CSV match: {name}")
            return name, 'csv'
    
    if csv_files:
        st.sidebar.write(f"⚠️ Using first CSV file found: {csv_files[0]}")
        return csv_files[0], 'csv'
    
    st.error("❌ No Excel or CSV files found in current directory")
    return None, None

@st.cache_data
def load_data():
    """Load and cache the data from Excel or CSV"""
    file_path, file_type = find_data_file()
    if not file_path:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        main_df = None
        summary_df = None
        
        if file_type == 'excel':
            # Load from Excel file
            st.sidebar.success(f"✅ Loading from Excel file: {file_path}")
            
            try:
                # Check if it's .xls file and handle dependency
                if file_path.endswith('.xls'):
                    try:
                        # Try to import xlrd
                        import xlrd
                    except ImportError:
                        st.error("❌ Missing dependency for .xls files. Please install xlrd or convert your file to .xlsx format.")
                        st.info("💡 **Quick Fix**: Save your Excel file as .xlsx format (newer Excel format) instead of .xls")
                        st.info("💡 **Alternative**: Install xlrd by running: `pip install xlrd`")
                        return pd.DataFrame(), pd.DataFrame()
                
                # First, let's see what sheets are available
                excel_file = pd.ExcelFile(file_path)
                available_sheets = excel_file.sheet_names
                st.sidebar.write(f"**Available sheets:** {available_sheets}")
                
                # Load main data from first sheet
                main_df = pd.read_excel(file_path, sheet_name=0)  # First sheet
                st.sidebar.success(f"✅ Main data loaded from first sheet: '{available_sheets[0]}'")
                st.sidebar.write(f"Main data shape: {main_df.shape}")
                
                # Try to find summary sheet
                summary_sheet_names = ['Competitors Summary', 'Summary', 'competitors summary']
                summary_df = None
                
                for sheet_name in summary_sheet_names:
                    if sheet_name in available_sheets:
                        try:
                            summary_df = pd.read_excel(file_path, sheet_name=sheet_name)
                            st.sidebar.success(f"✅ Summary data loaded from '{sheet_name}' sheet")
                            st.sidebar.write(f"Summary data shape: {summary_df.shape}")
                            st.sidebar.write(f"Summary data columns: {summary_df.columns.tolist()}")
                            st.sidebar.write("**Raw summary data:**")
                            st.sidebar.write(summary_df.head().to_string())
                            break
                        except Exception as e:
                            st.sidebar.error(f"Error loading sheet '{sheet_name}': {e}")
                
                if summary_df is None:
                    st.sidebar.warning(f"Could not find summary sheet. Tried: {summary_sheet_names}")
                    st.sidebar.warning("Available sheets: " + str(available_sheets))
                
            except Exception as e:
                st.sidebar.error(f"Error loading Excel sheets: {e}")
                
                # Check if it's the xlrd issue
                if "xlrd" in str(e).lower() or "xls" in str(e).lower():
                    st.error("❌ Cannot read .xls file. Missing xlrd dependency.")
                    st.info("💡 **Quick Fix**: Please save your Excel file as .xlsx format (File → Save As → Excel Workbook (.xlsx))")
                    st.info("💡 **Alternative**: Install xlrd by running: `pip install xlrd`")
                else:
                    import traceback
                    st.sidebar.error(traceback.format_exc())
                
                return pd.DataFrame(), pd.DataFrame()
        
        else:  # CSV file
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    main_df = pd.read_csv(file_path, encoding=encoding)
                    st.sidebar.success(f"✅ CSV loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if main_df is None:
                st.error("❌ Could not read the CSV file with any encoding")
                return pd.DataFrame(), pd.DataFrame()
        
        st.sidebar.write(f"📁 **File:** {file_path}")
        st.sidebar.write(f"📊 **Main Data Shape:** {main_df.shape[0]} rows × {main_df.shape[1]} columns")
        
        if summary_df is not None:
            st.sidebar.write(f"📊 **Summary Data Shape:** {summary_df.shape[0]} rows × {summary_df.shape[1]} columns")
            st.sidebar.write("**Summary Data Preview:**")
            st.sidebar.write(summary_df.head())
        
        return main_df, summary_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def clean_column_names(df):
    """Standardize column names"""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Handle merged cells - forward fill clinic names
    if 'Clinic (Competitor)' in df_cleaned.columns:
        df_cleaned['Clinic (Competitor)'] = df_cleaned['Clinic (Competitor)'].fillna(method='ffill')
    
    # Mapping of possible column names to standardized names
    column_mapping = {
        # Name variations
        'clinic_(competitor)': 'Name',
        'clinic_competitor': 'Name',
        'clinic_name': 'Name',
        'name': 'Name',
        'facility_name': 'Name',
        'clinic': 'Name',
        
        # Type variations
        'type': 'Type',
        'facility_type': 'Type',
        'clinic_type': 'Type',
        
        # Location variations - specifically for your CSV
        'location': 'Location',
        'city': 'Location',
        'address': 'Location',
        
        # ART Cycles variations - for your CSV "Total Cycles 2022"
        'total_cycles_2022': 'ART_Cycles',
        'total_cycles': 'ART_Cycles',
        'art_cycles': 'ART_Cycles',
        'cycles': 'ART_Cycles',
        'cycles_2022': 'ART_Cycles',
        
        # Deliveries variations - for your CSV "Deliveries  2022" (with double space)
        'deliveries_2022': 'Deliveries',
        'deliveries__2022': 'Deliveries',  # double space becomes double underscore
        'deliveries': 'Deliveries',
        'total_deliveries': 'Deliveries',
        'births': 'Deliveries',
        
        # Success Rate variations
        'success_rate': 'Success_Rate',
        'success_rate_2022': 'Success_Rate',
        'success rate': 'Success_Rate',
        'success_rate_%': 'Success_Rate',
        'success_rate_percent': 'Success_Rate',
        
        # Opening Year
        'opening_year': 'Opening_Year',
        'year_opened': 'Opening_Year',
        'established': 'Opening_Year',
        
        # Geographic coordinates
        'latitude': 'Latitude',
        'lat': 'Latitude',
        'longitude': 'Longitude',
        'lon': 'Longitude',
        'lng': 'Longitude'
    }
    
    # Clean column names (remove spaces, convert to lowercase for matching)
    original_columns = df_cleaned.columns.tolist()
    cleaned_columns = [col.strip().lower().replace(' ', '_').replace('__', '_') for col in original_columns]
    
    # Create mapping from cleaned names to standardized names
    new_column_names = {}
    for i, cleaned_col in enumerate(cleaned_columns):
        original_col = original_columns[i]
        if cleaned_col in column_mapping:
            new_column_names[original_col] = column_mapping[cleaned_col]
    
    # Rename columns
    df_cleaned = df_cleaned.rename(columns=new_column_names)
    
    # Log the column mapping
    if new_column_names:
        st.sidebar.write("🔄 **Column Mappings:**")
        for old, new in new_column_names.items():
            st.sidebar.write(f"  • {old} → {new}")
    
    return df_cleaned

def process_data(df):
    """Process and clean the data"""
    if df.empty:
        st.warning("⚠️ No data to process")
        return df
    
    # Clean column names first
    processed_df = clean_column_names(df)
    
    # Check for and handle duplicate column names
    duplicate_cols = processed_df.columns[processed_df.columns.duplicated()].tolist()
    if duplicate_cols:
        st.sidebar.warning(f"⚠️ Duplicate columns found: {duplicate_cols}")
        # Remove duplicate columns by keeping only the first occurrence
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]
    
    st.sidebar.write("📋 **Available Columns:**")
    for col in processed_df.columns:
        st.sidebar.write(f"  • {col}")
    
    # Convert numeric columns
    numeric_columns = ['ART_Cycles', 'Deliveries', 'Success_Rate', 'Latitude', 'Longitude']
    for col in numeric_columns:
        if col in processed_df.columns:
            try:
                # First, handle special text values
                processed_df[col] = processed_df[col].astype(str)
                
                # Replace non-numeric text with NaN
                processed_df[col] = processed_df[col].replace({
                    'Not Reported Yet': '',
                    'See Main Location': '',
                    'nan': '',
                    'NaN': '',
                    'None': ''
                })
                
                # Remove percentage signs and commas
                if col == 'Success_Rate':
                    processed_df[col] = processed_df[col].str.replace('%', '').str.replace(',', '')
                else:
                    processed_df[col] = processed_df[col].str.replace(',', '')
                
                # Convert empty strings to NaN, then to numeric
                processed_df[col] = processed_df[col].replace('', np.nan)
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
            except Exception as e:
                st.sidebar.error(f"Error processing numeric column {col}: {str(e)}")
    
    # Clean string columns
    string_columns = ['Name', 'Type', 'Location']
    for col in string_columns:
        if col in processed_df.columns:
            try:
                # Ensure we're working with a Series, not DataFrame
                if isinstance(processed_df[col], pd.DataFrame):
                    processed_df[col] = processed_df[col].iloc[:, 0]
                
                processed_df[col] = processed_df[col].astype(str).str.strip()
                # Remove any 'nan' strings and replace with empty string
                processed_df[col] = processed_df[col].replace(['nan', 'NaN', 'None'], '')
                
            except Exception as e:
                st.sidebar.error(f"Error processing string column {col}: {str(e)}")
    
    # Drop rows where Name is empty or NaN
    if 'Name' in processed_df.columns:
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['Name'].notna() & (processed_df['Name'] != '')]
        final_count = len(processed_df)
        if initial_count != final_count:
            st.sidebar.info(f"Removed {initial_count - final_count} rows with empty names")
    
    # Fill NaN values for numeric columns with 0
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(0)
    
    st.sidebar.write(f"✅ Processed data shape: {processed_df.shape}")
    
    return processed_df

def calculate_statistics_from_summary(summary_df, main_df):
    """Calculate statistics from the summary sheet data"""
    default_stats = {
        'total_clinics': 0, 'locations': 0, 'art_cycles': 0, 
        'pregnancies': 0, 'deliveries': 0, 'avg_success_rate': 0.0,
        'top_clinic_name': 'IVFMD', 'top_clinic_cycles': 2874, 'top_clinic_success_rate': 20.0
    }
    
    if summary_df is None or summary_df.empty:
        st.sidebar.warning("⚠️ No summary data available")
        return default_stats
    
    try:
        # Debug: Show the exact data structure
        st.sidebar.write("**EXCEL DATA DEBUG:**")
        st.sidebar.write(f"Summary shape: {summary_df.shape}")
        st.sidebar.write(f"Summary columns: {list(summary_df.columns)}")
        st.sidebar.write(f"First row data: {summary_df.iloc[0].tolist()}")
        
        # Read the exact values from your Excel structure
        row_data = summary_df.iloc[0]  # First data row
        
        # Get the success rate value and check if it needs to be multiplied by 100
        success_rate_raw = float(row_data.iloc[5]) if len(row_data) > 5 else 0.0
        # If the value is less than 1, it's likely in decimal format (0.1946 instead of 19.46)
        if success_rate_raw < 1:
            success_rate = success_rate_raw * 100
        else:
            success_rate = success_rate_raw
        
        # Map exactly to your Excel columns A-F
        stats = {
            'total_clinics': int(row_data.iloc[0]) if len(row_data) > 0 else 0,           # A: 8
            'locations': int(row_data.iloc[1]) if len(row_data) > 1 else 0,              # B: 14
            'art_cycles': int(row_data.iloc[2]) if len(row_data) > 2 else 0,             # C: 7437
            'pregnancies': int(row_data.iloc[3]) if len(row_data) > 3 else 0,            # D: 1887
            'deliveries': int(row_data.iloc[4]) if len(row_data) > 4 else 0,             # E: 1450
            'avg_success_rate': success_rate,                                             # F: 19.46%
            'top_clinic_name': 'IVFMD',
            'top_clinic_cycles': 2874,
            'top_clinic_success_rate': 20.0
        }
        
        st.sidebar.write("**FINAL STATS:**")
        for key, value in stats.items():
            st.sidebar.write(f"{key}: {value}")
        
        return stats
        
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        import traceback
        st.sidebar.error(traceback.format_exc())
        return default_stats

def calculate_statistics_from_main_data(df):
    """Calculate key statistics from main data as fallback"""
    default_stats = {
        'total_clinics': 0,
        'locations': 0,
        'art_cycles': 0,
        'pregnancies': 0,
        'deliveries': 0,
        'avg_success_rate': 0.0,
        'top_clinic_name': 'N/A',
        'top_clinic_cycles': 0,
        'top_clinic_success_rate': 0.0
    }
    
    if df.empty:
        return default_stats
    
    try:
        # Total clinics and other stats calculation
        total_clinics = len(df[df['Name'].notna() & (df['Name'] != '')]) if 'Name' in df.columns else len(df)
        
        # Locations count
        locations = len(df[df['Location'].notna() & (df['Location'] != '')]) if 'Location' in df.columns else 0
        
        # ART cycles sum
        art_cycles = int(df['ART_Cycles'].sum()) if 'ART_Cycles' in df.columns else 0
        
        # Deliveries sum
        deliveries = int(df['Deliveries'].sum()) if 'Deliveries' in df.columns else 0
        
        # Average success rate
        if 'Success_Rate' in df.columns:
            success_rates = df[df['Success_Rate'] > 0]['Success_Rate']
            avg_success_rate = success_rates.mean() if len(success_rates) > 0 else 0.0
        else:
            avg_success_rate = 0.0
        
        # Find top clinic
        if 'ART_Cycles' in df.columns and 'Name' in df.columns:
            top_clinic = df.loc[df['ART_Cycles'].idxmax()]
            top_clinic_name = top_clinic['Name']
            top_clinic_cycles = int(top_clinic['ART_Cycles'])
            top_clinic_success_rate = top_clinic.get('Success_Rate', 0.0)
        else:
            top_clinic_name = 'N/A'
            top_clinic_cycles = 0
            top_clinic_success_rate = 0.0
        
        stats = {
            'total_clinics': total_clinics,
            'locations': locations,
            'art_cycles': art_cycles,
            'pregnancies': 0,  # Not available in main data
            'deliveries': deliveries,
            'avg_success_rate': avg_success_rate,
            'top_clinic_name': top_clinic_name,
            'top_clinic_cycles': top_clinic_cycles,
            'top_clinic_success_rate': top_clinic_success_rate
        }
        
        return stats
        
    except Exception as e:
        st.sidebar.error(f"Error calculating statistics: {str(e)}")
        return default_stats

def create_map(df):
    """Display an interactive map with competitor points and heatmap layers"""
    if df.empty or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        st.info("No geographic data available for map")
        return

    # Filter invalid coords
    map_df = df[(df['Latitude'].notna()) & (df['Longitude'].notna()) &
                (df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
    if map_df.empty:
        st.info("No valid coordinates for mapping")
        return

    # Base map
    center_lat = map_df['Latitude'].mean()
    center_lon = map_df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=10,
                   tiles='OpenStreetMap',
                   prefer_canvas=True)

    # ➤ Competitor points layer
    competitor_layer = folium.FeatureGroup(name="Competitor Locations")
    heat_data = []

    for _, row in map_df.iterrows():
        # Extract info
        lat, lon = row['Latitude'], row['Longitude']
        name    = row.get('Name', 'Unknown')
        cycles  = row.get('ART_Cycles', 0)
        # color + size logic...
        color   = TYPE_COLORS.get(row.get('Type'), BRAND_COLORS['primary'])
        size    = 12 if cycles>1000 else 10 if cycles>500 else 8 if cycles>100 else 6

        # Popup + tooltip for points
        popup = folium.Popup(f"<b>{name}</b><br>Cycles: {cycles:,}", max_width=250)
        folium.CircleMarker(
            location=[lat, lon],
            radius=size,
            popup=popup,
            tooltip=f"{name} – {cycles:,} cycles",
            color='white', weight=2,
            fill=True, fillColor=color, fillOpacity=0.8
        ).add_to(competitor_layer)

        # Prepare heatmap data (lat, lon, weight)
        heat_data.append([lat, lon, min(cycles/100, 10)])

    competitor_layer.add_to(m)

    # ➤ Heatmap layer with tooltips
    heatmap_layer = folium.FeatureGroup(name="Heatmap")
    plugins.HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=13,
        radius=25,
        blur=20,
        gradient={
            '0.0': 'blue',
            '0.5': BRAND_COLORS['secondary'],
            '0.7': BRAND_COLORS['primary'],
            '1.0': 'red'
        }
    ).add_to(heatmap_layer)

    # Invisible markers to surface tooltips on hover
    for lat, lon, weight in heat_data:
        folium.CircleMarker(
            location=[lat, lon],
            radius=0, weight=0, fill=False,
            tooltip=(
                f"Lat: {lat:.4f}<br>"
                f"Lon: {lon:.4f}<br>"
                f"Weight: {weight:.1f}"
            )
        ).add_to(heatmap_layer)

    heatmap_layer.add_to(m)

    # Layer control and display
    folium.LayerControl().add_to(m)
    st.components.v1.html(m._repr_html_(), height=600, scrolling=False)


def create_plotly_map(map_df, center_lat, center_lon):
    """Create a plotly map as fallback"""
    try:
        # Create scatter map
        fig = go.Figure()
        
        # Add scatter points for each clinic type
        for clinic_type in map_df['Type'].unique() if 'Type' in map_df.columns else ['Default']:
            if 'Type' in map_df.columns:
                type_df = map_df[map_df['Type'] == clinic_type]
            else:
                type_df = map_df
                clinic_type = 'Clinic'
            
            # Create hover text
            hover_text = []
            for idx, row in type_df.iterrows():
                text = f"<b>{row.get('Name', 'Unknown')}</b><br>"
                text += f"Type: {clinic_type}<br>"
                text += f"Location: {row.get('Location', 'Unknown')}<br>"
                text += f"ART Cycles: {row.get('ART_Cycles', 0):,}<br>"
                text += f"Deliveries: {row.get('Deliveries', 0):,}<br>"
                text += f"Success Rate: {row.get('Success_Rate', 0):.1f}%"
                hover_text.append(text)
            
            # Add scatter trace
            fig.add_trace(go.Scattermapbox(
                lat=type_df['Latitude'],
                lon=type_df['Longitude'],
                mode='markers',
                marker=dict(
                    size=[min(20, max(8, cycles/100)) if cycles > 0 else 8 for cycles in type_df.get('ART_Cycles', [0]*len(type_df))],
                    color=TYPE_COLORS.get(clinic_type, BRAND_COLORS['primary']),
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo='text',
                name=clinic_type
            ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=9
            ),
            height=600,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")

def create_charts(df):
    """Create comprehensive charts and visualizations"""
    if df.empty:
        st.warning("⚠️ No data available for charts")
        return
    
    # Data Analysis header with same styling as Market Snapshot
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%); padding: 1rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <div style="text-align: center; width: 100%;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">Competitive Analysis</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for charts
    # Filter out rows with zero or missing ART cycles for meaningful analysis
    analysis_df = df[df['ART_Cycles'] > 0].copy()
    
    # Calculate years since opening if Opening_Year column exists
    current_year = 2024
    if 'Opening_Year' in analysis_df.columns:
        analysis_df['Years_Since_Opening'] = current_year - analysis_df['Opening_Year']
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Share Bar Chart
        if 'ART_Cycles' in analysis_df.columns and 'Name' in analysis_df.columns:
            # Calculate market share
            total_cycles = analysis_df['ART_Cycles'].sum()
            market_share_df = analysis_df.groupby('Name')['ART_Cycles'].sum().reset_index()
            market_share_df['Market_Share'] = (market_share_df['ART_Cycles'] / total_cycles * 100).round(1)
            market_share_df = market_share_df.sort_values('ART_Cycles', ascending=True)
            
            # Wrap long clinic names
            market_share_df['Name_Wrapped'] = market_share_df['Name'].apply(
                lambda x: '<br>'.join([x[i:i+30] for i in range(0, len(x), 30)]) if len(x) > 30 else x
            )
            
            fig_market = px.bar(
                market_share_df,
                x='ART_Cycles',
                y='Name_Wrapped',
                orientation='h',
                title=f"<b style='color:black'>Market Share by ART Cycles (Total: {total_cycles:,})</b>",
                text='Market_Share',
                color='ART_Cycles',
                color_continuous_scale=['#E0F7FA', BRAND_COLORS['primary']]
            )
            
            fig_market.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig_market.update_layout(
                height=500,  # Increased height for better visibility
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black', size=12),
                title=dict(font=dict(color='black', size=16)),
                xaxis=dict(title='ART Cycles', title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title='', title_font=dict(color='black'), tickfont=dict(color='black', size=10)),
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=200)  # More left margin for long names
            )
            
            st.plotly_chart(fig_market, use_container_width=True)
            
            # Calculate and display HHI
            hhi = (market_share_df['Market_Share'] ** 2).sum()
            st.markdown(f"""
            <div style="background: #FFFFFF; 
                        padding: 1rem; border-radius: 8px; border: 1px solid #E5E7EB; 
                        margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.9rem; color: {BRAND_COLORS['text']};">
                    <b>Herfindahl-Hirschman Index (HHI):</b> {hhi:.0f}<br>
                    <span style="font-size: 0.8rem; color: #666;">
                    {
                        "Highly Competitive Market (HHI < 1,500)" if hhi < 1500 
                        else "Moderately Concentrated (1,500 ≤ HHI < 2,500)" if hhi < 2500 
                        else "Highly Concentrated (HHI ≥ 2,500)"
                    }
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Market share data not available")
    
    with col2:
        # Success Rate vs Cycle Volume Scatter
        if all(col in analysis_df.columns for col in ['Success_Rate', 'ART_Cycles', 'Name']):
            # Prepare bubble size based on opening year if available
            if 'Opening_Year' in analysis_df.columns:
                # Older clinics get larger bubbles
                analysis_df['Bubble_Size'] = (current_year - analysis_df['Opening_Year'] + 5) * 2
                hover_data = ['Name', 'Opening_Year', 'Type']
            else:
                analysis_df['Bubble_Size'] = 20
                hover_data = ['Name', 'Type']
            
            fig_scatter = px.scatter(
                analysis_df,
                x='ART_Cycles',
                y='Success_Rate',
                size='Bubble_Size',
                title="<b style='color:black'>Success Rate vs. Cycle Volume</b>",
                hover_data=hover_data,
                color='Type' if 'Type' in analysis_df.columns else None,
                color_discrete_map=TYPE_COLORS
            )
            
            fig_scatter.update_layout(
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black', size=12),
                title=dict(font=dict(color='black', size=16)),
                xaxis=dict(title='ART Cycles', title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title='Success Rate (%)', title_font=dict(color='black'), tickfont=dict(color='black')),
                showlegend=True
            )
            
            # Add annotation for bubble size
            if 'Opening_Year' in analysis_df.columns:
                fig_scatter.add_annotation(
                    text="Bubble size = Years in operation",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color='gray')
                )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Success rate vs. cycle volume data not available")
    
    # Second row - Full width regression plot
    if 'Years_Since_Opening' in analysis_df.columns and 'ART_Cycles' in analysis_df.columns:
        # Filter out any invalid years
        regression_df = analysis_df[analysis_df['Years_Since_Opening'] > 0].copy()
        
        if not regression_df.empty:
            # Create scatter plot without trendline if statsmodels is not available
            fig_regression = px.scatter(
                regression_df,
                x='Years_Since_Opening',
                y='ART_Cycles',
                title="<b style='color:black'>ART Cycles vs. Years Since Opening</b>",
                hover_data=['Name', 'Opening_Year'],
                color='Type' if 'Type' in regression_df.columns else None,
                color_discrete_map=TYPE_COLORS
            )
            
            # Try to add trendline manually using numpy polynomial fit
            try:
                import numpy as np
                x = regression_df['Years_Since_Opening'].values
                y = regression_df['ART_Cycles'].values
                
                # Fit a polynomial of degree 2
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                
                # Generate smooth x values for the trend line
                x_trend = np.linspace(x.min(), x.max(), 100)
                y_trend = p(x_trend)
                
                # Add trendline
                fig_regression.add_scatter(
                    x=x_trend, 
                    y=y_trend, 
                    mode='lines',
                    name='Trend',
                    line=dict(color=BRAND_COLORS['primary'], width=3),
                    showlegend=False
                )
            except Exception as e:
                st.info("Trendline calculation skipped")
            
            fig_regression.update_layout(
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black', size=12),
                title=dict(font=dict(color='black', size=16)),
                xaxis=dict(title='Years Since Opening', title_font=dict(color='black'), tickfont=dict(color='black')),
                yaxis=dict(title='ART Cycles', title_font=dict(color='black'), tickfont=dict(color='black'))
            )
            
            st.plotly_chart(fig_regression, use_container_width=True)
    else:
        st.info("Opening year data not available for time-based analysis")

def create_data_table(df):
    """Create an enhanced data table with filtering"""
    
    if df.empty:
        st.warning("No data available")
        return
    
    # Add filters with improved styling
    st.markdown(f"""
    <div style="background: #F8F9FA; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border: 1px solid #E5E7EB;">
        <h4 style="color: {BRAND_COLORS['text']}; margin-bottom: 1rem;">Filter Options</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Type' in df.columns:
            type_filter = st.multiselect(
                "Filter by Type:",
                options=df['Type'].unique(),
                default=df['Type'].unique()
            )
            df_filtered = df[df['Type'].isin(type_filter)]
        else:
            df_filtered = df
    
    with col2:
        if 'Location' in df.columns:
            location_filter = st.multiselect(
                "Filter by Location:",
                options=df['Location'].unique(),
                default=df['Location'].unique()
            )
            df_filtered = df_filtered[df_filtered['Location'].isin(location_filter)]
    
    with col3:
        if 'Success_Rate' in df.columns:
            min_success = st.slider(
                "Minimum Success Rate:",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0
            )
            df_filtered = df_filtered[df_filtered['Success_Rate'] >= min_success]
    
    # Display filtered data count
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {BRAND_COLORS['light']} 0%, #ffffff 100%); 
                padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid {BRAND_COLORS['primary']};">
        <p style="margin: 0; color: {BRAND_COLORS['text']};">
            Showing <b>{len(df_filtered)}</b> of <b>{len(df)}</b> records
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display filtered data
    st.dataframe(
        df_filtered.style.background_gradient(
            subset=['ART_Cycles', 'Deliveries', 'Success_Rate'] if all(col in df_filtered.columns for col in ['ART_Cycles', 'Deliveries', 'Success_Rate']) else [],
            cmap='Blues'
        ),
        use_container_width=True,
        height=500
    )
    
    # Download button for filtered data
    if not df_filtered.empty:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"fertility_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="South Florida Fertility Market Analysis",
        page_icon="🏥",
        layout="wide"
    )
    
    # Set plotly theme to white background
    import plotly.io as pio
    pio.templates.default = "plotly_white"
    
    # Enhanced CSS styling
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main app background */
    .main {{
        background-color: #FFFFFF !important;
    }}
    
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    
    /* Remove all colored backgrounds */
    section[data-testid="stSidebar"] > div {{
        background-color: #F0F2F6 !important;
    }}
    
    .main > div {{
        background-color: #FFFFFF !important;
    }}
    
    .block-container {{
        background-color: #FFFFFF !important;
    }}
    
    /* Fix for the pink/red background issue */
    .element-container {{
        background-color: #FFFFFF !important;
    }}
    
    div[data-testid="stVerticalBlock"] {{
        background-color: #FFFFFF !important;
    }}
    
    div[data-testid="stHorizontalBlock"] {{
        background-color: #FFFFFF !important;
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: #FFFFFF !important;
    }}
    
    /* Remove pink from metric containers */
    div[data-testid="metric-container"] {{
        background-color: #FFFFFF !important;
    }}
    
    /* Fix iframe backgrounds (for maps) */
    iframe {{
        background-color: #FFFFFF !important;
        border: none !important;
    }}
    
    .stIFrame {{
        background-color: #FFFFFF !important;
    }}
    
    /* Fix plotly backgrounds */
    .js-plotly-plot {{
        background-color: #FFFFFF !important;
    }}
    
    .plotly {{
        background-color: #FFFFFF !important;
    }}
    
    /* Tab styling */
    .stTabs {{
        background-color: #FFFFFF !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: #FFFFFF !important;
        padding: 0.5rem 0;
        border-bottom: 2px solid {BRAND_COLORS['primary']} !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        padding: 0 24px;
        background-color: {BRAND_COLORS['primary']};
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {BRAND_COLORS['primary']};
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(0, 134, 163, 0.3);
    }}
    
    .stTabs [aria-selected="false"] {{
        background-color: {BRAND_COLORS['primary']};
        opacity: 0.7;
    }}
    
    /* Sidebar text visibility */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stText {{
        color: #262730 !important;
    }}
    
    /* Remove any remaining backgrounds */
    .css-1y4p8pa {{
        background-color: #FFFFFF !important;
    }}
    
    .css-1dp5vir {{
        background-color: #FFFFFF !important;
    }}
    
    /* AGGRESSIVE PINK REMOVAL */
    .stException {{
        display: none !important;
    }}
    
    div[data-testid="stException"] {{
        display: none !important;
    }}
    
    .stAlert {{
        background-color: #FFFFFF !important;
    }}
    
    /* Remove all possible pink/red backgrounds */
    [style*="background-color: rgb(255, 43, 43)"] {{
        background-color: #FFFFFF !important;
    }}
    
    [style*="background-color: rgb(255, 75, 75)"] {{
        background-color: #FFFFFF !important;
    }}
    
    [style*="background: rgb(255"] {{
        background: #FFFFFF !important;
    }}
    
    /* Override inline styles for error messages */
    div[kind="error"] {{
        background-color: #FFFFFF !important;
    }}
    
    /* Metric card styles */
    .metric-card {{
        background: linear-gradient(135deg, {BRAND_COLORS['light']} 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {BRAND_COLORS['primary']};
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        border: 1px solid #E5E7EB;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    
    .metric-card-top {{
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        border: 1px solid #F59E0B;
    }}
    
    .metric-card-top:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.2);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {BRAND_COLORS['primary']};
        line-height: 1;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value-top {{
        font-size: 2.5rem;
        font-weight: 700;
        color: #92400E;
        line-height: 1;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {BRAND_COLORS['text']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .main-header {{
        background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        position: relative;
    }}
    
    .logo-container {{
        position: absolute;
        left: 2rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 10;
    }}
    
    .title-container {{
        flex-grow: 1;
        text-align: center;
        padding: 0 200px;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and centered title
    try:
        # Load and encode the logo
        import base64
        with open("procreate_logo.png", "rb") as logo_file:
            logo_base64 = base64.b64encode(logo_file.read()).decode()
        
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 100px; max-width: 250px; object-fit: contain;">'
        
    except FileNotFoundError:
        st.sidebar.warning("Logo file 'procreate_logo.png' not found. Please ensure it's in the same directory as your script.")
        # Fallback to text placeholder
        logo_html = '<div style="width: 200px; height: 100px; background: rgba(255,255,255,0.2); border-radius: 8px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.3);"><span style="color: white; font-size: 0.8rem;">Logo</span></div>'
    
    st.markdown(f"""
    <div class="main-header">
        <div class="logo-container">
            {logo_html}
        </div>
        <div class="title-container">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; line-height: 1.2;">
                South Florida Fertility Market<br>
                <span style="font-size: 1.8rem;">Strategic Attractiveness & Competitive Outlook Analysis</span>
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and process data
    main_df, summary_df = load_data()
    
    if main_df.empty:
        st.error("❌ No data loaded. Please ensure your CSV file is in the correct directory.")
        st.stop()
    
    # Process main data for charts and maps
    processed_df = process_data(main_df)
    
    if processed_df.empty:
        st.error("❌ No data available after processing.")
        st.stop()
    
    # Calculate statistics from summary sheet (preferred) or main data (fallback)
    if summary_df is not None and not summary_df.empty:
        stats = calculate_statistics_from_summary(summary_df, processed_df)
    else:
        stats = calculate_statistics_from_main_data(processed_df)
    
    # Display KPIs
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%); padding: 1rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <div style="text-align: center; width: 100%;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">Market Snapshot</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # First row - General market data
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
            <div class="metric-label">Total ART Cycles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['pregnancies']:,}</div>
            <div class="metric-label">Total Pregnancies</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row - Deliveries and Success Rate
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['deliveries']:,}</div>
            <div class="metric-label">Total Deliveries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:  # Center column
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_success_rate']:.1f}%</div>
            <div class="metric-label">Market Avg Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top clinic divider and info
    if stats['top_clinic_name'] != 'N/A':
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); padding: 1rem; border-radius: 8px; margin: 2rem 0 1rem 0; border-left: 4px solid #F59E0B; text-align: center;">
            <h3 style="font-size: 1.25rem; font-weight: 600; color: #F59E0B; margin: 0;">🏆 Top Performing Clinic: {stats['top_clinic_name']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Second row - Top clinic specific data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-top">
                <div class="metric-value-top">{stats['top_clinic_cycles']:,}</div>
                <div class="metric-label">ART Cycles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-top">
                <div class="metric-value-top">{stats['top_clinic_success_rate']:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate market share if possible
            market_share = (stats['top_clinic_cycles'] / stats['art_cycles'] * 100) if stats['art_cycles'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card-top">
                <div class="metric-value-top">{market_share:.1f}%</div>
                <div class="metric-label">Market Share</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create tabs with enhanced content
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📋 Data Explorer", "🔧 Debug Info"])
    
    with tab1:
        # Geographic Distribution header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%); padding: 1rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
            <div style="text-align: center; width: 100%;">
                <h3 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Geographic Distribution</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Map section
        create_map(processed_df)
        
        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts section
        create_charts(processed_df)
    
    with tab2:
        # Style the data explorer section
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%); padding: 1rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
            <div style="text-align: center; width: 100%;">
                <h3 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Data Explorer</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        create_data_table(processed_df)
    
    with tab3:
        # Style the debug info section
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {BRAND_COLORS['primary']} 0%, {BRAND_COLORS['secondary']} 100%); padding: 1rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
            <div style="text-align: center; width: 100%;">
                <h3 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Debug Information</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: #F8F9FA; padding: 1.5rem; border-radius: 8px; border-left: 4px solid {BRAND_COLORS['primary']};">
                <h4 style="color: {BRAND_COLORS['text']}; margin-bottom: 1rem;">Data Info</h4>
                <p style="color: {BRAND_COLORS['text']}; margin: 0.5rem 0;">
                    <b>Shape:</b> {processed_df.shape[0]} rows × {processed_df.shape[1]} columns<br>
                    <b>Memory usage:</b> {processed_df.memory_usage(deep=True).sum() / 1024:.1f} KB
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Data types:**")
            st.dataframe(processed_df.dtypes, use_container_width=True)
        
        with col2:
            missing_data = processed_df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            st.markdown(f"""
            <div style="background: #F8F9FA; padding: 1.5rem; border-radius: 8px; border-left: 4px solid {BRAND_COLORS['primary']};">
                <h4 style="color: {BRAND_COLORS['text']}; margin-bottom: 1rem;">Missing Values</h4>
                <p style="color: {BRAND_COLORS['text']};">
                    {f"Found {len(missing_data)} columns with missing values" if len(missing_data) > 0 else "No missing values found!"}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if len(missing_data) > 0:
                st.dataframe(missing_data, use_container_width=True)
        
        # Show raw data sample
        st.markdown(f"""
        <div style="margin-top: 2rem;">
            <h4 style="color: {BRAND_COLORS['text']};">Raw Data Sample</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(processed_df.head(), use_container_width=True)
        
        # Show column statistics
        if not processed_df.empty:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown(f"""
                <div style="margin-top: 2rem;">
                    <h4 style="color: {BRAND_COLORS['text']};">Numeric Column Statistics</h4>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(processed_df[numeric_cols].describe(), use_container_width=True)

if __name__ == "__main__":
    main()