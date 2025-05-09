import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Istanbul Urban Heat Island Analysis",
    page_icon="🌡️",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    """
    Load data from CSV file.
    The @st.cache_data decorator prevents reloading on each interaction.
    
    Returns:
    pd.DataFrame: DataFrame with UHI data
    """
    try:
        # Try to load real data
        df = pd.read_csv("mock_data.csv")
        return df
    except:
        # If file not found, show error and use a very basic fallback
        st.error("Error: Could not load data file. Please make sure 'mock_data.csv' exists in the same directory as this app.")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['lat', 'lon', 'NDVI', 'NDBI', 'year', 'UHI_class'])

# Main title
st.title("Istanbul Urban Heat Island Analysis Dashboard")
st.write("This dashboard visualizes Urban Heat Island (UHI) severity in Istanbul using satellite-derived features from August of selected years.")

# Load the data
df = load_data()

# Check if data was loaded successfully
if not df.empty:
    # ------- SIDEBAR -------
    st.sidebar.title("Filters")
    
    # Year filter - specifically August of each year
    available_years = sorted(df['year'].unique().tolist())
    selected_year = st.sidebar.selectbox(
        "Select Year (August)", 
        available_years,
        index=len(available_years)-1  # Default to most recent year
    )
    
    # Feature type filter
    feature_options = ['NDVI', 'NDBI', 'UHI_class']
    selected_feature = st.sidebar.selectbox("Select Feature to Visualize", feature_options)
    
    # Filter data based on year selection
    filtered_df = df[df['year'] == selected_year].copy()
    
    # Display basic information about the selected data
    st.write(f"Showing {selected_feature} data for August {selected_year}")
    st.write(f"Number of data points: {len(filtered_df)}")
    
    # ------- MAIN CONTENT -------
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Spatial Distribution Map")
        
        # Center map on Istanbul
        istanbul_center = [41.0082, 28.9784]
        
        # Create a base map centered on Istanbul
        m = folium.Map(location=istanbul_center, zoom_start=10)
        
        # Helper function to determine circle color based on feature value
        def get_color(feature_value, feature_type):
            """
            Returns a color based on the feature value.
            
            Parameters:
            feature_value: The value of the selected feature
            feature_type: The type of feature (NDVI, NDBI, or UHI_class)
            
            Returns:
            str: Hexadecimal color code
            """
            if feature_type == 'NDVI':
                # NDVI: Higher values (greener) use green colors
                # NDVI typically ranges from -1 to 1
                if feature_value < 0:
                    return '#d73027'  # Red for negative values (water, bare soil)
                elif feature_value < 0.2:
                    return '#fee08b'  # Yellow for low vegetation
                elif feature_value < 0.5:
                    return '#66bd63'  # Light green for moderate vegetation
                else:
                    return '#1a9850'  # Dark green for high vegetation
            
            elif feature_type == 'NDBI':
                # NDBI: Higher values (more built-up) use red colors
                # NDBI typically ranges from -1 to 1
                if feature_value < -0.3:
                    return '#1a9850'  # Green for water bodies (negative values)
                elif feature_value < 0:
                    return '#66bd63'  # Light green for vegetation
                elif feature_value < 0.3:
                    return '#fee08b'  # Yellow for moderate built-up
                else:
                    return '#d73027'  # Red for high built-up
            
            else:  # UHI_class
                # UHI class: Higher values are worse (1-5)
                colors = {
                    1: '#1a9850',  # Green (Low)
                    2: '#66bd63',  # Light green (Moderate Low)
                    3: '#fee08b',  # Yellow (Moderate)
                    4: '#fc8d59',  # Orange (High)
                    5: '#d73027'   # Red (Very High)
                }
                return colors.get(int(feature_value), '#808080')  # Gray for unknown values
        
        # Add points to the map
        for idx, row in filtered_df.iterrows():
            # Get color based on feature value
            color = get_color(row[selected_feature], selected_feature)
            
            # Create popup text with all relevant information
            popup_text = f"""
            <b>Location:</b> {row['lat']:.4f}, {row['lon']:.4f}<br>
            <b>NDVI:</b> {row['NDVI']:.2f}<br>
            <b>NDBI:</b> {row['NDBI']:.2f}<br>
            <b>UHI Class:</b> {int(row['UHI_class'])}<br>
            <b>Year:</b> August {int(row['year'])}
            """
            
            # Add circle marker to map
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
        
        # Add a legend to the map
        if selected_feature == 'NDVI':
            title = 'NDVI (Vegetation Index)'
            legend_items = [
                ('< 0 (Water/Soil)', '#d73027'),
                ('0 - 0.2 (Low Veg.)', '#fee08b'),
                ('0.2 - 0.5 (Mod. Veg.)', '#66bd63'),
                ('> 0.5 (High Veg.)', '#1a9850')
            ]
        elif selected_feature == 'NDBI':
            title = 'NDBI (Built-up Index)'
            legend_items = [
                ('< -0.3 (Water)', '#1a9850'),
                ('-0.3 - 0 (Vegetation)', '#66bd63'),
                ('0 - 0.3 (Low Built-up)', '#fee08b'),
                ('> 0.3 (High Built-up)', '#d73027')
            ]
        else:  # UHI_class
            title = 'UHI Severity Class'
            legend_items = [
                ('1 - Low', '#1a9850'),
                ('2 - Moderate Low', '#66bd63'),
                ('3 - Moderate', '#fee08b'),
                ('4 - High', '#fc8d59'),
                ('5 - Very High', '#d73027')
            ]
        
        # Display the map in Streamlit
        folium_static(m)
        
        # Add a simple legend below the map
        st.write(f"**Legend for {title}:**")
        for label, color in legend_items:
            st.markdown(
                f'<span style="background-color:{color}; border-radius:50%; display:inline-block; width:15px; height:15px; margin-right:5px"></span> {label}',
                unsafe_allow_html=True
            )
    
    with col2:
        # Display statistics for the selected feature
        st.subheader(f"Statistics for {selected_feature}")
        
        if len(filtered_df) > 0:
            st.write(f"Average: {filtered_df[selected_feature].mean():.3f}")
            st.write(f"Minimum: {filtered_df[selected_feature].min():.3f}")
            st.write(f"Maximum: {filtered_df[selected_feature].max():.3f}")
            
            # Create a histogram of the selected feature
            st.subheader(f"Distribution of {selected_feature}")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if selected_feature == 'UHI_class':
                # For UHI class, use a bar chart since it's categorical
                value_counts = filtered_df['UHI_class'].value_counts().sort_index()
                ax.bar(
                    value_counts.index,
                    value_counts.values,
                    color=['#1a9850', '#66bd63', '#fee08b', '#fc8d59', '#d73027']
                )
                ax.set_xlabel('UHI Class')
                ax.set_ylabel('Count')
                ax.set_xticks(list(range(1, 6)))
                ax.set_xticklabels(['1 - Low', '2 - Mod. Low', '3 - Moderate', '4 - High', '5 - Very High'])
            else:
                # For continuous variables, use a histogram
                ax.hist(filtered_df[selected_feature], bins=15, alpha=0.7)
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
            
            st.pyplot(fig)
            
            # Display the data table
            st.subheader("Data Sample")
            st.dataframe(filtered_df[['lat', 'lon', 'NDVI', 'NDBI', 'UHI_class']].head(10))
    
    # Additional section for temporal analysis
    st.subheader("Temporal Analysis")
    
    # Group by year and calculate average values for each feature
    temporal_data = df.groupby('year').agg({
        'NDVI': 'mean',
        'NDBI': 'mean',
        'UHI_class': 'mean'
    }).reset_index()
    
    # Create a line chart for temporal trends
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot lines for each feature
    ax.plot(temporal_data['year'], temporal_data['NDVI'], marker='o', label='NDVI', color='green')
    ax.plot(temporal_data['year'], temporal_data['NDBI'], marker='s', label='NDBI', color='red')
    ax2 = ax.twinx()  # Create a second y-axis
    ax2.plot(temporal_data['year'], temporal_data['UHI_class'], marker='^', label='UHI Class', color='orange', linestyle='--')
    
    # Set labels and legend
    ax.set_xlabel('Year (August)')
    ax.set_ylabel('NDVI / NDBI')
    ax2.set_ylabel('UHI Class')
    ax.set_xticks(temporal_data['year'])
    
    # Add a title
    ax.set_title('Change in Urban Heat Island Indicators Over Time (August values)')
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Show the plot
    st.pyplot(fig)
    
    # Add interpretation
    st.write("""
    **Temporal Trend Interpretation:**
    - This chart shows how NDVI, NDBI, and UHI class values have changed over the selected years (August data).
    - A decreasing NDVI trend would indicate loss of vegetation over time.
    - An increasing NDBI trend would indicate more urbanization/built-up areas over time.
    - An increasing UHI class trend would indicate worsening urban heat island effects over time.
    """)
    
    # Additional section for correlations
    st.subheader("Feature Correlations")
    
    if len(filtered_df) > 0:
        # Create a correlation matrix
        correlation = filtered_df[['NDVI', 'NDBI', 'UHI_class']].corr()
        
        # Display the correlation matrix
        st.write("Correlation between features:")
        st.write(correlation.round(2))
        
        # Add interpretation
        st.write("""
        **Interpretation:**
        - NDVI and NDBI typically have a negative correlation (more vegetation = less built-up area)
        - UHI class typically has a negative correlation with NDVI (more vegetation = lower heat island effect)
        - UHI class typically has a positive correlation with NDBI (more built-up area = higher heat island effect)
        """)
    
    # Add information about the data and project
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About this Dashboard")
    st.sidebar.markdown("""
    This dashboard visualizes Urban Heat Island (UHI) severity in Istanbul using satellite-derived features.
    
    **Data details:**
    - Data from August of years 2000, 2005, 2010, 2015, 2020, and 2024
    - August chosen as the hottest month (most representative for UHI effects)
    
    **Features explanation:**
    - **NDVI**: Normalized Difference Vegetation Index
      - Measures vegetation density (-1 to +1)
      - Higher values indicate more vegetation
    - **NDBI**: Normalized Difference Built-up Index
      - Measures built-up areas (-1 to +1)
      - Higher values indicate more urbanization
    - **UHI_class**: Urban Heat Island classification
      - Ranges from 1 (Low) to 5 (Very High)
      - Indicates severity of urban heat island effect
    """)
    
    # Add a download button for the data
    st.sidebar.markdown("---")
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name=f"uhi_istanbul_august_{selected_year}_{selected_feature}.csv",
        mime="text/csv",
    )

else:
    st.error("No data available. Please check that your data file exists and has the correct format.")