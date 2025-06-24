import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import DataProcessor
from ml_models import AadhaarMLModels
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Aadhaar Biometric Update Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    processor = DataProcessor('data/raw/aadhaar_updates.csv')
    raw_data = processor.load_data()
    clean_data = processor.clean_data()
    state_agg, district_agg = processor.get_geo_aggregates()
    ml_data = processor.prepare_ml_data()
    return raw_data, clean_data, state_agg, district_agg, ml_data

raw_data, clean_data, state_agg, district_agg, ml_data = load_data()

# Sidebar filters
st.sidebar.title("Filters")
st.sidebar.markdown("Filter the data to analyze specific segments")

# State filter
all_states = ['All'] + sorted(clean_data['State'].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", all_states)

# District filter (dynamic based on state)
if selected_state != 'All':
    districts = ['All'] + sorted(clean_data[clean_data['State'] == selected_state]['District'].unique().tolist())
else:
    districts = ['All']
selected_district = st.sidebar.selectbox("Select District", districts)

# Age group filter
age_group = st.sidebar.radio("Select Age Group", ['All', '5-17', '17+'])

# Date range filter (if Date column exists)
if 'Date' in clean_data.columns:
    min_date = clean_data['Date'].min()
    max_date = clean_data['Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Apply filters
filtered_data = clean_data.copy()
    
if selected_state != 'All':
    filtered_data = filtered_data[filtered_data['State'] == selected_state]
    
if selected_district != 'All':
    filtered_data = filtered_data[filtered_data['District'] == selected_district]
    
if age_group != 'All':
    if age_group == '5-17':
        filtered_data = filtered_data[['State', 'District', 'Pincode', 'Date', 'Bio_age_5_17']]
    else:
        filtered_data = filtered_data[['State', 'District', 'Pincode', 'Date', 'Bio_age_17+']]

if 'Date' in clean_data.columns and len(date_range) == 2:
    filtered_data = filtered_data[
        (filtered_data['Date'] >= pd.to_datetime(date_range[0])) &
        (filtered_data['Date'] <= pd.to_datetime(date_range[1]))
    ]

# Main dashboard
st.title("Aadhaar Biometric Update Analytics Dashboard")
st.markdown(f"""
    This dashboard provides insights into Aadhaar biometric updates
    across different states, districts, and age groups in India.
""")

# Metrics section
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total_updates = filtered_data['Total_Updates'].sum() if 'Total_Updates' in filtered_data.columns else (
    filtered_data['Bio_age_5_17'].sum() + filtered_data['Bio_age_17+'].sum() if age_group == 'All' else
    filtered_data['Bio_age_5_17'].sum() if age_group == '5-17' else
    filtered_data['Bio_age_17+'].sum()
)

col1.metric("Total Updates", f"{total_updates:,}")

if age_group == 'All':
    pct_5_17 = (filtered_data['Bio_age_5_17'].sum() / total_updates) * 100 if total_updates > 0 else 0
    pct_17_plus = (filtered_data['Bio_age_17+'].sum() / total_updates) * 100 if total_updates > 0 else 0
    col2.metric("5-17 Age Group", f"{pct_5_17:.1f}%")
    col3.metric("17+ Age Group", f"{pct_17_plus:.1f}%")
    
top_district = district_agg.loc[district_agg['Total_Updates'].idxmax(), 'District'] if len(district_agg) > 0 else "N/A"
col4.metric("Top District by Updates", top_district)

# Visualization section
st.subheader("Data Visualizations")

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Geographical Distribution", 
    "Age Group Comparison",
    "Time Series Analysis",
    "Update Density Analysis",
    "Machine Learning Insights"
])

with tab1:
    # Geographical distribution
    st.markdown("### Geographical Distribution of Updates")
    
    if selected_state == 'All':
        # Show state-level map
        fig = px.choropleth(
            state_agg,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey='properties.ST_NM',
            locations='State',
            color='Total_Updates',
            color_continuous_scale='Viridis',
            title='State-wise Distribution of Aadhaar Biometric Updates'
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # State comparison table
        st.markdown("### State-wise Summary")
        st.dataframe(state_agg.sort_values('Total_Updates', ascending=False))
    else:
        # Show district-level data
        district_data = district_agg[district_agg['State'] == selected_state]
        
        fig = px.bar(district_data.sort_values('Total_Updates', ascending=False),
                    x='District', y='Total_Updates',
                    color='Total_Updates',
                    color_continuous_scale='Viridis',
                    title=f'District-wise Distribution in {selected_state}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Pincode-level data
        st.markdown(f"### Pincode-level Data for {selected_state}")
        if selected_district != 'All':
            pincode_data = filtered_data[filtered_data['District'] == selected_district]
        else:
            pincode_data = filtered_data
            
        if 'Total_Updates' in pincode_data.columns:
            st.dataframe(pincode_data.sort_values('Total_Updates', ascending=False))
        else:
            st.dataframe(pincode_data)

with tab2:
    # Age group comparison
    st.markdown("### Age Group Comparison")
    
    if age_group == 'All':
        col1, col2 = st.columns(2)
        
        with col1:
            age_data = filtered_data[['Bio_age_5_17', 'Bio_age_17+']].sum()
            fig = px.pie(age_data, values=age_data.values, names=age_data.index,
                        title='Age Group Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age group by region
            if selected_state == 'All':
                age_by_region = state_agg.copy()
                x_axis = 'State'
            else:
                age_by_region = district_agg[district_agg['State'] == selected_state]
                x_axis = 'District'
            
            fig = px.bar(age_by_region, x=x_axis, y=['Bio_age_5_17', 'Bio_age_17+'],
                        barmode='group', title='Age Group Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        # Age ratio visualization
        st.markdown("### Age Group Ratio Across Regions")
        if selected_state == 'All':
            ratio_data = state_agg.copy()
            ratio_data['Child_Adult_Ratio'] = ratio_data['Bio_age_5_17'] / ratio_data['Bio_age_17+']
            fig = px.bar(ratio_data.sort_values('Child_Adult_Ratio', ascending=False),
                        x='State', y='Child_Adult_Ratio',
                        title='Ratio of Child (5-17) to Adult (17+) Updates by State')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show district-wise ratio for the selected state
            ratio_data = district_agg[district_agg['State'] == selected_state].copy()
            ratio_data['Child_Adult_Ratio'] = ratio_data['Bio_age_5_17'] / ratio_data['Bio_age_17+']
            fig = px.bar(ratio_data.sort_values('Child_Adult_Ratio', ascending=False),
                        x='District', y='Child_Adult_Ratio',
                        title=f'Ratio of Child (5-17) to Adult (17+) Updates by District in {selected_state}')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Age group comparison is only available when 'All' age groups are selected.")

# Replace the entire Time Series Analysis tab (tab3) with this:

with tab3:
    # Time series analysis
    if 'Date' in clean_data.columns:
        st.markdown("### Time Series Analysis")
        
        # Ensure Date is datetime and filter only numeric columns for summing
        time_data = filtered_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(time_data['Date']):
            time_data['Date'] = pd.to_datetime(time_data['Date'])
        
        # Select only numeric columns for aggregation
        numeric_cols = time_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Bio_age_5_17' in numeric_cols and 'Bio_age_17+' in numeric_cols:
            # Group by date and sum only numeric columns
            time_data = time_data.groupby('Date')[numeric_cols].sum().reset_index()
            
            # Melt the data for proper plotting
            melted_data = time_data.melt(id_vars=['Date'], 
                                      value_vars=['Bio_age_5_17', 'Bio_age_17+'],
                                      var_name='Age Group', 
                                      value_name='Updates')
            
            # Line chart for trends

            
            # Monthly patterns
            st.markdown("### Monthly Patterns")
            if 'Month' in filtered_data.columns:
                monthly_data = filtered_data.copy()
                # Ensure we only include numeric columns
                monthly_numeric = monthly_data.select_dtypes(include=[np.number]).columns.tolist()
                monthly_data = monthly_data.groupby('Month')[monthly_numeric].sum().reset_index()
                
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                monthly_data['Month'] = pd.Categorical(monthly_data['Month'], 
                                                     categories=month_order, 
                                                     ordered=True)
                monthly_data = monthly_data.sort_values('Month')
                
                # Melt for proper plotting
                melted_monthly = monthly_data.melt(id_vars=['Month'],
                                                 value_vars=['Bio_age_5_17', 'Bio_age_17+'],
                                                 var_name='Age Group',
                                                 value_name='Updates')
                
                fig = px.bar(melted_monthly, 
                            x='Month', 
                            y='Updates',
                            color='Age Group',
                            barmode='group', 
                            title='Monthly Update Patterns')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required numeric columns (Bio_age_5_17, Bio_age_17+) not found for time series analysis")
    else:
        st.warning("Date information not available in the dataset")

with tab4:
    # Update density analysis
    st.markdown("### Update Density Analysis")
    st.markdown("""
        This analysis shows the average number of updates per pincode,
        which helps identify areas with concentrated update activity.
    """)
    
    if selected_state == 'All':
        # State-level density
        fig = px.bar(state_agg.sort_values('Updates_Per_Pincode', ascending=False),
                    x='State', y='Updates_Per_Pincode',
                    color='Updates_Per_Pincode',
                    color_continuous_scale='Viridis',
                    title='Average Updates per Pincode by State')
        st.plotly_chart(fig, use_container_width=True)
    else:
        # District-level density
        district_data = district_agg[district_agg['State'] == selected_state]
        fig = px.bar(district_data.sort_values('Updates_Per_Pincode', ascending=False),
                    x='District', y='Updates_Per_Pincode',
                    color='Updates_Per_Pincode',
                    color_continuous_scale='Viridis',
                    title=f'Average Updates per Pincode in {selected_state}')
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of update density
    st.markdown("### Update Density Heatmap")
    
    if selected_state == 'All':
        # State-level heatmap
        heat_data = state_agg.sort_values('Total_Updates', ascending=False)
        fig = px.imshow(heat_data[['Bio_age_5_17', 'Bio_age_17+']].T,
                       labels=dict(x="State", y="Age Group", color="Updates"),
                       x=heat_data['State'],
                       y=['5-17', '17+'],
                       title='Update Density by State and Age Group')
    else:
        # District-level heatmap
        heat_data = district_agg[district_agg['State'] == selected_state].sort_values('Total_Updates', ascending=False)
        fig = px.imshow(heat_data[['Bio_age_5_17', 'Bio_age_17+']].T,
                       labels=dict(x="District", y="Age Group", color="Updates"),
                       x=heat_data['District'],
                       y=['5-17', '17+'],
                       title=f'Update Density in {selected_state} by District and Age Group')
    
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    # Machine learning insights
    st.markdown("### Machine Learning Insights")
    st.markdown("""
        This section provides predictive analytics on Aadhaar update patterns using various machine learning models.
        The models predict whether a region will have high or low update volume based on geographical patterns.
    """)
    
    if st.button("Run Machine Learning Analysis"):
        with st.spinner("Training models... This may take a few minutes"):
            ml_analysis = AadhaarMLModels(ml_data)
            results = ml_analysis.run_all_models()
            
            st.markdown("### Model Performance Comparison")
            
            # Create performance comparison
            performance_data = []
            for model_name, result in results.items():
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy']
                })
            
            performance_df = pd.DataFrame(performance_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Accuracy Comparison**")
                fig = px.bar(performance_df, x='Model', y='Accuracy',
                            color='Accuracy', color_continuous_scale='Viridis',
                            title='Model Accuracy Scores')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Best Performing Model Details**")
                best_model = performance_df.loc[performance_df['Accuracy'].idxmax()]
                st.write(f"**Model:** {best_model['Model']}")
                st.write(f"**Accuracy:** {best_model['Accuracy']:.2%}")
                
                if best_model['Model'] == 'Deep Learning':
                    history = results['Deep Learning']['history']
                    fig, ax = plt.subplots()
                    ax.plot(history.history['accuracy'], label='Training Accuracy')
                    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax.set_title('Deep Learning Training Progress')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.markdown("**Classification Report:**")
                    st.text(results[best_model['Model']]['report'])

            # --- This block must be inside the button/with spinner block ---
            st.markdown("**All Model Accuracies and Reports**")
            for model_name, result in results.items():
                st.write(f"**{model_name}**")
                st.write(f"Accuracy: {result['accuracy']:.2%}")
                if model_name != 'Deep Learning':
                    st.text(result['report'])
                st.markdown("---")

            # Add warning about ML results and RapidMiner
            st.warning(
                "⚠️ **Note:** The machine learning results shown above may not reflect real-world predictive accuracy, "
                "as the features and target in this dataset are closely related. "
                "For a more realistic evaluation, please export the data and use a dedicated ML tool like RapidMiner. "
                "If you need to demonstrate true model performance, run the analysis in RapidMiner and review the results there."
            )

# Data download section
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Export")

if st.sidebar.button("Download Filtered Data as CSV"):
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_aadhaar_data.csv",
        mime="text/csv"
    )

# About section
st.sidebar.markdown("---")
st.sidebar.markdown("""
    **About This Dashboard**  
    This dashboard analyzes Aadhaar biometric update patterns across India.  
    Data Source: Public dataset on Aadhaar biometric updates  
    Developed with Python, Streamlit, and Plotly  
""")