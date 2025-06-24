import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        """Load raw data from CSV"""
        return pd.read_csv(self.file_path)
    
    def clean_data(self):
        """Clean and prepare the raw data"""
        raw_data = self.load_data()
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Bio_age_5_17', 'Bio_age_17+']
        for col in numeric_cols:
            if col in raw_data.columns:
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(0)
        
        # Calculate total updates
        if all(col in raw_data.columns for col in numeric_cols):
            raw_data['Total_Updates'] = raw_data['Bio_age_5_17'] + raw_data['Bio_age_17+']
        elif 'Total_Updates' in raw_data.columns:
            raw_data['Total_Updates'] = pd.to_numeric(raw_data['Total_Updates'], errors='coerce').fillna(0)
        
        # Handle date processing
        if 'Date' in raw_data.columns:
            raw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
            raw_data = raw_data.dropna(subset=['Date'])  # Remove rows with invalid dates
            raw_data['Year'] = raw_data['Date'].dt.year
            raw_data['Month'] = raw_data['Date'].dt.month_name()
        
        # Create update volume categories for ML
        if 'Total_Updates' in raw_data.columns:
            raw_data['Update_Volume_Category'] = pd.qcut(
                raw_data['Total_Updates'],
                q=3,
                labels=['Low', 'Medium', 'High']
            )
        
        # Calculate updates per pincode (density metric)
        if 'Pincode' in raw_data.columns:
            pincode_counts = raw_data['Pincode'].value_counts().reset_index()
            pincode_counts.columns = ['Pincode', 'Pincode_Count']
            raw_data = raw_data.merge(pincode_counts, on='Pincode', how='left')
            if 'Total_Updates' in raw_data.columns:
                raw_data['Updates_Per_Pincode'] = raw_data['Total_Updates'] / raw_data['Pincode_Count']
        
        return raw_data
    
    def get_geo_aggregates(self):
        """Generate geographical aggregates"""
        clean_data = self.clean_data()
        
        # State-level aggregates
        state_agg = clean_data.groupby('State').agg({
            'Bio_age_5_17': 'sum',
            'Bio_age_17+': 'sum',
            'Total_Updates': 'sum',
            'Pincode': 'nunique'
        }).reset_index()
        state_agg['Updates_Per_Pincode'] = state_agg['Total_Updates'] / state_agg['Pincode']
        
        # District-level aggregates
        district_agg = clean_data.groupby(['State', 'District']).agg({
            'Bio_age_5_17': 'sum',
            'Bio_age_17+': 'sum',
            'Total_Updates': 'sum',
            'Pincode': 'nunique'
        }).reset_index()
        district_agg['Updates_Per_Pincode'] = district_agg['Total_Updates'] / district_agg['Pincode']
        
        return state_agg, district_agg
    
    def prepare_ml_data(self):
        """Prepare data for machine learning"""
        clean_data = self.clean_data()
        ml_data = clean_data.copy()

        # Remove all columns directly related to the target
        drop_cols = [
            'Bio_age_5_17', 'Bio_age_17+', 'Total_Updates', 'Update_Volume_Category',
            'Updates_Per_Pincode', 'Pct_5_17', 'Pct_17+', 'Pincode_Count'
        ]
        drop_cols = [col for col in drop_cols if col in ml_data.columns]
        ml_data = ml_data.drop(columns=drop_cols)

        # Encode categorical features (State, District)
        for col in ['State', 'District']:
            if col in ml_data.columns:
                ml_data[col] = ml_data[col].astype('category').cat.codes

        # Keep only rows with a valid target
        ml_data = ml_data.dropna(subset=['State', 'District', 'Update_Volume_Category'])

        return ml_data