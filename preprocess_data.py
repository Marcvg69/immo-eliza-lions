# input file_path = 'data/raw/properties06191148_modified.csv'
# output_file_path_subset = 'data/cleaned/properties06191148_cleaned.csv' 
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset (make sure the path is correct)
file_path = 'data/raw/properties06191148_modified.csv'  # Replace with the correct path
data = pd.read_csv(file_path)

# Map the 'city' column to the 'Region' column using the provided mapping
belgium_regions = {
    'Antwerp': 'Flanders',
    'Limburg': 'Flanders',
    'East Flanders': 'Flanders',
    'Flemish Brabant': 'Flanders',
    'West Flanders': 'Flanders',
    'Hainaut': 'Wallonia',
    'Walloon Brabant': 'Wallonia',
    'Namur': 'Wallonia',
    'Liège': 'Wallonia',
    'Luxembourg': 'Wallonia',
    'Brussels': 'Brussels-Capital'
}

# Map cities to regions
data['Region'] = data['city'].map(belgium_regions)
data['Region'].fillna('Unknown', inplace=True)  # Fill missing values in 'Region' with 'Unknown'

# Handle missing values for numeric columns (using median imputation)
numeric_cols = ['living area(m²)', 'ground area(m²)', 'bedroom', 'bathroom', 'year built', 'mobiscore', 'EPC(kWh/m²)', 'price']
imputer = SimpleImputer(strategy='median')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Handle missing values for categorical columns (using 'Unknown' as placeholder)
categorical_cols = ['type', 'city', 'Region']
imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Normalize numeric columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Label encode categorical columns like 'type' and 'city'
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['city'] = label_encoder.fit_transform(data['city'])

# Save the processed dataset to the specified output file path
output_file_path = 'data/cleaned/properties06191148_cleaned.csv'  # Updated output file path
data.to_csv(output_file_path, index=False)

# Show the first few rows of the final processed data and provide the output file path
print(data.head())
