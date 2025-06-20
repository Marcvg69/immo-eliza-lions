import pandas as pd

# Load the original dataset
file_path = 'data/raw/properties06191148_modified.csv'
df = pd.read_csv(file_path)

# 1. Drop irrelevant columns
df.drop(['street', 'number', 'url'], axis=1, inplace=True)

# 2. Handle missing values
# Convert 'renovation obligation' column to boolean type before filling NaN values
df['renovation obligation'] = df['renovation obligation'].astype(bool)
df['renovation obligation'] = df['renovation obligation'].fillna(False)

# 3. Convert 'garage' column: 1.0 -> True and NaN -> False
df['garage'] = df['garage'].apply(lambda x: True if x == 1.0 else False)

# 4. Mapping postal codes to specific regions based on the first digits
def map_postcode_to_region(postcode):
    # Get the first 2 digits of the postal code (left to right)
    postcode_prefix = str(postcode)[:2]

    # Region mapping based on the first 2 digits of the postal code
    belgium_regions = {
        # Flanders
        '20': 'Antwerp', '21': 'Antwerp', '22': 'Antwerp', '23': 'Antwerp', '24': 'Antwerp', '25': 'Antwerp', '26': 'Antwerp',  # Antwerp
        '30': 'Limburg', '31': 'Limburg', '32': 'Limburg', '33': 'Limburg', '34': 'Limburg', '35': 'Limburg', '36': 'Limburg',  # Limburg
        '40': 'East Flanders', '41': 'East Flanders', '42': 'East Flanders', '43': 'East Flanders', '44': 'East Flanders', '45': 'East Flanders',  # East Flanders
        '50': 'West Flanders', '51': 'West Flanders', '52': 'West Flanders', '53': 'West Flanders', '54': 'West Flanders', '55': 'West Flanders',  # West Flanders
        '60': 'Flemish Brabant', '61': 'Flemish Brabant', '62': 'Flemish Brabant', '63': 'Flemish Brabant', '64': 'Flemish Brabant',  # Flemish Brabant
        
        # Wallonia
        '70': 'Hainaut', '71': 'Hainaut', '72': 'Hainaut', '73': 'Hainaut', '74': 'Hainaut', '75': 'Hainaut', '76': 'Hainaut',  # Hainaut
        '80': 'Walloon Brabant', '81': 'Walloon Brabant', '82': 'Walloon Brabant', '83': 'Walloon Brabant', '84': 'Walloon Brabant',  # Walloon Brabant
        '90': 'Namur', '91': 'Namur', '92': 'Namur', '93': 'Namur', '94': 'Namur', '95': 'Namur', '96': 'Namur',  # Namur
        '100': 'Liège', '101': 'Liège', '102': 'Liège', '103': 'Liège', '104': 'Liège', '105': 'Liège', '106': 'Liège',  # Liège
        '110': 'Luxembourg', '111': 'Luxembourg', '112': 'Luxembourg', '113': 'Luxembourg', '114': 'Luxembourg',  # Luxembourg
        
        # Brussels-Capital
        '12': 'Brussels', '13': 'Brussels', '14': 'Brussels', '15': 'Brussels', '16': 'Brussels', '17': 'Brussels',  # Brussels
    }

    # Return the corresponding region or 'Unknown' if the prefix is not found
    return belgium_regions.get(postcode_prefix, 'Unknown')

# Apply the mapping function to the 'postcode' column to create a new 'Region' column
df['Region'] = df['postcode'].apply(map_postcode_to_region)

# 5. Save the cleaned data to a new CSV file
output_file = 'data/cleaned/properties06191148_cleaned.csv'
df.to_csv(output_file, index=False)

output_file  # return the path of the cleaned file to check the output
