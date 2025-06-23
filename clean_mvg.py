# === STEP 1: Imports ===
import pandas as pd
import os

# === STEP 2: Load the raw dataset ===
# This loads the CSV file you previously scraped or downloaded.
input_path = "data/raw/immoweb-dataset.csv"
df = pd.read_csv(input_path)

# === STEP 3: Drop irrelevant or low-quality columns ===
# These columns are either not useful for analysis or too incomplete.
columns_to_drop = [
    "Unnamed: 0", "monthlyCost", "hasBalcony", "accessibleDisabledPeople",
    "url", "id", "diningRoomSurface", "streetFacadeWidth", "gardenSurface",
    "roomCount", "kitchenSurface", "livingRoomSurface", "floorCount", "facedeCount",
    "hasAttic", "hasBasement", "hasDressingRoom", "hasDiningRoom",
    "hasLift", "hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels",
    "kitchenType", "hasLivingRoom", "hasGarden", "gardenOrientation",
    "hasAirConditioning", "hasArmoredDoor", "hasVisiophone", "hasOffice",
    "hasSwimmingPool", "hasFireplace", "hasTerrace", "terraceOrientation",
    "terraceSurface", "postCode", "floodZoneType", "landSurface", "toiletCount"
]
# Drop only if the column exists
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# === STEP 4: Clean up text fields and duplicates ===
# Replace empty strings with NaN (standard missing value marker in pandas)
df.replace('', pd.NA, inplace=True)
# Strip leading/trailing whitespace from all text (object) columns ????
# Determine which columns where affected ????
for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str).str.strip()

# Remove exact duplicate rows
df.drop_duplicates(inplace=True)

# === STEP 5: Drop rows with missing price ===
# Price is our target variable — we can’t model or analyse without it.
df = df[df['price'].notna()].copy()

# === STEP 6: Add REGION column ===
# Group provinces into their major Belgian regions (Flanders, Wallonia, Brussels)
belgium_regions = {
    'Antwerp': 'Flanders', 'Limburg': 'Flanders', 'East Flanders': 'Flanders',
    'Flemish Brabant': 'Flanders', 'West Flanders': 'Flanders',
    'Hainaut': 'Wallonia', 'Walloon Brabant': 'Wallonia', 'Namur': 'Wallonia',
    'Liège': 'Wallonia', 'Luxembourg': 'Wallonia',
    'Brussels': 'Brussels'
}
df['region'] = df['province'].map(belgium_regions)

# === STEP 7: Convert categorical strings to numerical values ===
# These mappings convert qualitative labels to ordinal numerical values for modeling.

# Clean EPC scores like 'EPC_SCORE_A' into just 'A'
def extract_epc(value):
    return value.split('_')[-1] if isinstance(value, str) and '_' in value else value

# Define conversion tables
epc_mapping = {'A++': 9, 'A+': 8, 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
condition_mapping = {
    'JUST_RENOVATED': 6, 'AS_NEW': 5, 'GOOD': 4,
    'TO_BE_DONE_UP': 3, 'TO_RENOVATE': 2, 'TO_RESTORE': 1
}
type_mapping = {'APARTMENT': 1, 'HOUSE': 0}

# Apply the conversions
df['epcScore'] = df['epcScore'].apply(extract_epc).map(epc_mapping)
df['buildingCondition'] = df['buildingCondition'].map(condition_mapping)
df['type'] = df['type'].map(type_mapping)

# === STEP 8: Add price per m² metric ===
# Helps standardize property value comparisons across different sizes
df['price_per_m2'] = df['price'] / df['habitableSurface']

# === STEP 9: Save outputs ===
# Create output folder if not present
os.makedirs("data/cleaned", exist_ok=True)

# Save cleaned dataset to main CSV
main_output = "data/cleaned/immoweb-dataset_cleaned_mvg.csv"
df.to_csv(main_output, index=False)
print(f"✅ Cleaned dataset saved to: {main_output}")

# Save mapping tables as separate CSVs for transparency and documentation
pd.DataFrame(list(epc_mapping.items()), columns=["EPC Label", "EPC Score"]).to_csv("data/cleaned/mapping_epcScore.csv", index=False)
pd.DataFrame(list(condition_mapping.items()), columns=["Building Condition", "Score"]).to_csv("data/cleaned/mapping_buildingCondition.csv", index=False)
pd.DataFrame(list(type_mapping.items()), columns=["Property Type", "Code"]).to_csv("data/cleaned/mapping_type.csv", index=False)

print("📄 Mapping tables saved: epcScore, buildingCondition, type.")

# Maybe later to prepare for ML

# === STEP 8: Convert remaining categorical columns to numeric ===
# Many ML models only work with numbers, so we convert text labels using one-hot encoding
# This creates new columns for each category (e.g. province_Liège = 1 if applicable)
#cat_cols = df.select_dtypes(include='object').columns.tolist()
#if cat_cols:
#    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === STEP 10: Drop any rows that still contain missing values ===
# This is a simple, safe strategy when building first models ?????????????? maybe later befor ML....
# df = df.dropna()
